"""The run summary must make silent degradation impossible to miss.

Three shipped failures all looked like successful runs: MFA falling back to
CMUdict timing, Dr.VOT falling back to Praat for every token, and an old build
being installed. The first two are what these tests pin down.
"""
import types

import pytest

from tapa.config import TAPAConfig


class FakePipeline:
    """Just the summary logic, without loading any models."""

    def __init__(self, cfg, mfa_available=True):
        from tapa.pipeline import TAPAPipeline
        self.cfg = cfg
        self.mfa_available = mfa_available
        self._build_run_summary = types.MethodType(
            TAPAPipeline._build_run_summary, self)
        self._report_degradations = types.MethodType(
            TAPAPipeline._report_degradations, self)


def stop_data(methods):
    """{'drvot': 8, 'tapa-fallback': 2} -> nested per-speaker token dict."""
    toks = []
    for method, n in methods.items():
        toks += [{"vot_ms": 20.0, "vot_method": method} for _ in range(n)]
    return {"SPEAKER_00": {"t": toks}}


def averages(n_tokens=10):
    return {"SPEAKER_00": {"i": {"n_tokens": n_tokens}}}


def build(cfg, mfa_available=True, mfa_phones=None, methods=None,
          align_report=None, segments=None):
    p = FakePipeline(cfg, mfa_available)
    return p._build_run_summary(
        "rec.mp3",
        segments or [{"speaker": "SPEAKER_00", "start": 0, "end": 5}],
        [{"word": "hi"}] * 20,
        mfa_phones,
        align_report,
        stop_data(methods or {"tapa-praat": 5}),
        averages(), averages(), averages())


def test_clean_run_reports_no_degradation():
    s = build(TAPAConfig(), mfa_phones=[{"phone": "IY"}] * 30)
    assert s["degradations"] == []
    assert s["actual"]["alignment"].startswith("MFA")


def test_mfa_silent_fallback_is_recorded():
    """MFA present but produced nothing: the exact openfst failure mode."""
    s = build(TAPAConfig(), mfa_available=True, mfa_phones=None)
    assert any("CMUdict" in d for d in s["degradations"])
    assert s["actual"]["alignment"] == "CMUdict proportional timing"


def test_no_mfa_installed_is_not_a_degradation():
    """If MFA was never available, CMUdict is the expected path, not a failure."""
    s = build(TAPAConfig(), mfa_available=False, mfa_phones=None)
    assert s["degradations"] == []


def test_total_drvot_fallback_is_recorded():
    """The shipped failure: all 5537 tokens fell back and the run looked fine."""
    cfg = TAPAConfig(vot_backend="drvot")
    s = build(cfg, mfa_phones=[{"phone": "IY"}], methods={"tapa-fallback": 5537})
    assert any("every one of 5537" in d for d in s["degradations"])
    assert s["actual"]["vot_methods"] == {"tapa-fallback": 5537}


def test_partial_drvot_fallback_flagged_above_threshold():
    cfg = TAPAConfig(vot_backend="drvot")
    s = build(cfg, mfa_phones=[{"phone": "IY"}],
              methods={"drvot": 50, "tapa-fallback": 50})
    assert any("fell back" in d for d in s["degradations"])


def test_small_drvot_fallback_is_tolerated():
    cfg = TAPAConfig(vot_backend="drvot")
    s = build(cfg, mfa_phones=[{"phone": "IY"}],
              methods={"drvot": 98, "tapa-fallback": 2})
    assert s["degradations"] == []


def test_praat_backend_does_not_flag_fallback():
    """vot_backend='tapa' means Praat is the intended method, not a fallback."""
    s = build(TAPAConfig(), mfa_phones=[{"phone": "IY"}],
              methods={"tapa-praat": 200})
    assert s["degradations"] == []


def test_high_oov_share_is_flagged():
    report = {"n_words": 1000, "n_unaligned": 200, "share": 0.20,
              "types": ["2016"], "oov_file": None}
    s = build(TAPAConfig(), mfa_phones=[{"phone": "IY"}], align_report=report)
    assert any("no phoneme alignment" in d for d in s["degradations"])


def test_normal_oov_share_is_not_flagged():
    report = {"n_words": 1000, "n_unaligned": 19, "share": 0.019,
              "types": ["2016"], "oov_file": None}
    s = build(TAPAConfig(), mfa_phones=[{"phone": "IY"}], align_report=report)
    assert s["degradations"] == []


def test_strict_mode_raises_on_degradation():
    cfg = TAPAConfig(strict=True, vot_backend="drvot")
    p = FakePipeline(cfg)
    s = build(cfg, mfa_phones=[{"phone": "IY"}], methods={"tapa-fallback": 100})
    with pytest.raises(RuntimeError, match="strict=True"):
        p._report_degradations(s)


def test_non_strict_mode_warns_but_continues(capsys):
    cfg = TAPAConfig(vot_backend="drvot")
    p = FakePipeline(cfg)
    s = build(cfg, mfa_phones=[{"phone": "IY"}], methods={"tapa-fallback": 100})
    p._report_degradations(s)                      # must not raise
    assert "WARNING" in capsys.readouterr().out


def test_summary_records_requested_versus_actual():
    cfg = TAPAConfig(vot_backend="drvot", num_speakers=2, whisper_model="small.en")
    s = build(cfg, mfa_phones=[{"phone": "IY"}], methods={"drvot": 10})
    assert s["requested"]["vot_backend"] == "drvot"
    assert s["requested"]["num_speakers"] == 2
    assert s["counts"]["vot_tokens"] == 10
    assert s["counts"]["words"] == 20
