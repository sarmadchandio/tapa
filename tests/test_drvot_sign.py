"""Dr.VOT's voicing class must reach the stored value.

Both defects this file guards against shipped in the commit that introduced the
Dr.VOT backend and survived four months, because neither produces a visible
symptom: no crash, no empty column, coverage still reports 100%, and the
per-speaker averages stay in a range that looks like plausible VOT. The only
way to catch them is to assert on the relationship between the class and the
number, and on the clip geometry, directly.
"""

import numpy as np
import pytest

from tapa.config import TAPAConfig
from tapa.drvot import _build_clip_index, _cut_clip, _signed_vot


def test_prevoiced_token_is_stored_negative():
    """NEG_VOT must not arrive as a large positive VOT.

    This is the whole point: a token voiced 100 ms *before* release and one
    voiced 100 ms *after* are opposite ends of the voicing contrast. Storing
    the magnitude maps them onto the same number.
    """
    cfg = TAPAConfig()
    pos = {"vot_ms": 100.0, "vot_class_drvot": "POS_VOT"}
    neg = {"vot_ms": 100.0, "vot_class_drvot": "NEG_VOT"}

    assert _signed_vot(pos, cfg) == 100.0
    assert _signed_vot(neg, cfg) == -100.0
    assert _signed_vot(pos, cfg) != _signed_vot(neg, cfg)


def test_missing_class_is_treated_as_positive():
    """Dr.VOT has renamed its columns across versions; absence must not flip."""
    cfg = TAPAConfig()
    assert _signed_vot({"vot_ms": 42.0, "vot_class_drvot": None}, cfg) == 42.0
    assert _signed_vot({"vot_ms": 42.0, "vot_class_drvot": ""}, cfg) == 42.0


def test_legacy_flag_restores_unsigned_values():
    cfg = TAPAConfig(drvot_signed_vot=False)
    neg = {"vot_ms": 100.0, "vot_class_drvot": "NEG_VOT"}
    assert _signed_vot(neg, cfg) == 100.0


def test_clip_padding_default_stays_small():
    """Dr.VOT analyses one 250 ms window opening 50 ms before the first voicing
    in the clip. Pad far enough back to include the preceding vowel and that
    window locks onto the wrong event while coverage still reads 100%. 150 ms
    was the shipped default and measured d = -0.22 (backwards) against a
    burst-onset reference; 25 ms measured d = +1.39.
    """
    assert TAPAConfig().drvot_clip_pre_ms <= 50.0, (
        "raising drvot_clip_pre_ms points Dr.VOT's analysis window at the "
        "preceding vowel — see the note in tapa/config.py"
    )


def test_clip_starts_close_to_the_closure():
    """The cut itself must honour the padding, so the release is inside the
    window Dr.VOT will actually look at."""
    sr = 16000
    audio = np.zeros(int(2.0 * sr), dtype=np.float32)
    stop_start, vowel_end = 1.000, 1.120

    clip, clip_t0 = _cut_clip(audio, sr, stop_start, vowel_end,
                              pre_ms=25.0, post_ms=150.0)
    assert stop_start - clip_t0 == pytest.approx(0.025, abs=1e-3)
    assert len(clip) / sr == pytest.approx(0.025 + 0.120 + 0.150, abs=2e-3)


def test_clip_index_skips_stops_without_a_following_vowel(tmp_path):
    """VOT is only defined into a following vowel; anything else must not be
    silently measured and averaged in."""
    sr = 16000
    audio = np.zeros(int(3.0 * sr), dtype=np.float32)
    stops = {"SPEAKER_00": [
        {"start": 1.0, "end": 1.08, "ipa": "p", "arpabet": "P",
         "voicing": "voiceless", "place": "bilabial", "word": "pot",
         "following_phone": "AA", "following_start": 1.08, "following_end": 1.22},
        {"start": 2.0, "end": 2.08, "ipa": "p", "arpabet": "P",
         "voicing": "voiceless", "place": "bilabial", "word": "apt",
         "following_phone": "T", "following_start": 2.08, "following_end": 2.20},
    ]}

    index = _build_clip_index(stops, audio, tmp_path, sr, 25.0, 150.0)

    assert [e["word"] for e in index] == ["pot"]
    # One clip on disk, and its name is the key the summary CSV is joined on.
    assert sorted(p.name for p in tmp_path.glob("*.wav")) == [index[0]["filename"]]
