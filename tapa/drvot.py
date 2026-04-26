"""Dr.VOT backend for stop-VOT measurement.

Replaces TAPA's Praat-based ``measure_vot`` with the deep-learning Dr.VOT model
(MLSpeech/Dr.VOT, Shrem et al. 2019) while keeping TAPA's per-token output
schema so downstream averaging/CSV/JSON code is unchanged.

Workflow per recording:

  1. For every stop token TAPA's segment-identification step produced, slice a
     short word-sized clip from the in-memory audio buffer and write it as a
     16 kHz mono wav inside a Dr.VOT-shaped temp directory.
  2. Shell out to Dr.VOT's four-step pipeline (prepare_wav_dir →
     process_data_pipeline → predict → post_predict_script).
  3. Parse Dr.VOT's ``new_summary.csv`` (or ``summary.csv``) and join each row
     back to its source token via the deterministic clip filename.
  4. For any token Dr.VOT could not predict (window misalignment, voiceless
     audio, etc.), fall back to TAPA's signal-based ``measure_vot`` so we keep
     coverage. Each token gets a ``vot_method`` field recording which engine
     produced the value.

The Dr.VOT clone is invoked via a configurable Python interpreter
(``cfg.drvot_python``), defaulting to ``sys.executable``. On Google Colab the
default is fine because the notebook's working install proves Dr.VOT's slim
deps coexist with the Colab Python.
"""

from __future__ import annotations

import csv
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .config import TAPAConfig
from .consonants import measure_vot
from .phoneme_maps import ARPABET_VOWELS


DRVOT_REPO_URL = "https://github.com/MLSpeech/Dr.VOT"
HARD_CODED_PRAAT = "/home/yosi/custom_commands/praat"
PATCH_TARGETS = ("process_data/extract_voice_starts.py", "process_data/pitch_process.py")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[DrVOT] {msg}", flush=True)


def _which_praat() -> str:
    p = shutil.which("praat")
    if not p:
        raise RuntimeError(
            "Praat not found on PATH. On Colab: !apt-get install -y praat. "
            "On Ubuntu: sudo apt-get install praat."
        )
    return p


def setup_drvot(repo_dir: str | os.PathLike, force: bool = False) -> str:
    """Clone Dr.VOT into ``repo_dir``, patch hard-coded paths, chmod the binary.

    Idempotent — safe to call repeatedly. Returns the absolute repo path.

    Args:
        repo_dir: Destination directory. Created if missing.
        force: If True, re-clone even if the directory already exists.
    """
    repo = Path(repo_dir).resolve()
    if force and repo.exists():
        _log(f"force=True, removing existing {repo}")
        shutil.rmtree(repo)

    if not repo.exists():
        _log(f"cloning {DRVOT_REPO_URL} -> {repo}")
        subprocess.run(["git", "clone", DRVOT_REPO_URL, str(repo)], check=True)
    else:
        _log(f"reusing existing clone at {repo}")

    # Patch the hard-coded Praat path that ships in the repo.
    praat_path = _which_praat()
    patched = 0
    for rel in PATCH_TARGETS:
        f = repo / rel
        if not f.exists():
            continue
        text = f.read_text()
        if HARD_CODED_PRAAT in text:
            f.write_text(text.replace(HARD_CODED_PRAAT, praat_path))
            patched += 1
    _log(f"patched Praat path -> {praat_path} ({patched} file(s))")

    # Make the bundled feature extractor executable. The repo ships a Linux and
    # a macOS variant; we just chmod whichever exists.
    for cand in ("linux_VotFrontEnd2", "mac_VotFrontEnd2", "VotFrontEnd2"):
        b = repo / cand
        if b.exists():
            mode = b.stat().st_mode
            b.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            _log(f"chmod +x {cand}")

    # Quick sanity check on the model weights so failures surface early.
    weights = repo / "final_models" / "adv_model.model"
    if not weights.exists():
        raise RuntimeError(
            f"Expected Dr.VOT weights at {weights} but file is missing — "
            "the clone may be incomplete or the upstream layout changed."
        )
    return str(repo)


# ---------------------------------------------------------------------------
# Clip writing
# ---------------------------------------------------------------------------

def _cut_clip(audio_np: np.ndarray, sample_rate: int,
              t_start: float, t_end: float,
              pre_ms: float, post_ms: float) -> tuple[np.ndarray, float]:
    """Slice a clip from ``audio_np`` with padding around [t_start, t_end].

    Returns (clip, clip_t0) where ``clip_t0`` is the absolute time (in seconds)
    of the first sample in the clip — needed if we ever want to map Dr.VOT's
    in-clip boundary times back to the original recording.
    """
    pre_s = pre_ms / 1000.0
    post_s = post_ms / 1000.0
    a = max(0, int((t_start - pre_s) * sample_rate))
    b = min(len(audio_np), int((t_end + post_s) * sample_rate))
    return audio_np[a:b], a / sample_rate


def _build_clip_index(speaker_stops: dict, audio_np: np.ndarray,
                      raw_dir: Path, sample_rate: int,
                      pre_ms: float, post_ms: float) -> list[dict]:
    """Write one wav per stop token; return a list of token-index records."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []

    flat = [(spk, s) for spk, sl in speaker_stops.items() for s in sl]
    skipped_no_following = 0
    skipped_too_short = 0

    for idx, (spk, s) in enumerate(flat):
        foll = s.get("following_phone")
        # Match TAPA's eligibility: only score stops with a following vowel.
        if foll is None or foll not in ARPABET_VOWELS:
            skipped_no_following += 1
            continue
        t_start = float(s["start"])
        t_end_for_clip = float(s["following_end"] or s["end"])
        clip, clip_t0 = _cut_clip(audio_np, sample_rate, t_start, t_end_for_clip,
                                  pre_ms, post_ms)
        if len(clip) < int(0.04 * sample_rate):
            skipped_too_short += 1
            continue
        fname = f"tok_{idx:06d}.wav"
        sf.write(raw_dir / fname, clip, sample_rate, subtype="PCM_16")
        index.append({
            "idx": idx,
            "filename": fname,
            "speaker": spk,
            "ipa": s["ipa"],
            "arpabet": s["arpabet"],
            "voicing": s["voicing"],
            "place": s["place"],
            "abs_start": t_start,
            "abs_end": float(s["end"]),
            "following_start": s.get("following_start"),
            "following_end": s.get("following_end"),
            "following_phone": foll,
            "following_vowel": ARPABET_VOWELS.get(foll, ""),
            "word": s.get("word", ""),
            "clip_t0": clip_t0,
            "stop_info": s,  # kept so the TAPA fallback can re-measure
        })

    _log(f"prepared {len(index)} clips for inference "
         f"(skipped: {skipped_no_following} no-following-vowel, "
         f"{skipped_too_short} too-short)")
    return index


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path, label: str) -> None:
    """Run a subprocess, streaming each stdout/stderr line live with a prefix."""
    _log(f"-> {label}: {' '.join(cmd)}")
    # Merge stderr into stdout so progress + error lines arrive in order.
    proc = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,  # line-buffered
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[DrVOT/{label}] {line.rstrip()}", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"Dr.VOT step '{label}' failed (exit {rc}). See log lines above."
        )


def _invoke_drvot(repo_dir: Path, raw_dir: Path, processed_dir: Path,
                  out_dir: Path, python_bin: str) -> Path:
    """Run Dr.VOT's four-step pipeline. Returns path to the summary CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    all_files = raw_dir / "all_files"
    if all_files.exists():
        shutil.rmtree(all_files)
    # Dr.VOT's prepare_wav_dir.py expects --output_dir to exist (it doesn't
    # create it). Same for the other steps' working dirs above.
    all_files.mkdir(parents=True)

    # Step 1: flatten raw wavs into all_files/ + write files.txt
    _run([python_bin, "process_data/prepare_wav_dir.py",
          "--input_dir", str(raw_dir), "--output_dir", str(all_files)],
         repo_dir, "prepare")

    # Step 2: Praat voice-start detection + C++ feature extractor
    _run([python_bin, "process_data_pipeline.py",
          "--input_dir", str(all_files), "--output_dir", str(processed_dir)],
         repo_dir, "features")

    # Step 3: torch CNN inference -> per-clip TextGrids + summary.csv
    voice_starts = all_files / "voice_starts.txt"
    _run([python_bin, "predict.py",
          "--inference", str(processed_dir),
          "--out_dir", str(out_dir),
          "--durations", str(voice_starts)],
         repo_dir, "predict")

    # Step 4: rewrite summary -> new_summary.csv (uses original filenames)
    summary_in = out_dir / "summary.csv"
    files_txt = all_files / "files.txt"
    _run([python_bin, "post_predict_script.py",
          "--summary", str(summary_in),
          "--tg_dir", str(out_dir),
          "--filenames", str(files_txt)],
         repo_dir, "post")

    new_summary = out_dir / "new_summary.csv"
    if new_summary.exists():
        return new_summary
    if summary_in.exists():
        _log("post_predict_script did not produce new_summary.csv; "
             "falling back to summary.csv")
        return summary_in
    raise RuntimeError(f"No Dr.VOT summary CSV found in {out_dir}")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

_VAL_KEYS = ("vot_ms", "vot_value", "vot_dur", "vot_duration", "duration", "vot",
             "value", "ms")
_CLASS_KEYS = ("vot_type", "type", "class", "tag", "label", "predicted_class",
               "vot_class")


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    fn_lower = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand in fn_lower:
            return fn_lower[cand]
    # Substring match as a last resort (Dr.VOT has shifted column names across
    # versions, so be lenient).
    for f in fieldnames:
        fl = f.lower()
        for cand in candidates:
            if cand in fl:
                return f
    return None


def _parse_summary(csv_path: Path) -> dict[str, dict]:
    """Map filename stem -> {vot_ms, vot_class}."""
    out: dict[str, dict] = {}
    with open(csv_path, newline="") as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames:
            return out
        # Filename column candidates
        fn_col = _pick_column(rdr.fieldnames, ("filename", "file", "name", "path"))
        val_col = _pick_column(rdr.fieldnames, _VAL_KEYS)
        cls_col = _pick_column(rdr.fieldnames, _CLASS_KEYS)
        if fn_col is None or val_col is None:
            _log(f"summary CSV columns not recognized: {rdr.fieldnames!r}")
            return out
        for row in rdr:
            stem = Path(row[fn_col]).stem
            try:
                vot_ms = float(row[val_col])
            except (ValueError, TypeError):
                continue
            cls = (row.get(cls_col) or "").strip() if cls_col else ""
            out[stem] = {"vot_ms": round(vot_ms, 2),
                         "vot_class_drvot": cls or None}
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_all_stop_measurements_drvot(speaker_stops: dict, audio_np: np.ndarray,
                                        cfg: Optional[TAPAConfig] = None) -> dict:
    """Drop-in replacement for ``extract_all_stop_measurements`` using Dr.VOT.

    Same input/output schema; adds two extra fields per token:
      - ``vot_method``: ``"drvot"`` or ``"tapa-fallback"``
      - ``vot_class_drvot``: ``"POS"``/``"NEG"``/None when Dr.VOT measured

    Tokens Dr.VOT cannot predict are re-measured with TAPA's Praat path so we
    keep coverage. If the Dr.VOT setup itself fails (missing repo, missing
    weights, subprocess error), the whole stop-measurement step falls back to
    TAPA-Praat — the unified pipeline never crashes the run because of Dr.VOT.
    """
    if cfg is None:
        cfg = TAPAConfig()
    if not cfg.drvot_repo_dir:
        raise RuntimeError(
            "cfg.drvot_repo_dir is not set. Call tapa.drvot.setup_drvot() first "
            "and pass the path via TAPAConfig(drvot_repo_dir=...)."
        )

    repo = Path(cfg.drvot_repo_dir).resolve()
    if not repo.exists():
        raise RuntimeError(f"Dr.VOT repo not found at {repo}; run setup_drvot() first.")

    python_bin = cfg.drvot_python or sys.executable
    sr = cfg.sample_rate

    _log(f"primary backend: Dr.VOT  (repo={repo}, python={python_bin})")
    _log(f"clip window: pre={cfg.drvot_clip_pre_ms}ms / post={cfg.drvot_clip_post_ms}ms")

    work = Path(tempfile.mkdtemp(prefix="tapa_drvot_"))
    raw_dir = work / "raw"
    processed_dir = work / "processed"
    out_dir = work / "out_tg"

    try:
        # 1. Cut clips
        index = _build_clip_index(speaker_stops, audio_np, raw_dir, sr,
                                  cfg.drvot_clip_pre_ms, cfg.drvot_clip_post_ms)
        if not index:
            _log("no eligible stop tokens found — nothing for Dr.VOT to score")
            return {}

        # 2. Run Dr.VOT
        t0 = time.time()
        summary_csv = _invoke_drvot(repo, raw_dir, processed_dir, out_dir, python_bin)
        _log(f"Dr.VOT inference finished in {time.time() - t0:.1f}s; "
             f"reading {summary_csv.name}")

        # 3. Parse summary
        predictions = _parse_summary(summary_csv)
        _log(f"parsed {len(predictions)} predictions from summary")

        # 4. Build per-token results, falling back to TAPA-Praat where missing
        results = defaultdict(lambda: defaultdict(list))
        n_drvot = n_fallback = n_dropped = 0
        for entry in index:
            stem = Path(entry["filename"]).stem
            pred = predictions.get(stem)
            closure_ms = round((entry["abs_end"] - entry["abs_start"]) * 1000, 2)

            if pred is not None:
                meas = {
                    "vot_ms": pred["vot_ms"],
                    "burst_time": None,
                    "voicing_onset": None,
                    "closure_duration_ms": closure_ms,
                    "abs_start": entry["abs_start"],
                    "abs_end": entry["abs_end"],
                    "voicing": entry["voicing"],
                    "place": entry["place"],
                    "arpabet": entry["arpabet"],
                    "following_vowel": entry["following_vowel"],
                    "word": entry["word"],
                    "vot_method": "drvot",
                    "vot_class_drvot": pred["vot_class_drvot"],
                }
                results[entry["speaker"]][entry["ipa"]].append(meas)
                n_drvot += 1
                continue

            # Fallback: signal-based TAPA measurement on the same token.
            tapa_meas = measure_vot(entry["stop_info"], audio_np, cfg, sr)
            if tapa_meas is None:
                n_dropped += 1
                continue
            tapa_meas.update({
                "abs_start": entry["abs_start"],
                "abs_end": entry["abs_end"],
                "voicing": entry["voicing"],
                "place": entry["place"],
                "arpabet": entry["arpabet"],
                "following_vowel": entry["following_vowel"],
                "word": entry["word"],
                "vot_method": "tapa-fallback",
                "vot_class_drvot": None,
            })
            results[entry["speaker"]][entry["ipa"]].append(tapa_meas)
            n_fallback += 1

        total = n_drvot + n_fallback + n_dropped
        if total:
            pct = lambda n: f"{100.0 * n / total:.1f}%"
            _log(f"coverage: {n_drvot} Dr.VOT ({pct(n_drvot)}), "
                 f"{n_fallback} TAPA-Praat fallback ({pct(n_fallback)}), "
                 f"{n_dropped} dropped ({pct(n_dropped)})")
        return dict(results)

    except Exception as e:
        _log(f"FAILED: {type(e).__name__}: {e}")
        for line in traceback.format_exc().splitlines():
            _log(f"  {line}")
        _log("falling back to TAPA-Praat for the whole recording")
        # Total-failure path: behave as if vot_backend were "tapa".
        from .consonants import extract_all_stop_measurements as _tapa_all
        results = _tapa_all(speaker_stops, audio_np, cfg)
        for spk in results:
            for ph in results[spk]:
                for m in results[spk][ph]:
                    m.setdefault("vot_method", "tapa-fallback")
                    m.setdefault("vot_class_drvot", None)
        return results

    finally:
        # Clean up the temp dir unless the user asked us to keep it.
        if cfg.drvot_keep_temp:
            _log(f"keeping temp dir: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# `python -m tapa.drvot setup <dir>` entry point
# ---------------------------------------------------------------------------

def _main():
    import argparse
    p = argparse.ArgumentParser(prog="python -m tapa.drvot",
                                description="Dr.VOT setup utilities for TAPA.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="Clone Dr.VOT and patch it for use with TAPA.")
    s.add_argument("repo_dir", help="Destination directory (will be created).")
    s.add_argument("--force", action="store_true",
                   help="Re-clone even if the directory already exists.")
    args = p.parse_args()
    if args.cmd == "setup":
        path = setup_drvot(args.repo_dir, force=args.force)
        _log(f"setup complete -> {path}")
        _log("Now run TAPA with --vot-backend=drvot --drvot-repo " + path)
        _log("(Slim Python deps come from `pip install tapa[drvot]`.)")


if __name__ == "__main__":
    _main()
