"""Montreal Forced Aligner integration."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import librosa
import soundfile as sf
from praatio import textgrid as tgio

from .config import TAPAConfig


def prepare_mfa_input(audio_path, words, temp_dir, cfg=None, audio_np=None):
    """Prepare WAV + transcript for MFA alignment.

    ``audio_np`` (float32 mono at cfg.sample_rate) skips re-decoding the file
    when the caller already holds the audio in memory.
    """
    if cfg is None:
        cfg = TAPAConfig()
    os.makedirs(temp_dir, exist_ok=True)
    stem = Path(audio_path).stem
    wav_path = os.path.join(temp_dir, f"{stem}.wav")
    if audio_np is None:
        audio_np, _ = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
    sf.write(wav_path, audio_np, cfg.sample_rate)
    txt_path = os.path.join(temp_dir, f"{stem}.txt")
    transcript = " ".join(w["word"] for w in words)
    transcript = re.sub(r"[^\w\s']", "", transcript)
    with open(txt_path, "w") as f:
        f.write(transcript)
    return wav_path, txt_path


def prepare_mfa_input_segmented(audio_path, words, segments, temp_dir, cfg=None,
                                audio_np=None):
    """Prepare a per-utterance MFA corpus split at diarization boundaries.

    Aligning a long recording as ONE utterance makes MFA's alignment lattice
    scale with the full transcript length (at beam 100 / retry-beam 400 that
    is GBs of RAM and can hang on multi-hour audio). Splitting at the
    diarization segments the pipeline already has bounds the lattice to one
    segment and turns the job into MFA's normal corpus mode.

    Writes ``{stem}_NNNN.wav`` + ``.txt`` per segment that contains words.
    Returns {utterance_name: time_offset_seconds} for parse_textgrids_dir().
    """
    if cfg is None:
        cfg = TAPAConfig()
    os.makedirs(temp_dir, exist_ok=True)
    stem = Path(audio_path).stem
    if audio_np is None:
        audio_np, _ = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
    sr = cfg.sample_rate
    pad = cfg.mfa_utterance_pad

    # Every word goes to the segment containing its midpoint, or the nearest
    # segment otherwise — dropping out-of-segment words would lose tokens.
    groups = [[] for _ in segments]
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        idx = None
        for i, seg in enumerate(segments):
            if seg["start"] <= mid <= seg["end"]:
                idx = i
                break
        if idx is None:
            idx = min(range(len(segments)),
                      key=lambda i: max(segments[i]["start"] - mid,
                                        mid - segments[i]["end"], 0.0))
        groups[idx].append(w)

    offsets = {}
    for i, (seg, seg_words) in enumerate(zip(segments, groups)):
        if not seg_words:
            continue
        transcript = " ".join(w["word"] for w in seg_words)
        transcript = re.sub(r"[^\w\s']", "", transcript).strip()
        if not transcript:
            continue
        # Cover the segment AND its words' extents (Whisper timestamps can
        # stick out past VAD boundaries), plus padding so MFA sees context.
        t0 = min(seg["start"], seg_words[0]["start"]) - pad
        t1 = max(seg["end"], seg_words[-1]["end"]) + pad
        s = max(0, int(t0 * sr))
        e = min(len(audio_np), int(t1 * sr))
        if e - s < int(0.05 * sr):
            continue
        name = f"{stem}_{i:04d}"
        sf.write(os.path.join(temp_dir, f"{name}.wav"), audio_np[s:e], sr)
        with open(os.path.join(temp_dir, f"{name}.txt"), "w") as f:
            f.write(transcript)
        offsets[name] = s / sr
    return offsets


def parse_textgrids_dir(output_dir, offsets):
    """Merge per-utterance TextGrids back onto the recording's timeline.

    Returns one phone list (sorted by start) equivalent to what
    parse_textgrid() yields for a whole-recording alignment.
    """
    phones = []
    for tg_path in Path(output_dir).rglob("*.TextGrid"):
        offset = offsets.get(tg_path.stem)
        if offset is None:
            continue
        for ph in parse_textgrid(str(tg_path)):
            ph["start"] = round(ph["start"] + offset, 6)
            ph["end"] = round(ph["end"] + offset, 6)
            phones.append(ph)
    phones.sort(key=lambda p: p["start"])
    return phones


def find_mfa_bin(cfg=None):
    """Locate the MFA binary."""
    if cfg is None:
        cfg = TAPAConfig()
    if cfg.mfa_bin and os.path.exists(cfg.mfa_bin):
        return cfg.mfa_bin
    # Check common locations
    for path in ["/opt/miniforge/bin/mfa", "/usr/local/bin/mfa"]:
        if os.path.exists(path):
            return path
    return shutil.which("mfa")


def run_mfa_alignment(temp_dir, output_dir, cfg=None):
    """Run MFA alignment. Returns TextGrid path or None on failure."""
    if cfg is None:
        cfg = TAPAConfig()
    os.makedirs(output_dir, exist_ok=True)
    mfa_bin = find_mfa_bin(cfg)
    if not mfa_bin:
        print("    MFA not found -- falling back to CMUdict proportional timing")
        return None
    cmd = [mfa_bin, "align", temp_dir, "english_us_arpa", "english_us_arpa",
           output_dir, "--clean", "--single_speaker",
           "--output_format", "long_textgrid", "--beam", "100", "--retry_beam", "400"]
    # MFA shells out to OpenFst/Kaldi tools (fstcompile, ...) by bare name and
    # they live next to the mfa binary — invoking mfa by absolute path (e.g.
    # /opt/miniforge/bin/mfa on Colab) without its bin dir on PATH makes every
    # alignment die with "Could not find 'fstcompile'".
    env = os.environ.copy()
    env["PATH"] = str(Path(mfa_bin).resolve().parent) + os.pathsep + env.get("PATH", "")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                                env=env)
        if result.returncode != 0:
            print(f"    MFA failed: {result.stderr[-300:]}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("    MFA error -- falling back to CMUdict")
        return None
    tg_files = list(Path(output_dir).glob("*.TextGrid"))
    return str(tg_files[0]) if tg_files else None


def parse_textgrid(textgrid_path):
    """Parse a TextGrid file and extract phone intervals."""
    tg = tgio.openTextgrid(textgrid_path, includeEmptyIntervals=True)
    phone_tier = None
    for name in tg.tierNames:
        if "phone" in name.lower():
            phone_tier = name
            break
    if phone_tier is None:
        phone_tier = tg.tierNames[0] if tg.tierNames else None
    if phone_tier is None:
        return []
    tier = tg.getTier(phone_tier)
    phones = []
    for iv in tier.entries:
        lab = iv.label.strip()
        if lab and lab not in ("sil", "sp", "spn", ""):
            phones.append({"phone": lab, "start": float(iv.start), "end": float(iv.end)})
    return phones
