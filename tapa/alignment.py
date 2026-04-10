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


def prepare_mfa_input(audio_path, words, temp_dir, cfg=None):
    """Prepare WAV + transcript for MFA alignment."""
    if cfg is None:
        cfg = TAPAConfig()
    os.makedirs(temp_dir, exist_ok=True)
    stem = Path(audio_path).stem
    wav_path = os.path.join(temp_dir, f"{stem}.wav")
    audio_np, sr = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
    sf.write(wav_path, audio_np, cfg.sample_rate)
    txt_path = os.path.join(temp_dir, f"{stem}.txt")
    transcript = " ".join(w["word"] for w in words)
    transcript = re.sub(r"[^\w\s']", "", transcript)
    with open(txt_path, "w") as f:
        f.write(transcript)
    return wav_path, txt_path


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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
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
