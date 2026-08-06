"""Montreal Forced Aligner integration."""

import os
import re
import shutil
import signal
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


UNALIGNED_LABELS = {"spn", "sil", "sp", ""}


def summarize_alignment(output_dir):
    """Count words MFA could not convert to phonemes, across all TextGrids.

    MFA labels a word it cannot look up (or generate a pronunciation for) as
    ``spn``. Those words contribute no phonemes, so they silently vanish from
    every downstream measurement — this reports how many there were.

    Returns dict with n_words, n_unaligned, share, types (sorted word list),
    and oov_file (MFA's own OOV list, when it wrote one).
    """
    n_words = 0
    unaligned = []
    for tg_path in sorted(Path(output_dir).rglob("*.TextGrid")):
        try:
            tg = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=True)
        except Exception:
            continue
        names = {n.lower(): n for n in tg.tierNames}
        w_tier = next((names[n] for n in names if "word" in n), None)
        p_tier = next((names[n] for n in names if "phone" in n), None)
        if w_tier is None or p_tier is None:
            continue
        phones = [(float(i.start), float(i.end), i.label.strip())
                  for i in tg.getTier(p_tier).entries]
        for iv in tg.getTier(w_tier).entries:
            word = iv.label.strip()
            if not word or word in UNALIGNED_LABELS:
                continue
            n_words += 1
            ws, we = float(iv.start), float(iv.end)
            inside = [lab for s, e, lab in phones if s >= ws - 1e-6 and e <= we + 1e-6]
            if not any(lab and lab not in UNALIGNED_LABELS for lab in inside):
                unaligned.append(word)

    oov_file = next((str(p) for p in Path(output_dir).parent.rglob("oovs_found*.txt")), None)
    return {
        "n_words": n_words,
        "n_unaligned": len(unaligned),
        "share": (len(unaligned) / n_words) if n_words else 0.0,
        "types": sorted(set(unaligned)),
        "oov_file": oov_file,
    }


def print_alignment_report(report, method):
    """Print the alignment method and out-of-vocabulary summary."""
    print(f"          -> alignment method: {method}", flush=True)
    if report is None or not report["n_words"]:
        return
    n, tot, share = report["n_unaligned"], report["n_words"], report["share"]
    if n:
        types = report["types"]
        shown = ", ".join(types[:12]) + (" ..." if len(types) > 12 else "")
        print(f"          -> {n} of {tot} words ({share:.1%}) received no phoneme-level "
              f"alignment (out-of-vocabulary); they are excluded from all measurements",
              flush=True)
        print(f"             {len(types)} distinct word type(s): {shown}", flush=True)
        if report["oov_file"]:
            print(f"             MFA's own OOV list: {report['oov_file']}", flush=True)
    else:
        print(f"          -> all {tot} words received phoneme-level alignment", flush=True)


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


def _run_with_timeout(cmd, timeout_s, env):
    """Run a command, killing its whole process group if it overruns.

    subprocess.run(timeout=...) is not enough here. MFA starts worker
    processes, and when the parent is killed those workers keep the stdout and
    stderr pipes open, so the cleanup read after the timeout blocks forever —
    the run hangs indefinitely instead of falling back. Observed in practice:
    an alignment stuck for nearly an hour with a 30-minute timeout set. Running
    the child in its own session lets us kill the group and stop waiting.

    Returns (returncode, stderr). Raises subprocess.TimeoutExpired on overrun.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env, start_new_session=True)
    try:
        _, stderr = proc.communicate(timeout=timeout_s)
        return proc.returncode, stderr or ""
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:                       # reap; the group is gone so this returns
            proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        raise


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
           "--num_jobs", str(cfg.mfa_num_jobs),
           "--output_format", "long_textgrid", "--beam", "100", "--retry_beam", "400"]
    # MFA shells out to OpenFst/Kaldi tools (fstcompile, ...) by bare name and
    # they live next to the mfa binary — invoking mfa by absolute path (e.g.
    # /opt/miniforge/bin/mfa on Colab) without its bin dir on PATH makes every
    # alignment die with "Could not find 'fstcompile'".
    env = os.environ.copy()
    env["PATH"] = str(Path(mfa_bin).resolve().parent) + os.pathsep + env.get("PATH", "")
    try:
        returncode, stderr = _run_with_timeout(cmd, cfg.mfa_timeout_s, env)
        if returncode != 0:
            print(f"    MFA failed (exit {returncode}): {stderr[-300:]}")
            return None
    except FileNotFoundError:
        print("    MFA error -- falling back to CMUdict")
        return None
    except subprocess.TimeoutExpired:
        print(f"    MFA exceeded {cfg.mfa_timeout_s}s and was killed "
              f"-- falling back to CMUdict")
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
