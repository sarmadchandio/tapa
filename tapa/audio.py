"""Audio loading utilities."""

import subprocess
from pathlib import Path

import numpy as np


def load_audio_16k(path, sample_rate=16000):
    """Decode an audio file to a float32 mono array at ``sample_rate`` via ffmpeg.

    ffmpeg does the resampling/downmixing out-of-process, so the native-rate
    buffer never lives in Python memory — a 2 h 48 kHz stereo recording is
    ~2.6 GB decoded natively as float32, vs ~0.44 GB at 16 kHz mono. This is
    what keeps long recordings inside Colab's RAM budget.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(Path(path)),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(sample_rate),
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found — it is required to load audio. "
            "Install it (e.g. `apt install ffmpeg` / `conda install ffmpeg`)."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed to decode {path}: {e.stderr.decode(errors='replace').strip()}"
        )
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
