"""Vowel formant extraction using Praat."""

from collections import defaultdict

import numpy as np
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

from .config import TAPAConfig


def measure_vowel_formants(audio_np, cfg=None, sample_rate=16000):
    """Measure F1, F2, and pitch for a vowel segment."""
    if cfg is None:
        cfg = TAPAConfig()
    duration = len(audio_np) / sample_rate
    if duration < cfg.min_vowel_duration:
        return None
    trim = int(len(audio_np) * cfg.vowel_trim_fraction)
    if trim > 0 and len(audio_np) > 2 * trim + int(0.025 * sample_rate):
        audio_np = audio_np[trim:-trim]
    sound = parselmouth.Sound(audio_np, sampling_frequency=sample_rate)
    try:
        pitch_obj = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        voiced = pitch_obj.selected_array["frequency"]
        voiced = voiced[voiced > 0]
        median_pitch = float(np.median(voiced)) if len(voiced) > 0 else 0
        max_formant = 5500 if median_pitch >= 170 else 5000
    except Exception:
        median_pitch, max_formant = 0, 5500
    try:
        fobj = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                      maximum_formant=max_formant, window_length=0.025,
                                      pre_emphasis_from=50)
        f1 = call(fobj, "Get mean", 1, 0, 0, "Hertz")
        f2 = call(fobj, "Get mean", 2, 0, 0, "Hertz")
    except Exception:
        return None
    if not (cfg.f1_min <= f1 <= cfg.f1_max) or not (cfg.f2_min <= f2 <= cfg.f2_max) or f1 >= f2:
        return None
    return {"f1": round(f1, 1), "f2": round(f2, 1), "pitch": round(median_pitch, 1)}


def extract_all_vowel_formants(speaker_vowels, audio_np, cfg=None):
    """Extract formants for all identified vowel segments."""
    if cfg is None:
        cfg = TAPAConfig()
    results = defaultdict(lambda: defaultdict(list))
    all_v = [(spk, v) for spk, vl in speaker_vowels.items() for v in vl]
    for spk, v in tqdm(all_v, desc="Extracting vowel formants"):
        s = max(0, int(v["start"] * cfg.sample_rate))
        e = min(len(audio_np), int(v["end"] * cfg.sample_rate))
        chunk = audio_np[s:e]
        if len(chunk) < int(cfg.min_vowel_duration * cfg.sample_rate):
            continue
        fm = measure_vowel_formants(chunk, cfg, cfg.sample_rate)
        if fm:
            fm["abs_start"] = v["start"]
            fm["abs_end"] = v["end"]
            fm["arpabet"] = v["arpabet"]
            fm["word"] = v.get("word", "")
            results[spk][v["ipa"]].append(fm)
    return dict(results)
