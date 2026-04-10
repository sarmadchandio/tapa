"""Consonant measurements: stop VOT and fricative spectral moments."""

from collections import defaultdict

import numpy as np
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

from .config import TAPAConfig
from .phoneme_maps import ARPABET_VOWELS


def measure_vot(stop_info, audio_np, cfg=None, sample_rate=16000):
    """Measure Voice Onset Time for a stop consonant."""
    if cfg is None:
        cfg = TAPAConfig()
    foll = stop_info.get("following_phone")
    if foll is None or foll not in ARPABET_VOWELS:
        return None
    stop_start, stop_end = stop_info["start"], stop_info["end"]
    foll_start, foll_end = stop_info["following_start"], stop_info["following_end"]
    if foll_start is None or foll_end is None:
        return None

    a_start = int(stop_start * sample_rate)
    a_end = int(min(foll_start + 0.05, foll_end) * sample_rate)
    a_end = min(a_end, len(audio_np))
    if a_end - a_start < int(0.01 * sample_rate):
        return None

    seg = audio_np[a_start:a_end]
    sound = parselmouth.Sound(seg, sampling_frequency=sample_rate)

    # Find burst (intensity peak near stop-vowel boundary)
    try:
        intensity = sound.to_intensity(minimum_pitch=100, time_step=0.001)
        times = intensity.xs()
        values = [intensity.get_value(t) for t in times]
        rel_end = stop_end - stop_start
        burst_time, burst_val = None, -np.inf
        for t, v in zip(times, values):
            if v is not None and not np.isnan(v):
                if (rel_end * 0.33) <= t <= (rel_end + 0.01):
                    if v > burst_val:
                        burst_val = v
                        burst_time = stop_start + t
    except Exception:
        return None
    if burst_time is None:
        return None

    # Find voicing onset (first pitch after burst)
    try:
        pitch_obj = sound.to_pitch_cc(time_step=0.001, pitch_floor=75,
                                       pitch_ceiling=600, voicing_threshold=0.45)
        pt, pv = pitch_obj.xs(), pitch_obj.selected_array["frequency"]
        burst_rel = burst_time - stop_start
        voicing_onset = None
        for ti, vi in zip(pt, pv):
            if ti > burst_rel and vi > 0:
                voicing_onset = stop_start + ti
                break
    except Exception:
        return None
    if voicing_onset is None:
        return None

    vot = voicing_onset - burst_time
    if vot < -0.05 or vot > cfg.vot_max:
        return None
    return {
        "vot_ms": round(vot * 1000, 2),
        "burst_time": round(burst_time, 4),
        "voicing_onset": round(voicing_onset, 4),
        "closure_duration_ms": round((stop_end - stop_start) * 1000, 2),
    }


def extract_all_stop_measurements(speaker_stops, audio_np, cfg=None):
    """Extract VOT measurements for all identified stop consonants."""
    if cfg is None:
        cfg = TAPAConfig()
    results = defaultdict(lambda: defaultdict(list))
    all_s = [(spk, s) for spk, sl in speaker_stops.items() for s in sl]
    for spk, s in tqdm(all_s, desc="Extracting stop VOT"):
        meas = measure_vot(s, audio_np, cfg, cfg.sample_rate)
        if meas:
            meas["abs_start"] = s["start"]
            meas["abs_end"] = s["end"]
            meas["voicing"] = s["voicing"]
            meas["place"] = s["place"]
            meas["arpabet"] = s["arpabet"]
            meas["following_vowel"] = ARPABET_VOWELS.get(s.get("following_phone", ""), "")
            meas["word"] = s.get("word", "")
            results[spk][s["ipa"]].append(meas)
    return dict(results)


def measure_fricative_spectral(fric_info, audio_np, cfg=None, sample_rate=16000):
    """Measure spectral moments for a fricative segment."""
    if cfg is None:
        cfg = TAPAConfig()
    s = max(0, int(fric_info["start"] * sample_rate))
    e = min(len(audio_np), int(fric_info["end"] * sample_rate))
    seg = audio_np[s:e]
    if len(seg) < int(cfg.min_fricative_duration * sample_rate):
        return None
    trim = int(len(seg) * 0.10)
    if trim > 0 and len(seg) > 2 * trim + 64:
        seg = seg[trim:-trim]
    sound = parselmouth.Sound(seg, sampling_frequency=sample_rate)
    try:
        spectrum = sound.to_spectrum()
        cog = call(spectrum, "Get centre of gravity", 2)
        std_dev = call(spectrum, "Get standard deviation", 2)
        skewness = call(spectrum, "Get skewness", 2)
        kurtosis = call(spectrum, "Get kurtosis", 2)
    except Exception:
        return None
    if cog < 500 or cog > 12000:
        return None
    try:
        iobj = sound.to_intensity(minimum_pitch=100)
        mean_int = call(iobj, "Get mean", 0, 0, "dB")
    except Exception:
        mean_int = 0
    dur_ms = (fric_info["end"] - fric_info["start"]) * 1000
    return {"cog": round(cog, 1), "spectral_sd": round(std_dev, 1),
            "skewness": round(skewness, 3), "kurtosis": round(kurtosis, 3),
            "duration_ms": round(dur_ms, 2), "mean_intensity_dB": round(mean_int, 1)}


def extract_all_fricative_measurements(speaker_frics, audio_np, cfg=None):
    """Extract spectral moments for all identified fricatives."""
    if cfg is None:
        cfg = TAPAConfig()
    results = defaultdict(lambda: defaultdict(list))
    all_f = [(spk, f) for spk, fl in speaker_frics.items() for f in fl]
    for spk, f in tqdm(all_f, desc="Extracting fricative spectra"):
        meas = measure_fricative_spectral(f, audio_np, cfg, cfg.sample_rate)
        if meas:
            meas["abs_start"] = f["start"]
            meas["abs_end"] = f["end"]
            meas["voicing"] = f["voicing"]
            meas["place"] = f["place"]
            meas["arpabet"] = f["arpabet"]
            meas["word"] = f.get("word", "")
            results[spk][f["ipa"]].append(meas)
    return dict(results)
