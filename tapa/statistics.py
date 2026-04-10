"""Statistical averaging with MAD outlier rejection."""

import numpy as np

from .config import TAPAConfig


def _mad_filter(values, threshold=2.0):
    """Median Absolute Deviation outlier filter."""
    values = np.array(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.ones(len(values), dtype=bool)
    return np.abs(values - median) <= threshold * mad * 1.4826


def compute_vowel_averages(data, cfg=None):
    """Compute per-speaker, per-vowel average formants with outlier rejection."""
    if cfg is None:
        cfg = TAPAConfig()
    avgs = {}
    for spk, vowels in data.items():
        avgs[spk] = {}
        for v, ms in vowels.items():
            n = len(ms)
            if n < 1:
                continue
            if n == 1:
                m = ms[0]
                avgs[spk][v] = {"mean_f1": m["f1"], "mean_f2": m["f2"], "std_f1": 0, "std_f2": 0,
                                "mean_pitch": m["pitch"], "n_tokens": 1, "n_after_filtering": 1}
                continue
            f1 = np.array([m["f1"] for m in ms])
            f2 = np.array([m["f2"] for m in ms])
            pitch = np.array([m["pitch"] for m in ms])
            mask = _mad_filter(f1, cfg.mad_threshold) & _mad_filter(f2, cfg.mad_threshold)
            f1c, f2c, pc = f1[mask], f2[mask], pitch[mask]
            if len(f1c) == 0:
                continue
            avgs[spk][v] = {
                "mean_f1": round(float(np.mean(f1c)), 1), "mean_f2": round(float(np.mean(f2c)), 1),
                "std_f1": round(float(np.std(f1c, ddof=1)) if len(f1c) > 1 else 0, 1),
                "std_f2": round(float(np.std(f2c, ddof=1)) if len(f2c) > 1 else 0, 1),
                "mean_pitch": round(float(np.mean(pc)), 1),
                "n_tokens": n, "n_after_filtering": len(f1c)}
    return avgs


def compute_stop_averages(data, cfg=None):
    """Compute per-speaker, per-stop average VOT with outlier rejection."""
    if cfg is None:
        cfg = TAPAConfig()
    avgs = {}
    for spk, stops in data.items():
        avgs[spk] = {}
        for ph, ms in stops.items():
            n = len(ms)
            if n < 1:
                continue
            vots = np.array([m["vot_ms"] for m in ms])
            if n == 1:
                avgs[spk][ph] = {"mean_vot_ms": round(float(vots[0]), 2), "std_vot_ms": 0,
                                 "mean_closure_ms": round(ms[0]["closure_duration_ms"], 2),
                                 "voicing": ms[0]["voicing"], "place": ms[0]["place"],
                                 "n_tokens": 1, "n_after_filtering": 1}
                continue
            mask = _mad_filter(vots, cfg.mad_threshold)
            clean = vots[mask]
            if len(clean) == 0:
                continue
            closures = np.array([m["closure_duration_ms"] for m in ms])[mask]
            avgs[spk][ph] = {
                "mean_vot_ms": round(float(np.mean(clean)), 2),
                "std_vot_ms": round(float(np.std(clean, ddof=1)) if len(clean) > 1 else 0, 2),
                "mean_closure_ms": round(float(np.mean(closures)), 2),
                "voicing": ms[0]["voicing"], "place": ms[0]["place"],
                "n_tokens": n, "n_after_filtering": len(clean)}
    return avgs


def compute_fricative_averages(data, cfg=None):
    """Compute per-speaker, per-fricative average spectral moments with outlier rejection."""
    if cfg is None:
        cfg = TAPAConfig()
    avgs = {}
    for spk, frics in data.items():
        avgs[spk] = {}
        for ph, ms in frics.items():
            n = len(ms)
            if n < 1:
                continue
            cogs = np.array([m["cog"] for m in ms])
            if n == 1:
                m = ms[0]
                avgs[spk][ph] = {"mean_cog": m["cog"], "std_cog": 0,
                                 "mean_spectral_sd": m["spectral_sd"],
                                 "mean_skewness": m["skewness"], "mean_kurtosis": m["kurtosis"],
                                 "mean_duration_ms": m["duration_ms"],
                                 "voicing": m["voicing"], "place": m["place"],
                                 "n_tokens": 1, "n_after_filtering": 1}
                continue
            mask = _mad_filter(cogs, cfg.mad_threshold)
            if mask.sum() == 0:
                continue
            clean = [m for m, k in zip(ms, mask) if k]
            avgs[spk][ph] = {
                "mean_cog": round(float(np.mean([m["cog"] for m in clean])), 1),
                "std_cog": round(float(np.std([m["cog"] for m in clean], ddof=1)) if len(clean) > 1 else 0, 1),
                "mean_spectral_sd": round(float(np.mean([m["spectral_sd"] for m in clean])), 1),
                "mean_skewness": round(float(np.mean([m["skewness"] for m in clean])), 3),
                "mean_kurtosis": round(float(np.mean([m["kurtosis"] for m in clean])), 3),
                "mean_duration_ms": round(float(np.mean([m["duration_ms"] for m in clean])), 2),
                "voicing": clean[0]["voicing"], "place": clean[0]["place"],
                "n_tokens": n, "n_after_filtering": len(clean)}
    return avgs
