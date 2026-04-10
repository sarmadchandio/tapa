"""File I/O for saving pipeline results."""

import csv
import json


def save_json(data, path):
    """Save data as indented JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_vowel_averages_csv(avgs, path):
    """Save vowel formant averages to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "vowel", "mean_f1", "mean_f2", "std_f1", "std_f2",
                     "mean_pitch", "n_tokens", "n_after_filtering"])
        for spk in sorted(avgs):
            for v in sorted(avgs[spk]):
                d = avgs[spk][v]
                w.writerow([spk, v, d["mean_f1"], d["mean_f2"], d["std_f1"], d["std_f2"],
                            d["mean_pitch"], d["n_tokens"], d["n_after_filtering"]])


def save_stop_averages_csv(avgs, path):
    """Save stop VOT averages to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "phone", "voicing", "place", "mean_vot_ms", "std_vot_ms",
                     "mean_closure_ms", "n_tokens", "n_after_filtering"])
        for spk in sorted(avgs):
            for ph in sorted(avgs[spk]):
                d = avgs[spk][ph]
                w.writerow([spk, ph, d["voicing"], d["place"], d["mean_vot_ms"],
                            d["std_vot_ms"], d["mean_closure_ms"], d["n_tokens"],
                            d["n_after_filtering"]])


def save_fricative_averages_csv(avgs, path):
    """Save fricative spectral moment averages to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "phone", "voicing", "place", "mean_cog", "std_cog",
                     "mean_spectral_sd", "mean_skewness", "mean_kurtosis",
                     "mean_duration_ms", "n_tokens", "n_after_filtering"])
        for spk in sorted(avgs):
            for ph in sorted(avgs[spk]):
                d = avgs[spk][ph]
                w.writerow([spk, ph, d["voicing"], d["place"], d["mean_cog"], d["std_cog"],
                            d["mean_spectral_sd"], d["mean_skewness"], d["mean_kurtosis"],
                            d["mean_duration_ms"], d["n_tokens"], d["n_after_filtering"]])
