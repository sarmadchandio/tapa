"""Phoneme segment identification from MFA or CMUdict fallback."""

import re
from collections import defaultdict

from .config import TAPAConfig
from .phoneme_maps import ARPABET_FRICATIVES, ARPABET_STOPS, ARPABET_VOWELS


def strip_stress(phone):
    """Remove stress markers from ARPAbet phones."""
    return re.sub(r"[0-9]", "", phone).upper()


def identify_segments_from_mfa(phones, diarization_segments, cfg=None):
    """Categorize MFA-aligned phones into vowels, stops, and fricatives per speaker.

    Returns (speaker_vowels, speaker_stops, speaker_fricatives) dicts.
    """
    if cfg is None:
        cfg = TAPAConfig()
    speaker_vowels = defaultdict(list)
    speaker_stops = defaultdict(list)
    speaker_frics = defaultdict(list)

    for i, ph in enumerate(phones):
        base = strip_stress(ph["phone"])
        mid = (ph["start"] + ph["end"]) / 2
        speaker = None
        for seg in diarization_segments:
            if seg["start"] <= mid <= seg["end"]:
                speaker = seg["speaker"]
                break
        if speaker is None:
            continue

        if base in ARPABET_VOWELS:
            ipa = ARPABET_VOWELS[base]
            if cfg.target_vowels is not None and ipa not in cfg.target_vowels:
                continue
            if (ph["end"] - ph["start"]) >= cfg.min_vowel_duration:
                speaker_vowels[speaker].append({
                    "ipa": ipa, "arpabet": base,
                    "start": ph["start"], "end": ph["end"], "phone_raw": ph["phone"],
                })

        elif base in ARPABET_STOPS:
            if (ph["end"] - ph["start"]) >= cfg.min_stop_duration:
                info = ARPABET_STOPS[base].copy()
                info.update({"arpabet": base, "start": ph["start"], "end": ph["end"],
                             "phone_raw": ph["phone"]})
                if i + 1 < len(phones):
                    nxt = phones[i + 1]
                    info["following_phone"] = strip_stress(nxt["phone"])
                    info["following_start"] = nxt["start"]
                    info["following_end"] = nxt["end"]
                else:
                    info["following_phone"] = info["following_start"] = info["following_end"] = None
                speaker_stops[speaker].append(info)

        elif base in ARPABET_FRICATIVES:
            if (ph["end"] - ph["start"]) >= cfg.min_fricative_duration:
                info = ARPABET_FRICATIVES[base].copy()
                info.update({"arpabet": base, "start": ph["start"], "end": ph["end"],
                             "phone_raw": ph["phone"]})
                speaker_frics[speaker].append(info)

    return dict(speaker_vowels), dict(speaker_stops), dict(speaker_frics)


def identify_segments_from_cmudict(words, diarization_segments, cmudict, cfg=None):
    """Fallback: proportional timing using CMU Dictionary when MFA is unavailable.

    Returns (speaker_vowels, speaker_stops, speaker_fricatives) dicts.
    """
    if cfg is None:
        cfg = TAPAConfig()
    speaker_vowels = defaultdict(list)
    speaker_stops = defaultdict(list)
    speaker_frics = defaultdict(list)

    for w in words:
        w_mid = (w["start"] + w["end"]) / 2
        speaker = None
        for seg in diarization_segments:
            if seg["start"] <= w_mid <= seg["end"]:
                speaker = seg["speaker"]
                break
        if speaker is None:
            continue
        wc = re.sub(r"[^a-zA-Z']", "", w["word"]).lower()
        if not wc or wc not in cmudict:
            continue
        pron = cmudict[wc][0]
        n = len(pron)
        dur = w["end"] - w["start"]
        if n == 0 or dur <= 0:
            continue
        pd = dur / n
        for j, phone in enumerate(pron):
            base = re.sub(r"[0-9]", "", phone).upper()
            ps, pe = w["start"] + j * pd, w["start"] + (j + 1) * pd
            if base in ARPABET_VOWELS:
                ipa = ARPABET_VOWELS[base]
                if cfg.target_vowels is not None and ipa not in cfg.target_vowels:
                    continue
                if pd >= cfg.min_vowel_duration:
                    speaker_vowels[speaker].append({
                        "ipa": ipa, "arpabet": base, "start": ps, "end": pe,
                        "phone_raw": phone, "word": w["word"]})
            elif base in ARPABET_STOPS:
                if pd >= cfg.min_stop_duration:
                    info = ARPABET_STOPS[base].copy()
                    info.update({"arpabet": base, "start": ps, "end": pe,
                                 "phone_raw": phone, "word": w["word"]})
                    if j + 1 < n:
                        nxt = re.sub(r"[0-9]", "", pron[j + 1]).upper()
                        info["following_phone"] = nxt
                        info["following_start"] = pe
                        info["following_end"] = pe + pd
                    else:
                        info["following_phone"] = info["following_start"] = info["following_end"] = None
                    speaker_stops[speaker].append(info)
            elif base in ARPABET_FRICATIVES:
                if pd >= cfg.min_fricative_duration:
                    info = ARPABET_FRICATIVES[base].copy()
                    info.update({"arpabet": base, "start": ps, "end": pe,
                                 "phone_raw": phone, "word": w["word"]})
                    speaker_frics[speaker].append(info)

    return dict(speaker_vowels), dict(speaker_stops), dict(speaker_frics)
