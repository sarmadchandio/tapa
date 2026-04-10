"""Convenience wrappers for using pipeline components standalone."""

import os
from pathlib import Path

import librosa
import numpy as np

from .config import TAPAConfig


def diarize(audio_path, num_speakers=None, config=None):
    """Diarize an audio file and return speaker-labeled segments.

    Args:
        audio_path: Path to audio file (.mp3, .wav, .flac).
        num_speakers: Number of speakers (None = auto-detect).
        config: Optional TAPAConfig.

    Returns:
        List of dicts with keys: speaker, start, end.
    """
    from .diarization import assign_speakers, get_speech_segments, load_silero_vad
    from resemblyzer import VoiceEncoder

    cfg = config or TAPAConfig(num_speakers=num_speakers)
    if num_speakers is not None:
        cfg.num_speakers = num_speakers

    vad_model, get_speech_timestamps = load_silero_vad()
    segments, wav_t, sr = get_speech_segments(audio_path, vad_model, get_speech_timestamps, cfg)
    wav_np = wav_t.numpy().astype(np.float32)
    encoder = VoiceEncoder()
    return assign_speakers(segments, wav_np, sr, encoder, cfg)


def transcribe(audio_path, model_name="small.en", device=None):
    """Transcribe an audio file and return word-level timestamps.

    Args:
        audio_path: Path to audio file.
        model_name: Whisper model name (default: "small.en").
        device: torch device (default: auto-detect).

    Returns:
        List of dicts with keys: word, start, end.
    """
    import torch
    import whisper

    from .transcription import transcribe_audio

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(model_name, device=device)
    return transcribe_audio(audio_path, model)


def align(audio_path, words, config=None):
    """Run forced alignment on audio and return phone boundaries.

    Uses MFA if available, otherwise falls back to CMUdict proportional timing.

    Args:
        audio_path: Path to audio file.
        words: Word list from transcribe() (list of dicts with word, start, end).
        config: Optional TAPAConfig.

    Returns:
        List of dicts with keys: phone, start, end.
    """
    import shutil

    import nltk

    from .alignment import find_mfa_bin, parse_textgrid, prepare_mfa_input, run_mfa_alignment

    cfg = config or TAPAConfig()
    stem = Path(audio_path).stem
    mfa_bin = find_mfa_bin(cfg)

    if mfa_bin:
        mfa_in = os.path.join(cfg.mfa_temp_dir, stem)
        mfa_out = os.path.join(cfg.mfa_temp_dir, f"{stem}_aligned")
        prepare_mfa_input(audio_path, words, mfa_in, cfg)
        tg_path = run_mfa_alignment(mfa_in, mfa_out, cfg)
        if tg_path:
            phones = parse_textgrid(tg_path)
            shutil.rmtree(cfg.mfa_temp_dir, ignore_errors=True)
            return phones

    # CMUdict fallback — return proportional phone estimates
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict as _cmudict

    from .segments import identify_segments_from_cmudict

    cmu = _cmudict.dict()
    # Return raw phones via a dummy single-speaker segment covering the full audio
    duration = words[-1]["end"] if words else 0
    dummy_segments = [{"speaker": "SPEAKER_00", "start": 0, "end": duration}]
    sp_v, sp_s, sp_f = identify_segments_from_cmudict(words, dummy_segments, cmu, cfg)
    # Flatten all segments back to a phone list
    all_phones = []
    for seg_list in [sp_v, sp_s, sp_f]:
        for spk_segs in seg_list.values():
            for seg in spk_segs:
                all_phones.append({
                    "phone": seg.get("phone_raw", seg.get("arpabet", "")),
                    "start": seg["start"],
                    "end": seg["end"],
                })
    all_phones.sort(key=lambda p: p["start"])
    return all_phones


def extract_formants(audio_path, segments=None, config=None):
    """Extract vowel formants from an audio file.

    Args:
        audio_path: Path to audio file.
        segments: Diarization segments from diarize(). If None, treats entire
            audio as one speaker.
        config: Optional TAPAConfig.

    Returns:
        dict: {speaker: {vowel_ipa: [{"f1", "f2", "pitch", ...}]}}
    """
    import nltk

    from .alignment import find_mfa_bin, parse_textgrid, prepare_mfa_input, run_mfa_alignment
    from .segments import identify_segments_from_cmudict, identify_segments_from_mfa
    from .vowels import extract_all_vowel_formants

    cfg = config or TAPAConfig()
    audio_np, _ = librosa.load(audio_path, sr=cfg.sample_rate)

    if segments is None:
        duration = len(audio_np) / cfg.sample_rate
        segments = [{"speaker": "SPEAKER_00", "start": 0, "end": duration}]

    # Get phone boundaries
    words = transcribe(audio_path, model_name=cfg.whisper_model)
    phones = align(audio_path, words, cfg)

    sp_v, _, _ = identify_segments_from_mfa(phones, segments, cfg)
    if not sp_v:
        nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict as _cmudict
        cmu = _cmudict.dict()
        sp_v, _, _ = identify_segments_from_cmudict(words, segments, cmu, cfg)

    return extract_all_vowel_formants(sp_v, audio_np, cfg)


def extract_consonants(audio_path, segments=None, config=None):
    """Extract stop VOT and fricative spectral moments from an audio file.

    Args:
        audio_path: Path to audio file.
        segments: Diarization segments from diarize(). If None, treats entire
            audio as one speaker.
        config: Optional TAPAConfig.

    Returns:
        tuple: (stop_data, fricative_data) where each is
        {speaker: {phone_ipa: [measurement_dicts]}}
    """
    import nltk

    from .alignment import find_mfa_bin, parse_textgrid, prepare_mfa_input, run_mfa_alignment
    from .consonants import extract_all_fricative_measurements, extract_all_stop_measurements
    from .segments import identify_segments_from_cmudict, identify_segments_from_mfa

    cfg = config or TAPAConfig()
    audio_np, _ = librosa.load(audio_path, sr=cfg.sample_rate)

    if segments is None:
        duration = len(audio_np) / cfg.sample_rate
        segments = [{"speaker": "SPEAKER_00", "start": 0, "end": duration}]

    words = transcribe(audio_path, model_name=cfg.whisper_model)
    phones = align(audio_path, words, cfg)

    _, sp_s, sp_f = identify_segments_from_mfa(phones, segments, cfg)
    if not sp_s and not sp_f:
        nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict as _cmudict
        cmu = _cmudict.dict()
        _, sp_s, sp_f = identify_segments_from_cmudict(words, segments, cmu, cfg)

    stop_data = extract_all_stop_measurements(sp_s, audio_np, cfg)
    fric_data = extract_all_fricative_measurements(sp_f, audio_np, cfg)
    return stop_data, fric_data


def compute_averages(vowel_data=None, stop_data=None, fricative_data=None, config=None):
    """Compute per-speaker averages with outlier rejection.

    Pass in data from extract_formants() and/or extract_consonants().

    Returns:
        dict with keys: vowel_averages, stop_averages, fricative_averages
        (only present if corresponding data was provided).
    """
    from .statistics import compute_fricative_averages, compute_stop_averages, compute_vowel_averages

    cfg = config or TAPAConfig()
    result = {}
    if vowel_data is not None:
        result["vowel_averages"] = compute_vowel_averages(vowel_data, cfg)
    if stop_data is not None:
        result["stop_averages"] = compute_stop_averages(stop_data, cfg)
    if fricative_data is not None:
        result["fricative_averages"] = compute_fricative_averages(fricative_data, cfg)
    return result
