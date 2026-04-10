"""Convenience wrappers for using pipeline components standalone."""

import os
import shutil
from pathlib import Path

import librosa
import numpy as np
import torch

from .config import TAPAConfig


class Models:
    """Shared model cache for efficient multi-file processing.

    Loads models once on first use, then reuses them across calls.

    Usage::

        from tapa import load_models, diarize, transcribe

        models = load_models()
        for f in audio_files:
            segments = diarize(f, models=models)
            words = transcribe(f, models=models)
    """

    def __init__(self, config=None):
        self.cfg = config or TAPAConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vad_model = None
        self._get_speech_timestamps = None
        self._voice_encoder = None
        self._whisper_model = None
        self._cmudict = None

    @property
    def vad(self):
        if self._vad_model is None:
            from .diarization import load_silero_vad
            print("  Loading Silero VAD...")
            self._vad_model, self._get_speech_timestamps = load_silero_vad()
        return self._vad_model, self._get_speech_timestamps

    @property
    def voice_encoder(self):
        if self._voice_encoder is None:
            from resemblyzer import VoiceEncoder
            print("  Loading Resemblyzer...")
            self._voice_encoder = VoiceEncoder()
        return self._voice_encoder

    @property
    def whisper(self):
        if self._whisper_model is None:
            import whisper
            print(f"  Loading Whisper ({self.cfg.whisper_model})...")
            self._whisper_model = whisper.load_model(self.cfg.whisper_model, device=self.device)
        return self._whisper_model

    @property
    def cmudict(self):
        if self._cmudict is None:
            import nltk
            nltk.download("cmudict", quiet=True)
            from nltk.corpus import cmudict as _cmudict
            self._cmudict = _cmudict.dict()
        return self._cmudict


def load_models(config=None, whisper_model=None):
    """Load all models upfront and return a Models cache.

    Use this when processing multiple files to avoid reloading models.

    Args:
        config: Optional TAPAConfig.
        whisper_model: Override whisper model name (e.g. "medium.en").

    Returns:
        Models instance to pass to diarize(), transcribe(), etc.
    """
    cfg = config or TAPAConfig()
    if whisper_model is not None:
        cfg.whisper_model = whisper_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"TAPA: Using device: {device_name} ({'CUDA' if device.type == 'cuda' else 'CPU'})")
    print("Loading models...")

    models = Models(cfg)
    # Eagerly load all models
    _ = models.vad
    _ = models.voice_encoder
    _ = models.whisper
    _ = models.cmudict

    print("Models loaded.\n")
    return models


def diarize(audio_path, num_speakers=None, config=None, models=None):
    """Diarize an audio file and return speaker-labeled segments.

    Args:
        audio_path: Path to audio file (.mp3, .wav, .flac).
        num_speakers: Number of speakers (None = auto-detect).
        config: Optional TAPAConfig.
        models: Optional Models cache from load_models().

    Returns:
        List of dicts with keys: speaker, start, end.
    """
    from .diarization import assign_speakers, get_speech_segments

    cfg = config or (models.cfg if models else TAPAConfig())
    if num_speakers is not None:
        cfg = TAPAConfig(**{**cfg.__dict__, "num_speakers": num_speakers})

    if models is None:
        models = Models(cfg)

    vad_model, get_speech_timestamps = models.vad
    segments, wav_t, sr = get_speech_segments(audio_path, vad_model, get_speech_timestamps, cfg)
    wav_np = wav_t.numpy().astype(np.float32)
    return assign_speakers(segments, wav_np, sr, models.voice_encoder, cfg)


def transcribe(audio_path, model_name=None, config=None, models=None):
    """Transcribe an audio file and return word-level timestamps.

    Args:
        audio_path: Path to audio file.
        model_name: Whisper model name. Ignored if models is provided.
        config: Optional TAPAConfig.
        models: Optional Models cache from load_models().

    Returns:
        List of dicts with keys: word, start, end.
    """
    from .transcription import transcribe_audio

    if models is not None:
        return transcribe_audio(audio_path, models.whisper)

    import whisper

    cfg = config or TAPAConfig()
    name = model_name or cfg.whisper_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(name, device=device)
    return transcribe_audio(audio_path, model)


def align(audio_path, words, config=None, models=None):
    """Run forced alignment on audio and return phone boundaries.

    Uses MFA if available, otherwise falls back to CMUdict proportional timing.

    Args:
        audio_path: Path to audio file.
        words: Word list from transcribe() (list of dicts with word, start, end).
        config: Optional TAPAConfig.
        models: Optional Models cache from load_models().

    Returns:
        List of dicts with keys: phone, start, end.
    """
    from .alignment import find_mfa_bin, parse_textgrid, prepare_mfa_input, run_mfa_alignment
    from .segments import identify_segments_from_cmudict

    cfg = config or (models.cfg if models else TAPAConfig())
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

    # CMUdict fallback
    cmu = models.cmudict if models else _load_cmudict()
    duration = words[-1]["end"] if words else 0
    dummy_segments = [{"speaker": "SPEAKER_00", "start": 0, "end": duration}]
    sp_v, sp_s, sp_f = identify_segments_from_cmudict(words, dummy_segments, cmu, cfg)
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


def extract_formants(audio_path, segments=None, config=None, models=None):
    """Extract vowel formants from an audio file.

    Args:
        audio_path: Path to audio file.
        segments: Diarization segments from diarize(). If None, treats entire
            audio as one speaker.
        config: Optional TAPAConfig.
        models: Optional Models cache from load_models().

    Returns:
        dict: {speaker: {vowel_ipa: [{"f1", "f2", "pitch", ...}]}}
    """
    from .segments import identify_segments_from_cmudict, identify_segments_from_mfa
    from .vowels import extract_all_vowel_formants

    cfg = config or (models.cfg if models else TAPAConfig())
    audio_np, _ = librosa.load(audio_path, sr=cfg.sample_rate)

    if segments is None:
        duration = len(audio_np) / cfg.sample_rate
        segments = [{"speaker": "SPEAKER_00", "start": 0, "end": duration}]

    words = transcribe(audio_path, config=cfg, models=models)
    phones = align(audio_path, words, cfg, models=models)

    sp_v, _, _ = identify_segments_from_mfa(phones, segments, cfg)
    if not sp_v:
        cmu = models.cmudict if models else _load_cmudict()
        sp_v, _, _ = identify_segments_from_cmudict(words, segments, cmu, cfg)

    return extract_all_vowel_formants(sp_v, audio_np, cfg)


def extract_consonants(audio_path, segments=None, config=None, models=None):
    """Extract stop VOT and fricative spectral moments from an audio file.

    Args:
        audio_path: Path to audio file.
        segments: Diarization segments from diarize(). If None, treats entire
            audio as one speaker.
        config: Optional TAPAConfig.
        models: Optional Models cache from load_models().

    Returns:
        tuple: (stop_data, fricative_data) where each is
        {speaker: {phone_ipa: [measurement_dicts]}}
    """
    from .consonants import extract_all_fricative_measurements, extract_all_stop_measurements
    from .segments import identify_segments_from_cmudict, identify_segments_from_mfa

    cfg = config or (models.cfg if models else TAPAConfig())
    audio_np, _ = librosa.load(audio_path, sr=cfg.sample_rate)

    if segments is None:
        duration = len(audio_np) / cfg.sample_rate
        segments = [{"speaker": "SPEAKER_00", "start": 0, "end": duration}]

    words = transcribe(audio_path, config=cfg, models=models)
    phones = align(audio_path, words, cfg, models=models)

    _, sp_s, sp_f = identify_segments_from_mfa(phones, segments, cfg)
    if not sp_s and not sp_f:
        cmu = models.cmudict if models else _load_cmudict()
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


def _load_cmudict():
    import nltk
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict as _cmudict
    return _cmudict.dict()
