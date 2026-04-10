"""Main TAPA pipeline orchestrator."""

import os
import shutil
import warnings
from pathlib import Path

import librosa
import nltk
import numpy as np
import torch
import whisper
from resemblyzer import VoiceEncoder

from .alignment import find_mfa_bin, parse_textgrid, prepare_mfa_input, run_mfa_alignment
from .config import TAPAConfig
from .consonants import extract_all_fricative_measurements, extract_all_stop_measurements
from .diarization import (
    assign_speakers,
    get_speech_segments,
    load_silero_vad,
    save_diarization_csv,
)
from .io import (
    save_fricative_averages_csv,
    save_json,
    save_stop_averages_csv,
    save_vowel_averages_csv,
)
from .segments import identify_segments_from_cmudict, identify_segments_from_mfa
from .statistics import compute_fricative_averages, compute_stop_averages, compute_vowel_averages
from .transcription import save_transcription, transcribe_audio
from .vowels import extract_all_vowel_formants

warnings.filterwarnings("ignore")


class TAPAPipeline:
    """Speaker diarization + phonetic analysis pipeline.

    Usage::

        from tapa import TAPAPipeline

        pipeline = TAPAPipeline()
        results = pipeline.run("interview.mp3")
    """

    def __init__(self, config=None):
        self.cfg = config or TAPAConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models_loaded = False
        self.vad_model = None
        self.get_speech_timestamps = None
        self.voice_encoder = None
        self.whisper_model = None
        self.cmudict = None
        self.mfa_available = False

    def load_models(self):
        """Load all ML models. Called automatically on first run()."""
        if self._models_loaded:
            return

        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        print(f"TAPA: Using device: {device_name} ({'CUDA' if self.device.type == 'cuda' else 'CPU'})")
        print("[1/4] Loading Silero VAD...")
        self.vad_model, self.get_speech_timestamps = load_silero_vad()

        print("[2/4] Loading Resemblyzer...")
        self.voice_encoder = VoiceEncoder()

        print(f"[3/4] Loading Whisper ({self.cfg.whisper_model})...")
        self.whisper_model = whisper.load_model(self.cfg.whisper_model, device=self.device)

        print("[4/4] Checking MFA...")
        self.mfa_available = find_mfa_bin(self.cfg) is not None
        print(f"    MFA: {'found' if self.mfa_available else 'not found (will use CMUdict fallback)'}")

        nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict as _cmudict
        self.cmudict = _cmudict.dict()

        self._models_loaded = True

    def run(self, audio_path, results_dir=None):
        """Run the full pipeline on a single audio file.

        Args:
            audio_path: Path to an audio file (.mp3, .wav, .flac).
            results_dir: Output directory. Defaults to config.results_dir.

        Returns:
            dict with keys: diarization, transcription, vowel_averages,
            stop_averages, fricative_averages, and file paths.
        """
        self.load_models()

        results_dir = results_dir or self.cfg.results_dir
        os.makedirs(results_dir, exist_ok=True)
        stem = Path(audio_path).stem

        print(f"{'='*60}")
        print(f"Processing: {Path(audio_path).name}")
        print(f"{'='*60}")

        # Step 1: Diarization
        print("  -> Diarizing...")
        vad_segs, wav_t, wav_sr = get_speech_segments(
            audio_path, self.vad_model, self.get_speech_timestamps, self.cfg)
        wav_np = wav_t.numpy().astype(np.float32)
        segments = assign_speakers(vad_segs, wav_np, wav_sr, self.voice_encoder, self.cfg)
        speakers = set(s["speaker"] for s in segments)
        print(f"    {len(segments)} segments -> {len(speakers)} speaker(s)")
        diar_path = os.path.join(results_dir, f"{stem}_diarization.csv")
        save_diarization_csv(segments, diar_path)

        # Step 2: Transcription
        print("  -> Transcribing...")
        words = transcribe_audio(audio_path, self.whisper_model)
        print(f"    {len(words)} words")
        trans_path = os.path.join(results_dir, f"{stem}_transcription.csv")
        save_transcription(words, segments, trans_path)

        # Step 3: Forced alignment (MFA or CMUdict fallback)
        tg_path = None
        if self.mfa_available:
            print("  -> Running MFA forced alignment...")
            mfa_in = os.path.join(self.cfg.mfa_temp_dir, stem)
            mfa_out = os.path.join(self.cfg.mfa_temp_dir, f"{stem}_aligned")
            prepare_mfa_input(audio_path, words, mfa_in, self.cfg)
            tg_path = run_mfa_alignment(mfa_in, mfa_out, self.cfg)
            if tg_path:
                tg_dest = os.path.join(results_dir, f"{stem}_aligned.TextGrid")
                shutil.copy2(tg_path, tg_dest)
                print("    TextGrid saved")

        # Step 4: Identify phoneme segments
        if tg_path:
            print("  -> Parsing MFA phoneme boundaries...")
            phones = parse_textgrid(tg_path)
            print(f"    {len(phones)} phones")
            sp_v, sp_s, sp_f = identify_segments_from_mfa(phones, segments, self.cfg)
        else:
            print("  -> CMUdict proportional timing fallback...")
            sp_v, sp_s, sp_f = identify_segments_from_cmudict(words, segments, self.cmudict, self.cfg)

        nv = sum(len(v) for v in sp_v.values())
        ns = sum(len(v) for v in sp_s.values())
        nf = sum(len(v) for v in sp_f.values())
        print(f"    {nv} vowels, {ns} stops, {nf} fricatives")

        # Step 5: Acoustic measurements
        print("  -> Loading audio for analysis...")
        audio_np, _ = librosa.load(audio_path, sr=self.cfg.sample_rate)

        print("  -> Extracting vowel formants...")
        vowel_data = extract_all_vowel_formants(sp_v, audio_np, self.cfg)
        print("  -> Extracting stop VOT...")
        stop_data = extract_all_stop_measurements(sp_s, audio_np, self.cfg)
        print("  -> Extracting fricative spectral moments...")
        fric_data = extract_all_fricative_measurements(sp_f, audio_np, self.cfg)

        # Step 6: Compute averages
        v_avg = compute_vowel_averages(vowel_data, self.cfg)
        s_avg = compute_stop_averages(stop_data, self.cfg)
        f_avg = compute_fricative_averages(fric_data, self.cfg)

        # Step 7: Save results
        save_json(vowel_data, os.path.join(results_dir, f"{stem}_vowel_formants.json"))
        save_vowel_averages_csv(v_avg, os.path.join(results_dir, f"{stem}_vowel_averages.csv"))
        save_json(stop_data, os.path.join(results_dir, f"{stem}_stop_vot.json"))
        save_stop_averages_csv(s_avg, os.path.join(results_dir, f"{stem}_stop_averages.csv"))
        save_json(fric_data, os.path.join(results_dir, f"{stem}_fricative_spectra.json"))
        save_fricative_averages_csv(f_avg, os.path.join(results_dir, f"{stem}_fricative_averages.csv"))

        # Print summary
        method = "MFA" if tg_path else "CMUdict"
        print(f"\n  Done ({method}):")
        for spk in sorted(v_avg.keys()):
            ntv = sum(d["n_tokens"] for d in v_avg[spk].values())
            nts = sum(d["n_tokens"] for d in s_avg.get(spk, {}).values())
            ntf = sum(d["n_tokens"] for d in f_avg.get(spk, {}).values())
            print(f"    {spk}: {ntv} vowels, {nts} stops, {ntf} fricatives")

        # Cleanup MFA temp
        if os.path.exists(self.cfg.mfa_temp_dir):
            shutil.rmtree(self.cfg.mfa_temp_dir, ignore_errors=True)

        return {
            "diarization": segments,
            "words": words,
            "vowel_data": vowel_data,
            "stop_data": stop_data,
            "fricative_data": fric_data,
            "vowel_averages": v_avg,
            "stop_averages": s_avg,
            "fricative_averages": f_avg,
            "results_dir": results_dir,
        }

    def run_batch(self, audio_dir=None, results_dir=None):
        """Run the pipeline on all audio files in a directory.

        Args:
            audio_dir: Directory containing audio files. Defaults to config.audio_dir.
            results_dir: Output directory. Defaults to config.results_dir.

        Returns:
            dict mapping filename to results.
        """
        audio_dir = audio_dir or self.cfg.audio_dir
        results_dir = results_dir or self.cfg.results_dir
        extensions = (".mp3", ".wav", ".flac")
        audio_files = sorted([
            f for f in os.listdir(audio_dir)
            if f.endswith(extensions)
        ])
        print(f"Found {len(audio_files)} audio file(s) in {audio_dir}\n")
        all_results = {}
        for audio_file in audio_files:
            audio_path = os.path.join(audio_dir, audio_file)
            all_results[audio_file] = self.run(audio_path, results_dir)
            print()
        print(f"{'='*60}")
        print("Pipeline complete!")
        print(f"{'='*60}")
        return all_results
