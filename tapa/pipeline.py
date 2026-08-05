"""Main TAPA pipeline orchestrator."""

import os
import shutil
import warnings
from pathlib import Path

import nltk
import torch
import whisper
from resemblyzer import VoiceEncoder

from .alignment import (
    find_mfa_bin,
    parse_textgrid,
    parse_textgrids_dir,
    prepare_mfa_input,
    prepare_mfa_input_segmented,
    run_mfa_alignment,
)
from .audio import load_audio_16k
from .config import TAPAConfig
from .consonants import extract_all_fricative_measurements, extract_all_stop_measurements
from .diarization import (
    assign_speakers,
    get_speech_segments,
    load_silero_vad,
    save_diarization_csv,
)
from .download import download_youtube_audio, is_youtube_url
from .drvot import extract_all_stop_measurements_drvot
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
        print(f"[TAPA] Device: {device_name} ({'CUDA' if self.device.type == 'cuda' else 'CPU'})", flush=True)
        print(f"[TAPA] VOT backend: {self.cfg.vot_backend}"
              + (f"  (Dr.VOT repo: {self.cfg.drvot_repo_dir})"
                 if self.cfg.vot_backend == "drvot" else ""), flush=True)

        # Validate Dr.VOT setup up-front so missing prerequisites don't blow up
        # 5 minutes into the run (after diarization/transcription/alignment).
        if self.cfg.vot_backend == "drvot":
            self._ensure_drvot_ready()

        print("[TAPA 1/4] Loading Silero VAD...", flush=True)
        self.vad_model, self.get_speech_timestamps = load_silero_vad()

        print("[TAPA 2/4] Loading Resemblyzer (speaker embeddings)...", flush=True)
        self.voice_encoder = VoiceEncoder()

        print(f"[TAPA 3/4] Loading Whisper ({self.cfg.whisper_model})...", flush=True)
        self.whisper_model = whisper.load_model(self.cfg.whisper_model, device=self.device)

        print("[TAPA 4/4] Checking MFA...", flush=True)
        self.mfa_available = find_mfa_bin(self.cfg) is not None
        print(f"    MFA: {'found (precise phoneme alignment)' if self.mfa_available else 'NOT found — using CMUdict proportional fallback'}",
              flush=True)

        nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict as _cmudict
        self.cmudict = _cmudict.dict()

        self._models_loaded = True
        print("[TAPA] Models loaded.", flush=True)

    def _ensure_drvot_ready(self):
        """Validate (and auto-setup if needed) the Dr.VOT install before the run.

        Runs early in load_models() so a missing repo or weights doesn't waste
        the user's Whisper/MFA work.
        """
        if not self.cfg.drvot_repo_dir:
            raise RuntimeError(
                "vot_backend='drvot' requires drvot_repo_dir to be set. "
                "Pass TAPAConfig(drvot_repo_dir='/path/to/Dr.VOT'). "
                "On Colab the canonical location is '/content/Dr.VOT'."
            )
        repo = Path(self.cfg.drvot_repo_dir)
        weights = repo / "final_models" / "adv_model.model"
        if repo.exists() and weights.exists():
            print(f"[TAPA] Dr.VOT setup verified at {repo}", flush=True)
            return
        # Either the directory is missing entirely, or the clone is incomplete.
        # Auto-run setup. If praat or git is missing, this raises with an
        # actionable error before we've done any expensive work.
        from .drvot import setup_drvot
        if not repo.exists():
            print(f"[TAPA] Dr.VOT repo not found at {repo} — auto-running setup...",
                  flush=True)
        else:
            print(f"[TAPA] Dr.VOT clone at {repo} appears incomplete "
                  f"(missing {weights.name}) — re-running setup...", flush=True)
        setup_drvot(repo, force=False)
        print("[TAPA] Dr.VOT setup complete.", flush=True)

    def run(self, audio_path, results_dir=None):
        """Run the full pipeline on a single audio file or YouTube URL.

        Args:
            audio_path: Either a path to an audio file (.mp3, .wav, .flac) or
                a YouTube URL. URLs are downloaded to ``cfg.audio_dir`` as mp3
                first (filename = video ID, so result files trace back to the
                source recording).
            results_dir: Output directory. Defaults to config.results_dir.

        Returns:
            dict with keys: diarization, transcription, vowel_averages,
            stop_averages, fricative_averages, and file paths.
        """
        self.load_models()

        # Accept either a local path or a YouTube URL — download first if URL.
        if is_youtube_url(audio_path):
            print(f"[TAPA] Input is a YouTube URL — downloading mp3 to "
                  f"{self.cfg.audio_dir} @ {self.cfg.mp3_bitrate}k", flush=True)
            audio_path = download_youtube_audio(
                audio_path, self.cfg.audio_dir,
                bitrate=self.cfg.mp3_bitrate,
                cookies_file=self.cfg.youtube_cookies_file,
                cookies_from_browser=self.cfg.youtube_cookies_from_browser,
            )
            print(f"[TAPA] Saved {audio_path}", flush=True)

        results_dir = results_dir or self.cfg.results_dir
        os.makedirs(results_dir, exist_ok=True)
        stem = Path(audio_path).stem

        print(f"{'='*60}", flush=True)
        print(f"Processing: {Path(audio_path).name}", flush=True)
        print(f"{'='*60}", flush=True)

        # Decode once at 16 kHz mono; every stage below shares this buffer.
        # (Decoding natively and resampling in-process peaks at several GB for
        # multi-hour recordings — the old behavior that OOM'd Colab.)
        print("[TAPA] Decoding audio (16 kHz mono)...", flush=True)
        audio_np = load_audio_16k(audio_path, self.cfg.sample_rate)
        print(f"       -> {len(audio_np) / self.cfg.sample_rate / 60:.1f} min", flush=True)

        # Step 1: Diarization
        print("[STEP 1/6] Diarization (VAD + Resemblyzer clustering)...", flush=True)
        vad_segs, _, wav_sr = get_speech_segments(
            audio_np, self.vad_model, self.get_speech_timestamps, self.cfg)
        segments = assign_speakers(vad_segs, audio_np, wav_sr, self.voice_encoder, self.cfg)
        speakers = set(s["speaker"] for s in segments)
        print(f"          -> {len(segments)} segments / {len(speakers)} speaker(s)", flush=True)
        diar_path = os.path.join(results_dir, f"{stem}_diarization.csv")
        save_diarization_csv(segments, diar_path)

        # Step 2: Transcription
        print("[STEP 2/6] Transcription (Whisper)...", flush=True)
        words = transcribe_audio(audio_np, self.whisper_model)
        print(f"          -> {len(words)} words", flush=True)
        trans_path = os.path.join(results_dir, f"{stem}_transcription.csv")
        save_transcription(words, segments, trans_path)

        # Step 3: Forced alignment (MFA primary, CMUdict fallback)
        tg_path = None
        mfa_phones = None
        if self.mfa_available:
            split = self.cfg.mfa_split_utterances
            print("[STEP 3/6] Forced alignment (PRIMARY: Montreal Forced Aligner, "
                  + ("per-segment utterances" if split else "single utterance") + ")...",
                  flush=True)
            mfa_in = os.path.join(self.cfg.mfa_temp_dir, stem)
            mfa_out = os.path.join(self.cfg.mfa_temp_dir, f"{stem}_aligned")
            if split:
                offsets = prepare_mfa_input_segmented(
                    audio_path, words, segments, mfa_in, self.cfg, audio_np=audio_np)
                tg_path = run_mfa_alignment(mfa_in, mfa_out, self.cfg)
                if tg_path:
                    mfa_phones = parse_textgrids_dir(mfa_out, offsets)
                    tg_dest = os.path.join(results_dir, f"{stem}_aligned_textgrids")
                    shutil.rmtree(tg_dest, ignore_errors=True)
                    shutil.copytree(mfa_out, tg_dest)
                    print(f"          -> {len(offsets)} utterances aligned; TextGrids saved",
                          flush=True)
            else:
                prepare_mfa_input(audio_path, words, mfa_in, self.cfg, audio_np=audio_np)
                tg_path = run_mfa_alignment(mfa_in, mfa_out, self.cfg)
                if tg_path:
                    mfa_phones = parse_textgrid(tg_path)
                    tg_dest = os.path.join(results_dir, f"{stem}_aligned.TextGrid")
                    shutil.copy2(tg_path, tg_dest)
                    print("          -> MFA TextGrid saved", flush=True)
            if not tg_path:
                print("          -> MFA produced no TextGrid; using CMUdict fallback", flush=True)
        else:
            print("[STEP 3/6] Forced alignment (FALLBACK: CMUdict proportional, MFA unavailable)...",
                  flush=True)

        # Step 4: Identify phoneme segments
        print("[STEP 4/6] Identifying phoneme segments...", flush=True)
        if mfa_phones:
            print(f"          source: MFA  ({len(mfa_phones)} phones)", flush=True)
            sp_v, sp_s, sp_f = identify_segments_from_mfa(mfa_phones, segments, self.cfg)
        else:
            print("          source: CMUdict proportional timing", flush=True)
            sp_v, sp_s, sp_f = identify_segments_from_cmudict(words, segments, self.cmudict, self.cfg)

        nv = sum(len(v) for v in sp_v.values())
        ns = sum(len(v) for v in sp_s.values())
        nf = sum(len(v) for v in sp_f.values())
        print(f"          -> {nv} vowels, {ns} stops, {nf} fricatives", flush=True)

        # Step 5: Acoustic measurements (reuses the audio decoded up front)
        print("[STEP 5a]  Vowel formants (TAPA / Praat)...", flush=True)
        vowel_data = extract_all_vowel_formants(sp_v, audio_np, self.cfg)

        if self.cfg.vot_backend == "drvot":
            print("[STEP 5b]  Stop VOT (PRIMARY: Dr.VOT, FALLBACK per-token: TAPA / Praat)...",
                  flush=True)
            stop_data = extract_all_stop_measurements_drvot(sp_s, audio_np, self.cfg)
        else:
            print("[STEP 5b]  Stop VOT (TAPA / Praat)...", flush=True)
            stop_data = extract_all_stop_measurements(sp_s, audio_np, self.cfg)

        print("[STEP 5c]  Fricative spectral moments (TAPA / Praat)...", flush=True)
        fric_data = extract_all_fricative_measurements(sp_f, audio_np, self.cfg)

        # Step 6: Compute averages + save
        print("[STEP 6/6] Aggregating + saving results...", flush=True)
        v_avg = compute_vowel_averages(vowel_data, self.cfg)
        s_avg = compute_stop_averages(stop_data, self.cfg)
        f_avg = compute_fricative_averages(fric_data, self.cfg)

        save_json(vowel_data, os.path.join(results_dir, f"{stem}_vowel_formants.json"))
        save_vowel_averages_csv(v_avg, os.path.join(results_dir, f"{stem}_vowel_averages.csv"))
        save_json(stop_data, os.path.join(results_dir, f"{stem}_stop_vot.json"))
        save_stop_averages_csv(s_avg, os.path.join(results_dir, f"{stem}_stop_averages.csv"))
        save_json(fric_data, os.path.join(results_dir, f"{stem}_fricative_spectra.json"))
        save_fricative_averages_csv(f_avg, os.path.join(results_dir, f"{stem}_fricative_averages.csv"))

        align_method = "MFA" if mfa_phones else "CMUdict"
        vot_method = "Dr.VOT (+ TAPA fallback)" if self.cfg.vot_backend == "drvot" else "TAPA-Praat"
        print(f"\n[DONE] {Path(audio_path).name}  "
              f"alignment={align_method}, vot_backend={vot_method}", flush=True)
        for spk in sorted(v_avg.keys()):
            ntv = sum(d["n_tokens"] for d in v_avg[spk].values())
            nts = sum(d["n_tokens"] for d in s_avg.get(spk, {}).values())
            ntf = sum(d["n_tokens"] for d in f_avg.get(spk, {}).values())
            print(f"       {spk}: {ntv} vowels, {nts} stops, {ntf} fricatives", flush=True)

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
