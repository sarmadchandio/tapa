# TAPA — Text and Phonetic Analysis

Speaker diarization + phonetic analysis of audio recordings. Given an audio
file (or a YouTube URL), TAPA identifies who is speaking, transcribes what
they say, and extracts detailed acoustic measurements per speaker.

## What it does

1. **Speaker diarization** — identifies speakers and their time boundaries
2. **Transcription** — word-level transcript with timestamps (Whisper)
3. **Phoneme alignment** — precise phoneme boundaries (MFA, with CMUdict
   proportional fallback when MFA isn't installed)
4. **Vowel formants** — F1, F2, and pitch for each vowel token
5. **Stop consonant VOT** — Voice Onset Time. Backend is selectable:
   - `"tapa"` (default) — Praat-based signal heuristic; fast, deterministic
   - `"drvot"` — Dr.VOT CNN (Shrem et al. 2019); handles negative VOT
     (prevoicing); slower; per-token TAPA fallback for tokens it can't predict
6. **Fricative spectral moments** — Center of Gravity, spectral SD, skewness,
   kurtosis
7. **Per-speaker averages** — summary statistics with outlier rejection

Inputs accepted: local `.mp3` / `.wav` / `.flac` files, **and YouTube URLs**
(downloaded to mp3 via yt-dlp before processing).

---

## Quick Start: Google Colab end-to-end walkthrough

This is the canonical path for a new analysis. Open a fresh Colab notebook
with a **GPU runtime** (Runtime → Change runtime type → GPU) and paste these
cells in order.

### Cell 1 — install TAPA (with the optional Dr.VOT extra)

```python
# System deps. praat + sox are only needed if you'll use the Dr.VOT backend.
!apt-get install -y -qq praat sox

# Install TAPA. Drop the [drvot] suffix if you only want the default Praat-based
# VOT backend.
!pip install -q "tapa[drvot] @ git+https://github.com/sarmadchandio/tapa.git"
```

### Cell 2 — install MFA (recommended; gives precise phoneme boundaries)

Skip this cell to fall back to CMUdict proportional timing — TAPA still works,
just with less accurate phoneme boundaries.

```python
import os
if not os.path.exists("/opt/miniforge"):
    !wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    !bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge > /dev/null 2>&1

!/opt/miniforge/bin/mamba install -c conda-forge montreal-forced-aligner -y -q 2>&1 | tail -3
!/opt/miniforge/bin/mfa model download acoustic english_us_arpa
!/opt/miniforge/bin/mfa model download dictionary english_us_arpa

os.environ["PATH"] = "/opt/miniforge/bin:" + os.environ["PATH"]
```

### Cell 3 — clone + patch Dr.VOT (skip if using `vot_backend="tapa"`)

```python
!python -m tapa.drvot setup /content/Dr.VOT
```

This is idempotent — safe to re-run. It clones the upstream Dr.VOT repo,
patches a hard-coded Praat path inside it, and chmods the bundled feature
extractor binary.

### Cell 4 — run the pipeline

```python
from tapa import TAPAPipeline, TAPAConfig

cfg = TAPAConfig(
    mfa_bin="/opt/miniforge/bin/mfa",   # remove if you skipped Cell 2
    vot_backend="drvot",                # or "tapa" for the Praat-only path
    drvot_repo_dir="/content/Dr.VOT",   # remove if vot_backend="tapa"
)
pipeline = TAPAPipeline(config=cfg)

# Either a YouTube URL (auto-downloaded to ./audio/<video_id>.mp3)...
results = pipeline.run("https://www.youtube.com/watch?v=DPO7imV0LHg")

# ...or a local file:
# results = pipeline.run("/content/my_recording.mp3")
```

Result CSVs/JSONs land in `./results/` (configurable via `cfg.results_dir`
or the `results_dir` argument to `run()`). The video ID is the filename stem,
so for the URL above you get `DPO7imV0LHg_diarization.csv`,
`DPO7imV0LHg_vowel_averages.csv`, etc.

### Cell 5 — process several recordings in one session

Models are cached on the pipeline instance, so additional `run()` calls don't
reload Whisper / Resemblyzer / Silero / MFA.

```python
urls = [
    "https://www.youtube.com/watch?v=DPO7imV0LHg",
    "https://www.youtube.com/watch?v=...",
    "https://www.youtube.com/watch?v=...",
]
for u in urls:
    pipeline.run(u)
```

Or for a directory of local files:

```python
pipeline.run_batch(audio_dir="/content/recordings/", results_dir="/content/results/")
```

(`run_batch` does not currently accept URL lists — pass URLs through `run()`
in a loop as above.)

### What you should see at runtime

Stage banners stream live to the cell output. Key lines to watch:

```
[TAPA] Device: Tesla T4 (CUDA)
[TAPA] VOT backend: drvot  (Dr.VOT repo: /content/Dr.VOT)
[STEP 1/6] Diarization (VAD + Resemblyzer clustering)...
[STEP 2/6] Transcription (Whisper)...
[STEP 3/6] Forced alignment (PRIMARY: Montreal Forced Aligner)...
[STEP 4/6] Identifying phoneme segments...
          source: MFA  (16983 phones)
          -> 3372 vowels, 1240 stops, 1855 fricatives
[STEP 5b]  Stop VOT (PRIMARY: Dr.VOT, FALLBACK per-token: TAPA / Praat)...
[DrVOT] prepared 1132 clips for inference (skipped: 88 no-following-vowel, 20 too-short)
[DrVOT/predict] Model runs on : cpu
[DrVOT] coverage: 1118 Dr.VOT (98.8%), 14 TAPA-Praat fallback (1.2%), 0 dropped (0.0%)
[DONE] DPO7imV0LHg.mp3  alignment=MFA, vot_backend=Dr.VOT (+ TAPA fallback)
```

Pay attention to the `coverage:` line — it tells you how many stops Dr.VOT
actually scored vs how many fell back to TAPA-Praat. If the fallback rate is
high (>30%) on the first run, the clip window may be too tight; bump
`drvot_clip_pre_ms` / `drvot_clip_post_ms`.

### Wall-clock budget

For a 30-minute recording on a Colab T4 GPU runtime:

| Step | TAPA-only | TAPA + Dr.VOT |
|------|-----------|---------------|
| Whisper + diarization + MFA | ~3 min | ~3 min |
| Vowel formants + fricative moments | ~1 min | ~1 min |
| Stop VOT | <1 min | ~15–25 min (CPU) |
| **Total** | **~5 min** | **~20–30 min** |

Dr.VOT is CPU-only (the model is small enough that GPU offers no benefit).
First-ever run also pays a one-time ~5 min cost downloading Whisper, MFA
acoustic + dictionary, Silero, and Resemblyzer weights.

---

## Backend choices: TAPA vs Dr.VOT

| | TAPA (default) | Dr.VOT |
|---|---|---|
| Method | Praat: intensity peak → first f0 cycle | CNN trained on labeled VOTs (Shrem et al. 2019) |
| Speed | ~ms / token | ~1 s / token (CPU) |
| Negative VOT (prevoicing) | not handled | handled (POS / NEG class output) |
| Robustness on noisy/coarticulated speech | brittle | substantially better |
| Languages | language-agnostic in principle | English-trained; degrades on others |
| Extra setup | none | `python -m tapa.drvot setup <dir>` |

When you choose `vot_backend="drvot"`, every stop token in the output JSON
gets two extra fields: `vot_method` (`"drvot"` or `"tapa-fallback"`) and
`vot_class_drvot` (`"POS"` for aspirated, `"NEG"` for prevoiced, `null` for
fallback rows). The aggregated `*_stop_averages.csv` is unchanged so existing
analysis code keeps working.

---

## Installation (other environments)

```bash
# Core install
pip install git+https://github.com/sarmadchandio/tapa.git

# With Dr.VOT extras
pip install "tapa[drvot] @ git+https://github.com/sarmadchandio/tapa.git"
```

Requires **Python 3.10+** and **ffmpeg**. yt-dlp ≥ 2025 is the version that
keeps up with current YouTube; `pip install` always pulls the latest.

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

CUDA is used automatically when available (Whisper + Resemblyzer). Dr.VOT
itself runs on CPU. For Dr.VOT you also need `praat` and `sox` on PATH —
`apt-get install praat sox` on Ubuntu, `brew install praat sox` on macOS.

### MFA setup (non-Colab)

```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

### Dr.VOT setup (non-Colab)

```bash
python -m tapa.drvot setup ~/Dr.VOT
# then, when running TAPA:
tapa interview.mp3 --vot-backend drvot --drvot-repo ~/Dr.VOT
```

---

## Command line

```bash
# Local file
tapa interview.mp3
tapa file1.mp3 file2.wav -o my_results/

# YouTube URL (downloaded to --audio-dir, default audio/)
tapa "https://youtu.be/DPO7imV0LHg" -o my_results/
tapa "https://www.youtube.com/watch?v=DPO7imV0LHg" --audio-dir downloads/ --mp3-bitrate 256

# With Dr.VOT backend
tapa "https://youtu.be/DPO7imV0LHg" --vot-backend drvot --drvot-repo /content/Dr.VOT

# Other knobs
tapa podcast.mp3 --num-speakers 3 --whisper-model medium.en
```

Run `tapa --help` for the full flag list.

---

## Output files

For each recording (e.g. `interview.mp3` or `<video_id>.mp3`), TAPA writes
to the results directory:

| File | Description |
|------|-------------|
| `<stem>_diarization.csv` | Speaker segments (`speaker, start, end`) |
| `<stem>_transcription.csv` | Word-level transcript with speaker labels |
| `<stem>_transcription.txt` | Human-readable transcript |
| `<stem>_aligned.TextGrid` | MFA phoneme alignment (only if MFA installed) |
| `<stem>_vowel_formants.json` | Raw vowel F1/F2/pitch per token |
| `<stem>_vowel_averages.csv` | Per-speaker per-vowel average formants |
| `<stem>_stop_vot.json` | Raw stop VOT measurements per token |
| `<stem>_stop_averages.csv` | Per-speaker per-stop average VOT |
| `<stem>_fricative_spectra.json` | Raw fricative spectral moments per token |
| `<stem>_fricative_averages.csv` | Per-speaker per-fricative averages |

**When `vot_backend="drvot"`**, each token in `*_stop_vot.json` carries two
extra fields: `vot_method` (`"drvot"` / `"tapa-fallback"`) and
`vot_class_drvot` (`"POS"` / `"NEG"` / `null`). `burst_time` and
`voicing_onset` are populated only for the TAPA-Praat path.

### Sample output

`<stem>_vowel_averages.csv`:

```
speaker,vowel,mean_f1,mean_f2,std_f1,std_f2,mean_pitch,n_tokens,n_after_filtering
SPEAKER_00,i,393.0,2110.0,54.0,227.0,142.3,88,74
SPEAKER_00,æ,656.0,1634.0,124.0,176.0,138.5,72,63
```

`<stem>_stop_averages.csv`:

```
speaker,phone,voicing,place,mean_vot_ms,std_vot_ms,mean_closure_ms,n_tokens,n_after_filtering
SPEAKER_00,p,voiceless,bilabial,0.6,0.4,78.32,42,38
SPEAKER_00,t,voiceless,alveolar,4.2,12.0,62.15,104,104
```

### Working with results in pandas

```python
import pandas as pd
vowels = pd.read_csv("results/interview_vowel_averages.csv")
print(vowels[vowels["speaker"] == "SPEAKER_00"][["vowel", "mean_f1", "mean_f2", "n_after_filtering"]])
```

### Working with the results dict

```python
results = pipeline.run("interview.mp3")

for seg in results["diarization"][:5]:
    print(f"{seg['speaker']}: {seg['start']:.1f}s - {seg['end']:.1f}s")

for speaker, vowels in results["vowel_data"].items():
    for vowel_ipa, measurements in vowels.items():
        print(f"{speaker} /{vowel_ipa}/: {len(measurements)} tokens")
```

---

## Per-component API (advanced)

You don't need this for normal use — `TAPAPipeline.run()` does it all. But
the steps are exposed individually for custom workflows.

```python
from tapa import (
    load_models, diarize, transcribe, align,
    extract_formants, extract_consonants, compute_averages,
)

models = load_models()                                # load once, reuse
segments = diarize("interview.mp3", models=models)
words = transcribe("interview.mp3", models=models)
phones = align("interview.mp3", words, models=models)
formants = extract_formants("interview.mp3", segments, models=models)
stop_data, fricative_data = extract_consonants("interview.mp3", segments, models=models)
avgs = compute_averages(vowel_data=formants, stop_data=stop_data, fricative_data=fricative_data)
```

Every function takes optional `config=` and `models=` parameters. Without
`models=`, each call reloads everything from scratch.

YouTube downloads are also exposed:

```python
from tapa import download_youtube_audio, is_youtube_url
mp3 = download_youtube_audio("https://youtu.be/DPO7imV0LHg", "audio/", bitrate="192")
```

Dr.VOT internals are exposed if you want to call it directly on already-
identified stop tokens:

```python
from tapa import setup_drvot, extract_all_stop_measurements_drvot
# setup_drvot("/content/Dr.VOT")  # one-time
stop_data = extract_all_stop_measurements_drvot(speaker_stops, audio_np, cfg)
```

---

## Configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio_dir` | `"audio/"` | Where to save audio downloaded from URLs |
| `results_dir` | `"results/"` | Output directory |
| `mfa_temp_dir` | `"mfa_temp/"` | Scratch dir for MFA |
| `sample_rate` | `16000` | Audio sample rate for processing |
| `whisper_model` | `"small.en"` | Whisper model (`tiny.en`, `base.en`, `small.en`, `medium.en`, `large`) |
| `mfa_bin` | `None` | Path to MFA binary (`None` = auto-detect) |
| `num_speakers` | `None` | Number of speakers (`None` = auto-detect) |
| `min_segment_duration` | `0.1` | Minimum speech segment duration (seconds) |
| `merge_gap` | `0.5` | Merge same-speaker segments closer than this (seconds) |
| `min_vowel_duration` | `0.03` | Minimum vowel duration to analyze (seconds) |
| `vowel_trim_fraction` | `0.15` | Fraction to trim from vowel edges before measurement |
| `f1_min` / `f1_max` | `150` / `1500` | Valid F1 range (Hz) |
| `f2_min` / `f2_max` | `400` / `4000` | Valid F2 range (Hz) |
| `min_stop_duration` | `0.015` | Minimum stop consonant duration (seconds) |
| `min_fricative_duration` | `0.03` | Minimum fricative duration (seconds) |
| `vot_max` | `0.150` | Maximum valid VOT (seconds) — TAPA backend only |
| `mad_threshold` | `2.0` | MAD outlier rejection threshold |
| `target_vowels` | `None` | Set of IPA vowels to analyze (`None` = all) |
| **YouTube** | | |
| `mp3_bitrate` | `"192"` | yt-dlp `preferredquality` for URL downloads (kbps as string) |
| **VOT backend** | | |
| `vot_backend` | `"tapa"` | `"tapa"` (Praat-based) or `"drvot"` (Dr.VOT CNN + per-token TAPA fallback) |
| `drvot_repo_dir` | `None` | Path to a Dr.VOT clone — required when `vot_backend="drvot"` |
| `drvot_python` | `None` | Python interpreter for Dr.VOT subprocesses (`None` = current Python) |
| `drvot_clip_pre_ms` | `150.0` | Padding before stop closure when cutting clips for Dr.VOT |
| `drvot_clip_post_ms` | `150.0` | Padding after the following vowel |
| `drvot_keep_temp` | `False` | Keep the per-recording Dr.VOT temp dir for inspection |

Supported audio formats: `.mp3`, `.wav`, `.flac`. URLs: any standard YouTube
URL form (`youtube.com/watch?v=…`, `youtu.be/…`, `youtube.com/shorts/…`).

---

## License

MIT
