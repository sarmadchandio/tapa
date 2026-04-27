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
   - `"drvot"` — Dr.VOT, a neural-network model (Shrem et al. 2019); handles
     negative VOT (prevoicing); slower; with automatic per-token TAPA fallback
     for stops it can't predict
6. **Fricative spectral moments** — Center of Gravity, spectral SD, skewness,
   kurtosis
7. **Per-speaker averages** — summary statistics with outlier rejection

Inputs accepted: local `.mp3` / `.wav` / `.flac` files, **and YouTube URLs**
(downloaded to mp3 automatically before processing).

---

## Before you begin

- **You need a Google account.** The walkthrough below uses Google Colab —
  a free, browser-based notebook environment.
- **The pipeline works best on English audio.** Whisper has multilingual
  models you can swap in (`whisper_model="medium"`), but Dr.VOT is
  English-trained, so for non-English recordings use the default
  `vot_backend="tapa"`.
- **Speaker labels are auto-assigned.** Diarization produces labels like
  `SPEAKER_00`, `SPEAKER_01` in the order speakers are first heard. The
  mapping to real names is up to you. If you know the number of speakers,
  set `num_speakers=N` for more reliable clustering.
- **Plan for time.** A 30-minute recording takes about 5 min with the
  default backend, or ~25 min with the Dr.VOT backend. See the wall-clock
  table below.

---

## Quick Start: Google Colab end-to-end walkthrough

This is the canonical path for a new analysis.

1. Open <https://colab.research.google.com>, sign in, and click
   **File → New Notebook**.
2. Switch on the GPU: **Runtime → Change runtime type → T4 GPU → Save**.
3. Paste each cell below, in order, into separate notebook cells and run
   them top-to-bottom.

### Cell 1 — install TAPA

Decide first whether you want the Dr.VOT backend (more accurate stop VOT,
slower) or the default Praat-based backend (fast, deterministic). Compare
them in the **[Backend choices](#backend-choices-tapa-vs-drvot)** section
below before continuing.

If you want **only the default backend**:

```python
!apt-get install -y -qq ffmpeg
!pip install -q "git+https://github.com/sarmadchandio/tapa.git"
```

If you want **the Dr.VOT backend** (recommended for stop-VOT analysis):

```python
# Dr.VOT ships its own GUI-Praat binary that needs GTK 2 at runtime, so we
# install libgtk2.0-0 + a couple of friends along with praat and sox.
# ffmpeg is needed for YouTube downloads.
!apt-get install -y -qq ffmpeg praat sox libgtk2.0-0 libglib2.0-0 libxtst6
# The "tapa[drvot]" form tells pip to also install Dr.VOT's extra deps
# (boltons, pydub, textgrid, noisereduce). The "@ git+..." form tells pip
# to install from this GitHub repo rather than from PyPI.
!pip install -q "tapa[drvot] @ git+https://github.com/sarmadchandio/tapa.git"
```

### Cell 2 — install MFA (recommended)

The Montreal Forced Aligner gives precise phoneme boundaries. **Skip this
cell** if you don't need them — TAPA falls back to a less accurate
proportional-timing method based on CMUdict.

```python
# This installs Miniforge (a conda variant), then uses it to install MFA
# and download the English acoustic + pronunciation dictionary models.
# First run takes 2–4 minutes — be patient.
import os
if not os.path.exists("/opt/miniforge"):
    !wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    !bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge > /dev/null 2>&1

!/opt/miniforge/bin/mamba install -c conda-forge montreal-forced-aligner -y -q 2>&1 | tail -3
!/opt/miniforge/bin/mfa model download acoustic english_us_arpa
!/opt/miniforge/bin/mfa model download dictionary english_us_arpa

os.environ["PATH"] = "/opt/miniforge/bin:" + os.environ["PATH"]
```

### Cell 3 — Dr.VOT setup is automatic

**No cell needed.** When you run the pipeline below with
`vot_backend="drvot"`, TAPA automatically clones Dr.VOT into the path you
gave for `drvot_repo_dir`, patches the hard-coded paths inside it, and
verifies the feature-extractor binary — all in-process. Watch for the
`[TAPA] Dr.VOT repo not found at ... — auto-running setup...` line in the
log on the first run.

If you ever want to run setup explicitly (e.g. to pre-warm the clone),
**you must use the explicit Python path** because Cell 2 puts miniforge's
Python first on `PATH` and miniforge doesn't have `tapa` installed:

```python
import sys
!{sys.executable} -m tapa.drvot setup /content/Dr.VOT
```

A bare `!python -m tapa.drvot ...` after Cell 2 will fail with
`ModuleNotFoundError: No module named 'tapa'` for that reason.

### Cell 4 — run the pipeline

Pick the variant that matches the choices you made above and paste it in.

**Variant A — default backend, MFA, YouTube URL:**

```python
from tapa import TAPAPipeline, TAPAConfig

cfg = TAPAConfig(mfa_bin="/opt/miniforge/bin/mfa")
pipeline = TAPAPipeline(config=cfg)
results = pipeline.run("https://www.youtube.com/watch?v=DPO7imV0LHg")
```

**Variant B — Dr.VOT backend, MFA, YouTube URL:**

```python
from tapa import TAPAPipeline, TAPAConfig

cfg = TAPAConfig(
    mfa_bin="/opt/miniforge/bin/mfa",
    vot_backend="drvot",
    drvot_repo_dir="/content/Dr.VOT",
)
pipeline = TAPAPipeline(config=cfg)
results = pipeline.run("https://www.youtube.com/watch?v=DPO7imV0LHg")
```

**Variant C — local audio file instead of a URL:**

Replace the URL with a path to a file you uploaded to Colab (left sidebar
→ folder icon → upload):

```python
results = pipeline.run("/content/my_recording.mp3")
```

Result CSVs and JSONs land in `./results/`. The video ID (or local
filename) becomes the stem, so for the URL above you'll get
`DPO7imV0LHg_diarization.csv`, `DPO7imV0LHg_vowel_averages.csv`, etc.

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

Progress messages stream live to the cell output as each stage runs. Lines
beginning with `[TAPA]` come from the main pipeline; `[DrVOT]` lines come
from the Dr.VOT subprocess.

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

The `coverage:` line is the most important one to glance at. It tells you
how many stop consonants Dr.VOT successfully measured versus how many
needed the TAPA-Praat fallback. If the fallback rate is high (>30%) on a
clean recording, the audio window we send to Dr.VOT may be too tight; you
can widen it with `drvot_clip_pre_ms=200, drvot_clip_post_ms=200` in the
config.

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
| Method | Praat: intensity peak → first f0 cycle | Neural-network model trained on labeled VOTs (Shrem et al. 2019) |
| Speed | ~ms per token | ~1 s per token (CPU) |
| Negative VOT (prevoicing) | not handled | handled (`POS` aspirated / `NEG` prevoiced output) |
| Robustness on noisy or coarticulated speech | brittle | substantially better |
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

### Reading the measurements

For readers who haven't worked with these acoustic measurements before:

- **F1, F2** (Hz) — vowel formants. F1 is roughly inversely related to vowel
  height (lower F1 ≈ closer/higher vowel like /i/, /u/; higher F1 ≈ more
  open vowel like /æ/, /ɑ/). F2 reflects backness (higher F2 ≈ more front
  like /i/; lower F2 ≈ more back like /u/). A scatter of `mean_f2` (x, axis
  reversed) vs `mean_f1` (y, axis reversed) is the classic "vowel space"
  plot.
- **VOT** (ms) — voice onset time. Time from a stop's burst release to the
  start of voicing in the following segment. Aspirated voiceless stops
  (English /p t k/) have positive VOT (~30–100 ms); prevoiced stops (some
  varieties of /b d g/) have negative VOT.
- **Spectral CoG** (Hz) — center of gravity of a fricative's spectrum.
  Sibilants like /s, ʃ/ have high CoG (>3000 Hz); /f, θ/ have lower CoG.
  Skewness, kurtosis, and spectral SD are the higher-order moments.
- **`n_tokens` vs `n_after_filtering`** — total tokens identified vs how
  many survived MAD outlier rejection (default threshold = 2). Use
  `n_after_filtering` when reporting averages.

### Working with results in pandas

```python
import pandas as pd
vowels = pd.read_csv("results/DPO7imV0LHg_vowel_averages.csv")
print(vowels[vowels["speaker"] == "SPEAKER_00"][["vowel", "mean_f1", "mean_f2", "n_after_filtering"]])

# Quick vowel-space plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
sub = vowels[vowels["speaker"] == "SPEAKER_00"]
ax.scatter(sub["mean_f2"], sub["mean_f1"])
for _, r in sub.iterrows():
    ax.annotate(r["vowel"], (r["mean_f2"], r["mean_f1"]))
ax.invert_xaxis(); ax.invert_yaxis()
ax.set_xlabel("F2 (Hz)"); ax.set_ylabel("F1 (Hz)")
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

## Common issues

**"`No video formats found`" / yt-dlp errors when passing a URL.**
Your runtime has a stale `yt-dlp`. Restart the Colab runtime (Runtime →
Disconnect and delete runtime), then re-run the install cell — it pulls
the latest version. On a non-Colab machine: `pip install -U yt-dlp` and
retry.

**Whisper / diarization runs on CPU instead of GPU.** The `[TAPA] Device:`
line will say `CPU` — meaning you forgot to switch the runtime. Click
Runtime → Change runtime type → T4 GPU → Save, then run the cells again.

**MFA install seems hung in Cell 2.** It legitimately takes 2–4 minutes the
first time (Miniforge download + conda solve + acoustic + dictionary
models). It's quick on subsequent runs in the same session.

**Speaker labels seem off.** Diarization is automatic — `SPEAKER_00`,
`SPEAKER_01` are assigned in order of first appearance, not in the order
you'd expect. If you know the number of speakers, set
`TAPAConfig(num_speakers=2)` for cleaner clustering. Then rename them by
post-processing the CSVs.

**Dr.VOT coverage is low (<70%).** Two common causes: (a) clip windows are
too tight — pass `drvot_clip_pre_ms=200, drvot_clip_post_ms=200` in
`TAPAConfig`; (b) the recording is heavily reverberant or noisy and Dr.VOT
genuinely can't anchor — the TAPA-Praat fallback will still measure those
tokens.

**Non-English audio.** Use `TAPAConfig(whisper_model="medium")` (not
`medium.en`). Stick to `vot_backend="tapa"` since Dr.VOT was English-trained.

**"Out of memory" on long recordings.** Free-tier Colab has ~12 GB RAM. For
recordings longer than ~1 hour use `TAPAConfig(whisper_model="tiny.en")`,
or split the audio first with ffmpeg.

**Colab session disconnected mid-run.** Free-tier sessions time out after
~90 minutes idle. For long batches use a Colab Pro runtime, or save your
intermediate results to Google Drive (`drive.mount("/content/drive")` then
set `results_dir="/content/drive/MyDrive/tapa_results/"`).

---

## Citing

If you use this pipeline in academic work, please cite:

- The Dr.VOT paper (when using `vot_backend="drvot"`):
  Shrem, Goldrick & Keshet (2019). "Dr.VOT: Measuring Positive and Negative
  Voice Onset Time in the Wild." *Interspeech 2019*, 629–633.
- Whisper, Resemblyzer, Praat / parselmouth, and MFA, as appropriate.

---

## License

MIT
