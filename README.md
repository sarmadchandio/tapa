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
   - `"drvot"` — Dr.VOT, a neural-network model (Shrem et al. 2019); handles
     negative VOT (prevoicing); slower; with automatic per-token TAPA fallback
     for stops it can't predict. **Use this for any VOT result you intend to
     report**, and pair it with MFA — see the warning below.
   - `"tapa"` (default) — Praat-based signal heuristic; fast, deterministic,
     and needs no setup, but unreliable on conversational speech: measured on
     a 2-minute sample it returned a median of 0.5 ms for both voiceless and
     voiced stops, with roughly 70 % of tokens at that floor, where Dr.VOT
     gave 22 ms and 16 ms on the same audio. It remains the default only
     because it requires no extra installation.
6. **Fricative spectral moments** — Center of Gravity, spectral SD, skewness,
   kurtosis
7. **Per-speaker averages** — summary statistics with outlier rejection

Inputs accepted: local `.mp3` / `.wav` / `.flac` files, **and YouTube URLs**
(downloaded to mp3 automatically before processing).

---

## Before you begin

- **Two ways to run this.** In Google Colab (nothing to install, free GPU) or
  on your own machine. [Which setup do I need?](#which-setup-do-i-need)
  compares them; follow one section and ignore the other.
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
- **YouTube downloads use your cookies by default**, including your browser's
  own YouTube session when one is available, so YouTube attributes the
  download to your account. See [Usage policy](#usage-policy) for what is sent
  where and how to switch it off. Local audio files never touch the network.

---

## Tutorial notebook

The fastest way in is the tutorial notebook, which runs end to end in Colab
with no local setup, explains each stage, and walks through every output file:

**[Open the tutorial in Colab](https://colab.research.google.com/github/sarmadchandio/tapa/blob/master/notebooks/tapa_tutorial.ipynb)**
 · [view the notebook](notebooks/tapa_tutorial.ipynb)

It is preset-driven — `quick` for a two-minute smoke test, `recommended`
(MFA + Dr.VOT) for analysis you intend to report — with every individual
setting exposed and documented. The sections below cover the same ground in
prose, and remain the reference for running TAPA outside Colab.

## Which setup do I need?

Two ways to run TAPA. Pick one and follow only that section — everything after
them (backends, outputs, configuration, troubleshooting) applies to both.

| | **Google Colab** | **Local machine** |
|---|---|---|
| Setup | nothing to install; runs in a browser | Python 3.10+, ffmpeg, and a package manager |
| GPU | free T4, no hardware needed | your own, or CPU (slower) |
| Recording length | up to ~5 hours; the session ends when you close it | limited only by your disk |
| Sensitive audio | uploaded to Google's servers | never leaves your machine |
| Best for | trying TAPA, one-off analyses, no local setup | studies, batches, restricted data |

**→ [Set up on Google Colab](#setup-google-colab)**  ·  **→ [Set up locally](#setup-local-machine)**

If you are working with interviews or any identifiable participants, use the
local route and read the [usage policy](#usage-policy) first.

---

## Setup: Google Colab

*Running locally instead? Skip to [Set up locally](#setup-local-machine).*

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
# Dr.VOT calls Praat and sox while extracting features; without either, every
# token silently falls back to the Praat estimator. We install Praat's
# "barren" build, which is headless — Dr.VOT bundles a GUI Praat that needs
# GTK 2 and fails on a stock Colab image. ffmpeg is needed for YouTube.
!apt-get install -y -qq ffmpeg sox
!wget -q -O /tmp/praat.tar.gz https://github.com/praat/praat/releases/download/v6.4.30/praat6430_linux-intel64-barren.tar.gz
!tar xzf /tmp/praat.tar.gz -C /tmp && mv -f /tmp/praat_barren /usr/local/bin/praat && chmod +x /usr/local/bin/praat
# The "tapa[drvot]" form tells pip to also install Dr.VOT's extra deps
# (boltons, pydub, textgrid). The "@ git+..." form tells pip to install from
# this GitHub repo rather than from PyPI.
!pip install -q "tapa[drvot] @ git+https://github.com/sarmadchandio/tapa.git"
# Confirm the tools are all present before running anything long:
!python -m tapa.environment --vot-backend drvot --drvot-repo /content/Dr.VOT
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

> **If the YouTube download fails with a "confirm you're not a bot" error:**
> export `cookies.txt` from a logged-in browser on your own computer (e.g.
> with the *Get cookies.txt LOCALLY* extension), upload it to Colab via the
> left-sidebar folder icon, and re-run the cell — TAPA picks up
> `/content/cookies.txt` automatically. Full walkthrough in
> [Common issues](#common-issues).

> **Note — YouTube cookies are used by default.** YouTube increasingly
> blocks anonymous downloads, so TAPA looks for cookies automatically: first
> a `cookies.txt` file (`$TAPA_YT_COOKIES`, the working directory,
> `/content`, or `~`), then, on a machine that has a browser installed, that
> browser's own YouTube cookies. **If you are logged in, these are your
> account's cookies, and YouTube sees the download as coming from your
> account.** Cookies are read locally and sent only to YouTube as part of
> the download request — TAPA never stores, copies, or transmits them
> anywhere else. To turn this off, pass
> `TAPAConfig(youtube_cookies_from_browser="none")` (CLI:
> `--yt-cookies-from-browser none`); to use a specific file instead, pass
> `youtube_cookies_file="/path/to/cookies.txt"` (CLI: `--yt-cookies`).
> Analysing a local audio file never touches YouTube or cookies at all.

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

### Cell 6 — download all results as a zip

Colab's `/content/` directory disappears when the runtime disconnects, so
zip the results and pull them to your local machine before logging off.
This packages every CSV, JSON, and TextGrid the pipeline wrote and
triggers a browser download:

```python
import shutil
from google.colab import files

# Bundle everything in the results dir into a single archive.
zip_path = shutil.make_archive("/content/tapa_results", "zip", "/content/results")
print(f"Wrote {zip_path}")

# Trigger a browser download (the file lands in your local Downloads folder).
files.download(zip_path)
```

If the pipeline wrote to a custom directory, point `make_archive` at that
path instead. If you want a timestamped filename so you can keep multiple
runs straight:

```python
from datetime import datetime
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_path = shutil.make_archive(f"/content/tapa_results_{stamp}", "zip", "/content/results")
files.download(zip_path)
```

For very long runs you can also save directly to your Google Drive (the
zip survives runtime disconnects without manual download): mount Drive
with `from google.colab import drive; drive.mount('/content/drive')` and
write `make_archive("/content/drive/MyDrive/tapa_results", "zip", "/content/results")`.

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
clean recording, try widening the padding *after* the vowel
(`drvot_clip_post_ms=200`).

Do **not** raise `drvot_clip_pre_ms` to chase coverage. Dr.VOT anchors its
analysis window on the first voicing in the clip, so padding back into the
preceding vowel points that window at the wrong event — coverage stays high
while the numbers quietly stop meaning anything. See the note on that
setting in `tapa/config.py`.

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

## Setup: local machine

*Using Colab instead? See [Set up on Google Colab](#setup-google-colab).*

Requires **Python 3.10+** and **ffmpeg**. CUDA is used automatically when
available (Whisper and Resemblyzer); Dr.VOT itself runs on CPU.

### 1. System packages

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### 2. TAPA

```bash
# Core install
pip install git+https://github.com/sarmadchandio/tapa.git

# ...or with the Dr.VOT extras, if you want the neural VOT backend
pip install "tapa[drvot] @ git+https://github.com/sarmadchandio/tapa.git"
```

yt-dlp is the component that keeps up with YouTube's changes, and `pip
install` always pulls the current version.

### 3. Montreal Forced Aligner — recommended

Without MFA, phoneme boundaries fall back to CMUdict proportional timing,
which is too approximate to publish from.

```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

### 4. Dr.VOT — needed only for stop VOT

Dr.VOT calls Praat and sox while extracting features; without either, every
token silently falls back to the Praat estimator.

```bash
# Ubuntu/Debian
sudo apt-get install praat sox
# macOS
brew install praat sox

python -m tapa.drvot setup ~/Dr.VOT
```

On a headless machine — a server or an HPC node — install Praat's
[barren build](https://github.com/praat/praat/releases) instead of the
packaged one: it needs no GUI libraries, which is the usual cause of Dr.VOT
failing to start.

### 5. Check the setup

```bash
python -m tapa.environment --vot-backend drvot --drvot-repo ~/Dr.VOT
```

This reports every external tool the configuration needs, with a fix for
anything missing, and confirms Praat can run a script without a display. Do
this before a long run: a missing tool does not stop the pipeline, it quietly
degrades the results.

### 6. Run it

```python
from tapa import TAPAPipeline, TAPAConfig

pipeline = TAPAPipeline(TAPAConfig(
    vot_backend="drvot",              # omit for the Praat backend
    drvot_repo_dir="~/Dr.VOT",
    strict=True,                      # refuse to degrade silently
))
results = pipeline.run("interview.mp3")
```

Or from the shell — see [Command line](#command-line) for all flags:

```bash
tapa interview.mp3 -o results/ --vot-backend drvot --drvot-repo ~/Dr.VOT
```

---

## Backend choices: TAPA vs Dr.VOT

| | TAPA (default) | Dr.VOT |
|---|---|---|
| Method | Praat: intensity peak → first f0 cycle | Neural-network model trained on labeled VOTs (Shrem et al. 2019) |
| Speed | ~ms per token | ~1 s per token (CPU) |
| Negative VOT (prevoicing) | not handled | handled (`POS` aspirated / `NEG` prevoiced output) |
| Robustness on noisy or coarticulated speech | brittle | substantially better |
| Languages | language-agnostic in principle | English-trained; degrades on others |
| Extra setup | none | `python -m tapa.drvot setup <dir>`, plus Praat and sox |
| Measured on a 2-minute conversational sample | voiceless 0.5 ms, voiced 0.5 ms, ~70 % of tokens at the floor | voiceless 22 ms, voiced 16 ms, no tokens at the floor |

The last row is the one that matters for study design. English voiceless stops
should measure tens of milliseconds longer than voiced ones; the Praat backend
did not separate them at all on conversational audio, so **treat its VOT output
as indicative only**. Vowel formants and fricative moments are unaffected by
this choice — both backends share the same code for those.

Dr.VOT also depends on the phoneme boundaries it is handed. Run it with MFA:
on the same sample it gave voiceless 22 ms against voiced 16 ms with forced
alignment, but voiced *longer* than voiceless (39 ms against 25 ms) when fed
CMUdict's approximate boundaries, which is not a possible result.

When you choose `vot_backend="drvot"`, every stop token in the output JSON
gets two extra fields: `vot_method` (`"drvot"` or `"tapa-fallback"`) and
`vot_class_drvot` (`"POS"` for aspirated, `"NEG"` for prevoiced, `null` for
fallback rows). The aggregated `*_stop_averages.csv` is unchanged so existing
analysis code keeps working.

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
| `<stem>_aligned.TextGrid` | MFA phoneme alignment, when aligning the recording as one utterance |
| `<stem>_aligned_textgrids/` | One TextGrid per speaker turn, when `mfa_split_utterances=True` (the default) |
| `<stem>_run_summary.json` | What the run actually did — see below |
| `<stem>_vowel_formants.json` | Raw vowel F1/F2/pitch per token |
| `<stem>_vowel_averages.csv` | Per-speaker per-vowel average formants |
| `<stem>_stop_vot.json` | Raw stop VOT measurements per token |
| `<stem>_stop_averages.csv` | Per-speaker per-stop average VOT |
| `<stem>_fricative_spectra.json` | Raw fricative spectral moments per token |
| `<stem>_fricative_averages.csv` | Per-speaker per-fricative averages |

### `<stem>_run_summary.json` — check this before trusting a run

The pipeline degrades rather than stopping: if MFA fails it falls back to
CMUdict proportional timing, and if Dr.VOT fails it falls back to the Praat
estimator, token by token. Both produce a complete set of result files, so a
degraded run looks exactly like a good one. The run summary is how you tell
them apart. It records what you *requested* against what actually *happened*
— alignment method, VOT method with per-method token counts, speaker count,
out-of-vocabulary share — plus a `degradations` list, which is also printed at
the end of the run. An empty list means every component you asked for ran.

Set `TAPAConfig(strict=True)` to turn any degradation into an error instead,
which is what you want for numbers you intend to publish. To check the
external tools a configuration needs before starting, run
`python -m tapa.environment --vot-backend drvot --drvot-repo /path/to/Dr.VOT`.

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
| `num_speakers` | `None` | Number of speakers (`None` = estimate it). Supply it when you know the count — estimation is good but never better than knowing |
| `max_speakers` | `8` | Upper bound when estimating |
| `min_speaker_silhouette` | `0.15` | Below this no split is convincing and the recording is reported as one speaker |
| `min_segments_per_speaker` | `4` | Caps the estimate on short recordings, so a handful of segments cannot become a handful of speakers |
| `min_speaker_share` | `0.0` | `0` = off. e.g. `0.02` absorbs any speaker holding under 2 % of the speech into the nearest one |
| `strict` | `False` | Raise instead of degrading quietly (MFA falling back to CMUdict, Dr.VOT falling back to Praat). Turn on for results you intend to publish |
| `mfa_split_utterances` | `True` | Align per diarization segment rather than feeding MFA the whole recording as one utterance; keeps alignment memory flat with duration |
| `mfa_num_jobs` | `2` | MFA worker processes; 2 suits Colab's 2 vCPUs |
| `mfa_timeout_s` | `1800` | Kill MFA and fall back if an alignment overruns |
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
| `drvot_clip_pre_ms` | `25.0` | Padding before stop closure when cutting clips for Dr.VOT. Keep small — raising it breaks the measurement |
| `drvot_clip_post_ms` | `150.0` | Padding after the following vowel |
| `drvot_keep_temp` | `False` | Keep the per-recording Dr.VOT temp dir for inspection |
| `drvot_signed_vot` | `True` | Store prevoiced tokens as negative VOT. `False` reproduces pre-fix output |

Supported audio formats: `.mp3`, `.wav`, `.flac`. URLs: any standard YouTube
URL form (`youtube.com/watch?v=…`, `youtu.be/…`, `youtube.com/shorts/…`).

---

## Common issues

Items marked **(Colab)** apply only to the notebook environment;
**(local)** only to your own machine. Everything else applies to both.


**"`No video formats found`" / yt-dlp errors when passing a URL.**
Your runtime has a stale `yt-dlp`. Restart the Colab runtime (Runtime →
Disconnect and delete runtime), then re-run the install cell — it pulls
the latest version. On a non-Colab machine: `pip install -U yt-dlp` and
retry.

**"`Sign in to confirm you're not a bot`" when downloading. (both)** TAPA tries two
downloaders automatically: first `yt-dlp` with alternate player clients
(`mweb`, `tv_simply`, `android_vr`, `web_safari`), and if that gets bot-
checked, `pytubefix` with a different code path. On a local machine with a
browser installed, TAPA also reads that browser's YouTube cookies
automatically. One of those clears the challenge for most videos with no
manual steps required.

If everything falls over (it does happen on heavily-flagged Colab IPs),
the deterministic fix is real cookies from a logged-in browser:

1. On your own computer, install the
   [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
   extension (Chrome; also on Firefox add-ons).
2. Open <https://youtube.com> while logged in and click the extension to
   export `cookies.txt`.
3. Upload it to Colab: folder icon in the left sidebar → upload. Keep the
   name `cookies.txt` — it lands in `/content/`.
4. Re-run the pipeline cell, unchanged. TAPA auto-discovers `cookies.txt`
   in `/content/`, the working directory, `~`, or a path set in the
   `TAPA_YT_COOKIES` environment variable — no config needed. You'll see
   `[TAPA] Using auto-discovered YouTube cookies file: ...` in the log.

To point at a non-standard location explicitly:
`TAPAConfig(youtube_cookies_file="/path/to/cookies.txt")`, or on the CLI
`tapa "https://..." --yt-cookies /path/to/cookies.txt`.
Cookies are only sent to YouTube for the fetch — your audio file isn't
re-uploaded anywhere.

**Whisper / diarization runs on CPU instead of GPU. (Colab)** The `[TAPA] Device:`
line will say `CPU` — meaning you forgot to switch the runtime. Click
Runtime → Change runtime type → T4 GPU → Save, then run the cells again.

**MFA install seems hung. (Colab)** It legitimately takes 2–4 minutes the
first time (Miniforge download + conda solve + acoustic + dictionary
models). It's quick on subsequent runs in the same session.

**Speaker labels seem off.** Diarization is automatic — `SPEAKER_00`,
`SPEAKER_01` are assigned in order of first appearance, not in the order
you'd expect. If you know the number of speakers, set
`TAPAConfig(num_speakers=2)` for cleaner clustering. Then rename them by
post-processing the CSVs.

**Dr.VOT coverage is low (<70%).** Two common causes: (a) the clip ends too
soon after the vowel — pass `drvot_clip_post_ms=200` in `TAPAConfig`; (b) the
recording is heavily reverberant or noisy and Dr.VOT genuinely can't anchor —
the TAPA-Praat fallback will still measure those tokens. Leave
`drvot_clip_pre_ms` alone: raising it inflates coverage while making the
measurements worse (see `tapa/config.py`).

**Non-English audio.** Use `TAPAConfig(whisper_model="medium")` (not
`medium.en`). Stick to `vot_backend="tapa"` since Dr.VOT was English-trained.

**"Out of memory" on long recordings. (Colab)** Free-tier Colab has ~12 GB RAM. For
recordings longer than ~1 hour use `TAPAConfig(whisper_model="tiny.en")`,
or split the audio first with ffmpeg.

**Session disconnected mid-run. (Colab)** Free-tier sessions time out after
~90 minutes idle. For long batches use a Colab Pro runtime, or save your
intermediate results to Google Drive (`drive.mount("/content/drive")` then
set `results_dir="/content/drive/MyDrive/tapa_results/"`).

---

## Usage policy

### The downloader uses your YouTube cookies by default

YouTube increasingly refuses anonymous downloads, particularly from cloud IP
ranges such as Colab's. So when TAPA is given a YouTube URL it looks for
cookies automatically, in this order:

1. a cookies file — the path in `$TAPA_YT_COOKIES` if set, then
   `cookies.txt` or `youtube_cookies.txt` in the working directory, then the
   same two names in `/content` (Colab's upload directory), then
   `~/cookies.txt`;
2. failing that, on a machine with a browser installed, **the YouTube cookies
   belonging to that browser profile**, taken straight from its cookie store.

**If you are signed in to YouTube, these are your account's cookies, and
YouTube sees the download as coming from your account.** Your account is
therefore subject to whatever rate limiting, logging, or enforcement YouTube
applies, exactly as if you had loaded the video in your browser. Treat this
the same way you would treat any tool acting on your behalf with your session.

What TAPA does with them: cookies are read locally and attached to the
download request to YouTube. They are never written to the results directory,
never uploaded anywhere else, and never logged. TAPA prints only the path of
the cookie file it chose, or the name of the browser it read from.

To turn the behaviour off, or pin it to a file you control:

```python
TAPAConfig(youtube_cookies_from_browser="none")          # no cookies at all
TAPAConfig(youtube_cookies_file="/path/to/cookies.txt")  # only this file
```
```bash
tapa "https://..." --yt-cookies-from-browser none
tapa "https://..." --yt-cookies /path/to/cookies.txt
```

Nothing in this section applies when you analyse a local audio file: the
download code never runs, and no network request is made.

### Downloading other people's videos

Downloading from YouTube may conflict with its Terms of Service, and the
recordings themselves are usually copyrighted. Whether your use is permitted —
research exemptions, fair use or fair dealing, an institutional licence, or the
uploader's own permission — depends on your jurisdiction and your institution.
That judgement is yours to make. TAPA provides the mechanism, not the licence.

### Recordings of people

Speech recordings are personal data, and analysing them can reveal
characteristics their speakers never consented to share. If you are working
with interviews, clinical recordings, or anything involving identifiable
participants, follow your institution's human-subjects process.

For that setting it matters that **all analysis is local**: every model — VAD,
speaker embeddings, Whisper, MFA, Praat, Dr.VOT — runs on your machine, and no
audio, transcript, or measurement is transmitted anywhere. Network access is
used only to install the software and download model weights on first use, and
for the optional YouTube downloader. Once the model caches are populated you
can disconnect from the network entirely and the pipeline still runs, which is
a straightforward way to demonstrate the guarantee to a review board.

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
