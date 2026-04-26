# TAPA - Text and Phonetic Analysis

A Python package for **speaker diarization** and **phonetic analysis** of audio recordings. Given an audio file, TAPA identifies who is speaking, transcribes what they say, and extracts detailed acoustic measurements per speaker.

## What it does

1. **Speaker diarization** — identifies speakers and their time boundaries
2. **Transcription** — word-level transcript with timestamps
3. **Phoneme alignment** — precise phoneme boundaries (MFA or CMUdict fallback)
4. **Vowel formants** — F1, F2, and pitch for each vowel token
5. **Stop consonant VOT** — Voice Onset Time measurements
6. **Fricative spectral moments** — Center of Gravity, spectral SD, skewness, kurtosis
7. **Per-speaker averages** — summary statistics with outlier rejection

## Installation

```bash
pip install git+https://github.com/sarmadchandio/tapa.git
```

Requires **Python 3.9+** and **ffmpeg**:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

TAPA automatically uses **CUDA** if available (speeds up Whisper and speaker embeddings significantly). CPU works but is slower.

### Optional: Montreal Forced Aligner

MFA gives precise phoneme boundaries. Without it, TAPA falls back to CMUdict proportional timing (less accurate but functional).

```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

### Google Colab setup

Colab has torch and ffmpeg pre-installed, so just install TAPA:

```python
!pip install -q git+https://github.com/sarmadchandio/tapa.git
```

To add MFA support (Colab doesn't have conda, so use Miniforge):

```python
import os, subprocess

# Install Miniforge + MFA (takes 2-4 minutes on first run)
if not os.path.exists("/opt/miniforge"):
    !wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    !bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge > /dev/null 2>&1

!/opt/miniforge/bin/mamba install -c conda-forge montreal-forced-aligner -y -q 2>&1 | tail -3
!/opt/miniforge/bin/mfa model download acoustic english_us_arpa
!/opt/miniforge/bin/mfa model download dictionary english_us_arpa

# Add to PATH so TAPA can find it
os.environ["PATH"] = "/opt/miniforge/bin:" + os.environ["PATH"]
```

Then use TAPA normally:

```python
from tapa import TAPAPipeline, TAPAConfig

# Point to the MFA binary (or skip this if you added it to PATH above)
cfg = TAPAConfig(mfa_bin="/opt/miniforge/bin/mfa")
pipeline = TAPAPipeline(config=cfg)
results = pipeline.run("your_audio.mp3")
```

**Colab notes:**
- Colab already has torch with CUDA and ffmpeg — no extra setup needed for those
- Use a **GPU runtime** (Runtime > Change runtime type > GPU) for best performance
- Free tier has ~12GB RAM — large audio files (>1 hour) may need `whisper_model="tiny.en"` to fit in memory
- MFA install persists within a session but resets when the runtime disconnects

### Optional: Dr.VOT for stop consonants

By default TAPA measures Voice Onset Time with a Praat-based signal heuristic
(burst = intensity peak; voicing onset = first f0 cycle). You can swap this for
the deep-learning **Dr.VOT** model (Shrem et al., Interspeech 2019) which
handles negative VOT (prevoicing) and is more robust on noisy or coarticulated
speech. The unified pipeline keeps everything else (diarization, transcription,
alignment, vowel formants, fricative spectral moments) on the TAPA path and
only swaps in Dr.VOT for the stop step. Tokens Dr.VOT cannot predict
automatically fall back to TAPA-Praat so coverage never drops.

**One-time setup (Colab):**

```python
# 1. System deps Dr.VOT needs at runtime (apt — can't be folded into pip)
!apt-get install -y -qq praat sox

# 2. Install TAPA *with* the [drvot] extra in one line. Includes boltons,
#    pydub, textgrid, noisereduce — the slim Python deps Dr.VOT needs at
#    runtime. Do NOT install Dr.VOT's full requirements.txt — it pins older
#    torch/numpy that conflict with Colab's preinstalled stack.
!pip install -q "tapa[drvot] @ git+https://github.com/sarmadchandio/tapa.git"

# 3. Clone + patch Dr.VOT (idempotent — safe to re-run)
!python -m tapa.drvot setup /content/Dr.VOT
```

**Run with Dr.VOT as the stop backend:**

```python
from tapa import TAPAPipeline, TAPAConfig

cfg = TAPAConfig(
    mfa_bin="/opt/miniforge/bin/mfa",   # if you set up MFA above
    vot_backend="drvot",
    drvot_repo_dir="/content/Dr.VOT",
)
pipeline = TAPAPipeline(config=cfg)
results = pipeline.run("your_audio.mp3")
```

Equivalent CLI:

```bash
tapa your_audio.mp3 --vot-backend drvot --drvot-repo /content/Dr.VOT
```

Each stop token in `*_stop_vot.json` then carries two extra fields:
`vot_method` (`"drvot"` or `"tapa-fallback"`) and `vot_class_drvot`
(`"POS"` for aspirated / `"NEG"` for prevoiced / `null`). The aggregated CSV
(`*_stop_averages.csv`) is unchanged.

**Dr.VOT notes:**
- The model is small (~1.5 MB) and runs on CPU at ~1 s/token; GPU offers no benefit
- English-trained — accuracy degrades on other languages
- Per-recording overhead scales linearly with the number of stops (~5 min for a 30-min clip with ~300 stops)

## Quick Start

```python
from tapa import TAPAPipeline

pipeline = TAPAPipeline()
results = pipeline.run("interview.mp3")
```

That's it. Results are saved to `results/` and returned as a dict.

## Usage

### Run the full pipeline

```python
from tapa import TAPAPipeline

pipeline = TAPAPipeline()
results = pipeline.run("interview.mp3")

# Print per-speaker vowel formants
for speaker, vowels in results["vowel_averages"].items():
    print(f"\n{speaker}:")
    for vowel, data in vowels.items():
        print(f"  /{vowel}/  F1={data['mean_f1']:.0f}  F2={data['mean_f2']:.0f}  (n={data['n_after_filtering']})")
```

### Use individual components

Each step can be used independently:

```python
from tapa import diarize, transcribe, align, extract_formants, extract_consonants, compute_averages

# Just speaker diarization
segments = diarize("interview.mp3")

# Just transcription
words = transcribe("interview.mp3")

# Just phoneme alignment
phones = align("interview.mp3", words)

# Vowel formants (pass segments for per-speaker results, or omit for single-speaker)
formants = extract_formants("interview.mp3", segments)

# Stop VOT + fricative spectral moments
stop_data, fricative_data = extract_consonants("interview.mp3", segments)

# Per-speaker averages with outlier rejection (pass any combination)
avgs = compute_averages(vowel_data=formants, stop_data=stop_data, fricative_data=fricative_data)
```

### Processing multiple files

Use `load_models()` to load models once and reuse them across files. Without it, each function call reloads models from scratch.

```python
from tapa import load_models, diarize, transcribe, extract_formants

models = load_models()
# Output: TAPA: Using device: NVIDIA A100-SXM4-40GB (CUDA)
#         Loading models...
#         Models loaded.

for f in ["interview1.mp3", "interview2.mp3", "podcast.wav"]:
    segments = diarize(f, models=models)
    words = transcribe(f, models=models)
    formants = extract_formants(f, segments, models=models)
```

Or use the pipeline class, which caches models automatically:

```python
from tapa import TAPAPipeline

pipeline = TAPAPipeline()

# Process all audio files in a directory
results = pipeline.run_batch(audio_dir="recordings/", results_dir="output/")

# Or specific files (models loaded once on first call)
for f in ["file1.mp3", "file2.wav"]:
    results = pipeline.run(f)
```

### Custom configuration

```python
from tapa import TAPAPipeline, TAPAConfig, load_models, diarize

cfg = TAPAConfig(
    num_speakers=2,                 # Force 2-speaker detection (default: auto-detect)
    whisper_model="medium.en",      # Larger Whisper model (default: small.en)
    target_vowels={"i", "u", "a"}, # Only analyze these vowels in IPA (default: all)
    mad_threshold=2.5,              # Looser outlier rejection (default: 2.0)
)

# Works with the pipeline
pipeline = TAPAPipeline(config=cfg)
results = pipeline.run("dialogue.wav")

# And with standalone functions
models = load_models(config=cfg)
segments = diarize("dialogue.wav", models=models)
```

### Command line

```bash
tapa interview.mp3
tapa file1.mp3 file2.wav -o my_results/
tapa podcast.mp3 --num-speakers 3 --whisper-model medium.en

# YouTube URLs are downloaded to mp3 first (saved under --audio-dir, default audio/)
tapa "https://youtu.be/DPO7imV0LHg" -o my_results/
tapa "https://www.youtube.com/watch?v=DPO7imV0LHg" --audio-dir downloads/ --mp3-bitrate 256
```

## Output Files

For each audio file (e.g., `interview.mp3`), TAPA saves to the results directory:

| File | Description |
|------|-------------|
| `interview_diarization.csv` | Speaker segments (speaker, start, end) |
| `interview_transcription.csv` | Word-level transcript with speaker labels |
| `interview_transcription.txt` | Human-readable transcript |
| `interview_aligned.TextGrid` | MFA phoneme alignment (only if MFA is installed) |
| `interview_vowel_formants.json` | Raw vowel F1/F2/pitch per token |
| `interview_vowel_averages.csv` | Per-speaker per-vowel average formants |
| `interview_stop_vot.json` | Raw stop VOT measurements per token |
| `interview_stop_averages.csv` | Per-speaker per-stop average VOT |
| `interview_fricative_spectra.json` | Raw fricative spectral moments per token |
| `interview_fricative_averages.csv` | Per-speaker per-fricative averages |

### Sample output

**Vowel averages CSV** (`interview_vowel_averages.csv`):

```
speaker,vowel,mean_f1,mean_f2,std_f1,std_f2,mean_pitch,n_tokens,n_after_filtering
SPEAKER_00,i,393.0,2110.0,54.0,227.0,142.3,88,74
SPEAKER_00,æ,656.0,1634.0,124.0,176.0,138.5,72,63
SPEAKER_01,i,400.0,2418.0,41.0,196.0,198.7,903,805
```

**Stop averages CSV** (`interview_stop_averages.csv`):

```
speaker,phone,voicing,place,mean_vot_ms,std_vot_ms,mean_closure_ms,n_tokens,n_after_filtering
SPEAKER_00,p,voiceless,bilabial,0.6,0.4,78.32,42,38
SPEAKER_00,t,voiceless,alveolar,4.2,12.0,62.15,104,104
```

### Working with results in pandas

```python
import pandas as pd

vowels = pd.read_csv("results/interview_vowel_averages.csv")
speaker_0 = vowels[vowels["speaker"] == "SPEAKER_00"]
print(speaker_0[["vowel", "mean_f1", "mean_f2", "n_after_filtering"]])
```

### Working with the results dict

```python
results = pipeline.run("interview.mp3")

# Diarization segments
for seg in results["diarization"][:5]:
    print(f"  {seg['speaker']}: {seg['start']:.1f}s - {seg['end']:.1f}s")

# Word-level transcript
for word in results["words"][:10]:
    print(f"  {word['word']} ({word['start']:.2f}s)")

# Vowel formant data (nested: speaker -> vowel -> list of measurements)
for speaker, vowels in results["vowel_data"].items():
    for vowel_ipa, measurements in vowels.items():
        print(f"  {speaker} /{vowel_ipa}/: {len(measurements)} tokens")
```

## API Reference

### Model loading

| Function | Description |
|----------|-------------|
| `load_models(config=None, whisper_model=None)` | Load all models once, returns `Models` cache. Prints device info. |

### Convenience functions

All importable from `tapa`. Every function accepts optional `config=` and `models=` parameters.

| Function | Description | Returns |
|----------|-------------|---------|
| `diarize(path, num_speakers=, config=, models=)` | Speaker diarization | `[{"speaker", "start", "end"}]` |
| `transcribe(path, model_name=, config=, models=)` | Whisper transcription | `[{"word", "start", "end"}]` |
| `align(path, words, config=, models=)` | Forced phoneme alignment | `[{"phone", "start", "end"}]` |
| `extract_formants(path, segments=, config=, models=)` | Vowel F1/F2/pitch | `{speaker: {vowel: [measurements]}}` |
| `extract_consonants(path, segments=, config=, models=)` | Stop VOT + fricative spectra | `(stop_data, fricative_data)` |
| `compute_averages(vowel_data=, stop_data=, fricative_data=, config=)` | Averages with outlier rejection | `{"vowel_averages", "stop_averages", ...}` |

### Pipeline class

| Method | Description |
|--------|-------------|
| `TAPAPipeline(config=None)` | Initialize with optional `TAPAConfig` |
| `pipeline.run(audio_path, results_dir=None)` | Process a single file, returns results dict |
| `pipeline.run_batch(audio_dir=None, results_dir=None)` | Process all audio files in a directory |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | `16000` | Audio sample rate for processing |
| `whisper_model` | `"small.en"` | Whisper model (`tiny.en`, `base.en`, `small.en`, `medium.en`, `large`) |
| `num_speakers` | `None` | Number of speakers (`None` = auto-detect) |
| `min_segment_duration` | `0.1` | Minimum speech segment duration (seconds) |
| `merge_gap` | `0.5` | Merge same-speaker segments closer than this (seconds) |
| `min_vowel_duration` | `0.03` | Minimum vowel duration to analyze (seconds) |
| `vowel_trim_fraction` | `0.15` | Fraction to trim from vowel edges before measurement |
| `f1_min` / `f1_max` | `150` / `1500` | Valid F1 range (Hz) |
| `f2_min` / `f2_max` | `400` / `4000` | Valid F2 range (Hz) |
| `min_stop_duration` | `0.015` | Minimum stop consonant duration (seconds) |
| `min_fricative_duration` | `0.03` | Minimum fricative duration (seconds) |
| `vot_max` | `0.150` | Maximum valid VOT (seconds) |
| `mad_threshold` | `2.0` | MAD outlier rejection threshold |
| `target_vowels` | `None` | Set of IPA vowels to analyze (`None` = all) |
| `mfa_bin` | `None` | Path to MFA binary (`None` = auto-detect) |

Supported audio formats: `.mp3`, `.wav`, `.flac`

## License

MIT
