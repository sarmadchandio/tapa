# TAPA - Text and Phonetic Analysis

A Python pipeline for **speaker diarization** and **phonetic analysis** of audio recordings. Combines Whisper (transcription), Montreal Forced Aligner (phoneme alignment), and Praat (acoustic measurements) to extract detailed per-speaker phonetic data.

## What it does

Given an audio file, TAPA will:

1. **Diarize** speakers using Silero VAD + Resemblyzer embeddings + hierarchical clustering
2. **Transcribe** speech with word-level timestamps using OpenAI Whisper
3. **Force-align** phonemes using Montreal Forced Aligner (or CMUdict fallback)
4. **Measure vowel formants** (F1, F2, pitch) using Praat
5. **Measure stop consonant VOT** (Voice Onset Time) using Praat
6. **Measure fricative spectral moments** (Center of Gravity, spectral SD, skewness, kurtosis)
7. **Compute per-speaker averages** with MAD outlier rejection

## Installation

### From GitHub

```bash
pip install git+https://github.com/yourusername/tapa.git
```

### From source

```bash
git clone https://github.com/yourusername/tapa.git
cd tapa
pip install -e .
```

### System dependencies

You need `ffmpeg` installed for audio decoding:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Optional: Montreal Forced Aligner

MFA provides precise phoneme boundaries. Without it, TAPA falls back to CMUdict proportional timing (less accurate but still functional).

```bash
# Install via conda
conda install -c conda-forge montreal-forced-aligner

# Download English models
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

## Quick Start

### Python API

```python
from tapa import TAPAPipeline

# Initialize and run on a single file
pipeline = TAPAPipeline()
results = pipeline.run("interview.mp3")

# Access results directly
for speaker, vowels in results["vowel_averages"].items():
    print(f"\n{speaker}:")
    for vowel, data in vowels.items():
        print(f"  /{vowel}/  F1={data['mean_f1']:.0f}  F2={data['mean_f2']:.0f}  (n={data['n_after_filtering']})")
```

### Command line

```bash
# Process a single file
tapa interview.mp3

# Process multiple files with custom output directory
tapa file1.mp3 file2.wav -o my_results/

# Specify number of speakers and whisper model
tapa podcast.mp3 --num-speakers 3 --whisper-model medium.en
```

### With custom configuration

```python
from tapa import TAPAPipeline, TAPAConfig

cfg = TAPAConfig(
    num_speakers=2,              # Force 2-speaker detection
    whisper_model="medium.en",   # Use larger Whisper model
    target_vowels={"i", "u", "a"},  # Only analyze these vowels (IPA)
    results_dir="my_output/",
    mad_threshold=2.5,           # Looser outlier rejection
)

pipeline = TAPAPipeline(config=cfg)
results = pipeline.run("dialogue.wav")
```

### Batch processing

```python
from tapa import TAPAPipeline

pipeline = TAPAPipeline()

# Process all audio files in a directory
results = pipeline.run_batch(audio_dir="recordings/", results_dir="output/")

# results is a dict: {filename: result_dict}
for filename, result in results.items():
    n_speakers = len(set(s["speaker"] for s in result["diarization"]))
    print(f"{filename}: {n_speakers} speakers")
```

## Output Files

For each audio file (e.g., `interview.mp3`), TAPA produces:

| File | Description |
|------|-------------|
| `interview_diarization.csv` | Speaker segments (speaker, start, end) |
| `interview_transcription.csv` | Word-level transcript with speaker labels |
| `interview_transcription.txt` | Human-readable transcript |
| `interview_aligned.TextGrid` | MFA phoneme alignment (Praat format) |
| `interview_vowel_formants.json` | Raw vowel F1/F2/pitch per token |
| `interview_vowel_averages.csv` | Per-speaker per-vowel average formants |
| `interview_stop_vot.json` | Raw stop VOT measurements per token |
| `interview_stop_averages.csv` | Per-speaker per-stop average VOT |
| `interview_fricative_spectra.json` | Raw fricative spectral moments per token |
| `interview_fricative_averages.csv` | Per-speaker per-fricative averages |

## Working with results

### Load CSV results with pandas

```python
import pandas as pd

vowels = pd.read_csv("results/interview_vowel_averages.csv")
stops = pd.read_csv("results/interview_stop_averages.csv")
fricatives = pd.read_csv("results/interview_fricative_averages.csv")

# Filter to a specific speaker
speaker_0 = vowels[vowels["speaker"] == "SPEAKER_00"]
print(speaker_0[["vowel", "mean_f1", "mean_f2", "n_after_filtering"]])
```

### Use results dict directly

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

## Configuration Reference

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

## GPU Support

TAPA automatically uses CUDA if available. This significantly speeds up Whisper transcription and speaker embedding extraction.

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## License

MIT
