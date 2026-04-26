"""TAPA - Text and Phonetic Analysis pipeline for speaker diarization and acoustic analysis."""

from .config import TAPAConfig
from .download import download_youtube_audio, is_youtube_url
from .drvot import extract_all_stop_measurements_drvot, setup_drvot
from .pipeline import TAPAPipeline
from .shortcuts import (
    Models,
    align,
    compute_averages,
    diarize,
    extract_consonants,
    extract_formants,
    load_models,
    transcribe,
)

__version__ = "0.1.0"
__all__ = [
    "TAPAPipeline",
    "TAPAConfig",
    "Models",
    "load_models",
    "diarize",
    "transcribe",
    "align",
    "extract_formants",
    "extract_consonants",
    "compute_averages",
    "download_youtube_audio",
    "is_youtube_url",
    "setup_drvot",
    "extract_all_stop_measurements_drvot",
]
