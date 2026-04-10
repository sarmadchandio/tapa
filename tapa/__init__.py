"""TAPA - Text and Phonetic Analysis pipeline for speaker diarization and acoustic analysis."""

from .config import TAPAConfig
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
]
