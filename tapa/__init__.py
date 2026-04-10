"""TAPA - Text and Phonetic Analysis pipeline for speaker diarization and acoustic analysis."""

from .config import TAPAConfig
from .pipeline import TAPAPipeline
from .shortcuts import (
    align,
    compute_averages,
    diarize,
    extract_consonants,
    extract_formants,
    transcribe,
)

__version__ = "0.1.0"
__all__ = [
    "TAPAPipeline",
    "TAPAConfig",
    "diarize",
    "transcribe",
    "align",
    "extract_formants",
    "extract_consonants",
    "compute_averages",
]
