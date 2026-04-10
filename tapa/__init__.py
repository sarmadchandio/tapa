"""TAPA - Text and Phonetic Analysis pipeline for speaker diarization and acoustic analysis."""

from .config import TAPAConfig
from .pipeline import TAPAPipeline

__version__ = "0.1.0"
__all__ = ["TAPAPipeline", "TAPAConfig"]
