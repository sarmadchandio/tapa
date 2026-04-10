"""Default configuration for the TAPA pipeline."""

from dataclasses import dataclass
from typing import Optional, Set


@dataclass
class TAPAConfig:
    # Directories
    audio_dir: str = "audio/"
    results_dir: str = "results/"
    mfa_temp_dir: str = "mfa_temp/"

    # Audio
    sample_rate: int = 16000

    # MFA binary path (auto-detected if None)
    mfa_bin: Optional[str] = None

    # Diarization
    num_speakers: Optional[int] = None
    min_segment_duration: float = 0.1
    merge_gap: float = 0.5

    # Vowel formant extraction
    min_vowel_duration: float = 0.03
    vowel_trim_fraction: float = 0.15
    f1_min: float = 150
    f1_max: float = 1500
    f2_min: float = 400
    f2_max: float = 4000

    # Consonant extraction
    min_stop_duration: float = 0.015
    min_fricative_duration: float = 0.03
    vot_max: float = 0.150
    fricative_freq_range: tuple = (1000, 11025)

    # Outlier rejection
    mad_threshold: float = 2.0

    # Target vowels (None = all)
    target_vowels: Optional[Set[str]] = None

    # Whisper model
    whisper_model: str = "small.en"
