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

    # Align per diarization segment (MFA corpus mode) instead of feeding the
    # whole recording as one utterance. Single-utterance alignment makes MFA's
    # lattice scale with the full transcript — GBs of RAM / possible hangs on
    # long recordings — so keep this on unless debugging alignment itself.
    mfa_split_utterances: bool = True
    mfa_utterance_pad: float = 0.25  # seconds of context around each utterance
    # MFA worker processes. MFA's default (3) costs ~2-3 GB RAM apiece in
    # corpus mode; on Colab (2 vCPUs, ~12.7 GB RAM) extra workers add memory
    # pressure without speed. Raise on real multi-core machines.
    mfa_num_jobs: int = 2

    # Diarization
    num_speakers: Optional[int] = None
    # When num_speakers is None the count is estimated by picking the
    # silhouette-best clustering between 2 and max_speakers. If no split beats
    # min_speaker_silhouette the recording is treated as a single speaker.
    max_speakers: int = 8
    min_speaker_silhouette: float = 0.15
    min_segments_per_speaker: int = 4   # caps the estimate on short recordings
    # When num_speakers is None the speaker count is estimated, which can split
    # one talker into several clusters. Set e.g. 0.02 to absorb any cluster
    # holding under 2 % of the speech into the nearest speaker. Prefer setting
    # num_speakers when the true count is known — that is always more reliable.
    min_speaker_share: float = 0.0
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

    # When run() gets a YouTube URL, the downloaded mp3 is saved under audio_dir
    # at this bitrate (kbps, passed to yt-dlp's preferredquality).
    mp3_bitrate: str = "192"

    # Cookies for yt-dlp, used when YouTube serves "Sign in to confirm you're
    # not a bot". cookies_from_browser defaults to "auto": read cookies from the
    # first installed browser found (firefox, chrome, ...), or run cookieless
    # when none is installed (e.g. Colab). Set to a browser name ("chrome",
    # "firefox", ...) to force one, or None/"none" to disable. A cookies_file,
    # if given, takes precedence over browser auto-detection. When cookies_file
    # is None, well-known locations are probed automatically ($TAPA_YT_COOKIES,
    # then cookies.txt / youtube_cookies.txt in the CWD, /content — Colab's
    # upload dir — and ~), so on Colab uploading cookies.txt is all it takes.
    youtube_cookies_file: Optional[str] = None          # path to Netscape cookies.txt
    youtube_cookies_from_browser: Optional[str] = "auto"

    # Raise instead of quietly degrading. Off by default so a run always
    # produces something, but every fallback has silently cost real analyses:
    # MFA failing to CMUdict timing, or Dr.VOT failing to Praat for every
    # token, while the run still reported success. Turn this on for anything
    # whose numbers you intend to publish.
    strict: bool = False

    # VOT backend: "tapa" (Praat-based) or "drvot" (Dr.VOT CNN, with TAPA fallback)
    vot_backend: str = "tapa"
    drvot_repo_dir: Optional[str] = None
    drvot_python: Optional[str] = None  # None -> sys.executable
    drvot_clip_pre_ms: float = 150.0    # padding before stop closure when cutting clips
    drvot_clip_post_ms: float = 150.0   # padding after the following vowel
    drvot_keep_temp: bool = False       # keep clip temp dir for debugging
