"""TAPA - Text and Phonetic Analysis pipeline for speaker diarization and acoustic analysis.

Names are resolved on first use rather than at import. Importing the package
therefore costs nothing and, more importantly, does not require torch, whisper
or soundfile to be installed — so `python -m tapa.environment` can diagnose an
environment that is too broken to run the pipeline.
"""
import importlib

__version__ = "0.1.0"

#: public name -> module it lives in
_EXPORTS = {
    "TAPAConfig": ".config",
    "TAPAPipeline": ".pipeline",
    "download_youtube_audio": ".download",
    "is_youtube_url": ".download",
    "setup_drvot": ".drvot",
    "extract_all_stop_measurements_drvot": ".drvot",
    "Models": ".shortcuts",
    "load_models": ".shortcuts",
    "diarize": ".shortcuts",
    "transcribe": ".shortcuts",
    "align": ".shortcuts",
    "extract_formants": ".shortcuts",
    "extract_consonants": ".shortcuts",
    "compute_averages": ".shortcuts",
    "check_environment": ".environment",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module, __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
