"""Test setup.

The pipeline's heavy dependencies (torch, whisper, librosa, parselmouth,
resemblyzer) are not needed to exercise its logic, and requiring them would
mean the suite only runs on a machine set up for inference. Any that are
missing get a stub so the pure-logic tests import cleanly; tests that genuinely
need a real dependency ask for it via the fixtures at the bottom and skip when
it is absent.
"""
import importlib
import shutil
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_OPTIONAL = ["torch", "torchaudio", "whisper", "librosa", "soundfile",
             "parselmouth", "resemblyzer", "nltk", "praatio", "tqdm"]


def _stub(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__stubbed__ = True

    def _missing(*_a, **_k):
        raise RuntimeError(f"stubbed dependency {name!r} was actually called")

    mod.__getattr__ = lambda attr: _missing
    return mod


for _name in _OPTIONAL:
    try:
        importlib.import_module(_name)
    except Exception:                                   # noqa: BLE001
        sys.modules.setdefault(_name, _stub(_name))
        if _name == "praatio":
            sys.modules.setdefault("praatio.textgrid", _stub("praatio.textgrid"))
        if _name == "tqdm":
            sys.modules["tqdm"].tqdm = lambda it, **k: it


def _have(name):
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__stubbed__", False):
        return False
    try:
        importlib.import_module(name)
        return True
    except Exception:                                   # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def ffmpeg():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    return shutil.which("ffmpeg")


@pytest.fixture(scope="session")
def praatio():
    if not _have("praatio"):
        pytest.skip("praatio not installed")
    return importlib.import_module("praatio")
