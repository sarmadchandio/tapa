"""YouTube → mp3 downloader for the TAPA pipeline.

Two-library fallback chain to maximize success on Colab / cloud IPs that
YouTube routinely bot-checks:

  1. yt-dlp (default) — alternate player_client list to dodge most bot checks,
     plus optional cookies file / browser cookies for stubborn videos.
  2. pytubefix (fallback) — different code path that sometimes succeeds when
     yt-dlp gets blocked, since it constructs stream URLs from the JS player
     directly rather than going through yt-dlp's challenge surface.

Final mp3 is written into `output_dir` named `<video_id>.mp3` so all
downstream pipeline outputs trace back to the source recording.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse


YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be",
}

# Default alternate player clients used to bypass YouTube's "Sign in to
# confirm you're not a bot" check that fires on datacenter IPs (Colab, AWS).
# yt-dlp tries these in order and uses the first that returns playable streams
# without triggering the web bot challenge.
DEFAULT_YT_PLAYER_CLIENTS = ["mweb", "tv_simply", "android_vr", "web_safari"]

# Browsers probed by cookies_from_browser="auto", in preference order.
# Firefox first: its cookie store needs no OS keyring to decrypt, so extraction
# works even in headless sessions where Chrome's keyring may be locked.
# Values are yt-dlp browser names -> executables that indicate it's installed.
AUTO_COOKIE_BROWSERS = [
    ("firefox", ["firefox"]),
    ("chrome", ["google-chrome", "google-chrome-stable", "chrome"]),
    ("chromium", ["chromium", "chromium-browser"]),
    ("brave", ["brave-browser", "brave"]),
    ("edge", ["microsoft-edge", "microsoft-edge-stable"]),
    ("vivaldi", ["vivaldi"]),
    ("opera", ["opera"]),
]


def _detect_cookie_browser() -> Optional[str]:
    """Return the yt-dlp name of the first installed browser, or None (e.g. Colab)."""
    for browser, executables in AUTO_COOKIE_BROWSERS:
        if any(shutil.which(exe) for exe in executables):
            return browser
    return None


# Env var and well-known locations probed for a Netscape cookies.txt when no
# cookies_file is passed explicitly. /content is Colab's upload directory, so
# on Colab "upload cookies.txt via the sidebar, re-run the cell" just works
# with zero configuration.
COOKIE_FILE_ENV_VAR = "TAPA_YT_COOKIES"
COOKIE_FILE_CANDIDATES = [
    "cookies.txt",
    "youtube_cookies.txt",
    "/content/cookies.txt",
    "/content/youtube_cookies.txt",
    "~/cookies.txt",
]


def _discover_cookies_file() -> Optional[str]:
    """Find a cookies.txt via $TAPA_YT_COOKIES or well-known paths, else None."""
    env = os.environ.get(COOKIE_FILE_ENV_VAR)
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return str(p.resolve())
        print(f"[TAPA] Warning: {COOKIE_FILE_ENV_VAR}={env} does not exist — ignoring.",
              flush=True)
    for cand in COOKIE_FILE_CANDIDATES:
        p = Path(cand).expanduser()
        if p.is_file():
            return str(p.resolve())
    return None


def is_youtube_url(s: str) -> bool:
    """Return True if `s` looks like a YouTube URL."""
    try:
        u = urlparse(s)
    except ValueError:
        return False
    return u.scheme in ("http", "https") and u.hostname in YOUTUBE_HOSTS


def youtube_video_id(url: str) -> str | None:
    """Best-effort extraction of the 11-char YouTube video ID from a URL."""
    u = urlparse(url)
    if u.hostname == "youtu.be":
        return u.path.lstrip("/").split("/")[0] or None
    if u.path == "/watch":
        vs = parse_qs(u.query).get("v")
        return vs[0] if vs else None
    parts = [p for p in u.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in ("shorts", "embed", "v"):
        return parts[1]
    return None


def _is_bot_check_error(exc: BaseException) -> bool:
    """Heuristic: did this yt-dlp failure come from YouTube's bot challenge?"""
    msg = str(exc).lower()
    return ("sign in to confirm you" in msg
            or "confirm you're not a bot" in msg
            or "not a bot" in msg)


def _download_with_ytdlp(url: str, out_dir: Path, bitrate: str,
                         cookies_file: Optional[str],
                         cookies_from_browser: Optional[str]) -> str:
    """Primary download path: yt-dlp with alternate player clients + optional cookies."""
    import yt_dlp

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": DEFAULT_YT_PLAYER_CLIENTS}},
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": bitrate,
        }],
    }
    if cookies_file:
        opts["cookiefile"] = str(cookies_file)
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info.get("id") or youtube_video_id(url)
    if not video_id:
        raise RuntimeError(f"yt-dlp could not determine video ID for {url}")
    mp3_path = out_dir / f"{video_id}.mp3"
    if not mp3_path.exists():
        raise RuntimeError(
            f"yt-dlp finished but {mp3_path} is missing — ffmpeg post-processing likely failed."
        )
    return str(mp3_path.resolve())


def _download_with_pytubefix(url: str, out_dir: Path, bitrate: str) -> str:
    """Fallback: pytubefix downloads audio stream, then ffmpeg transcodes to mp3.

    Different library, different signature/cipher path — sometimes succeeds when
    yt-dlp gets bot-challenged on Colab/cloud IPs.
    """
    try:
        from pytubefix import YouTube
    except ImportError as e:
        raise ImportError(
            "pytubefix is required for the YouTube download fallback. "
            "It should be installed automatically with TAPA — try `pip install pytubefix`."
        ) from e

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. pytubefix downloads native audio streams "
            "(usually .m4a/.webm) which need ffmpeg to transcode to mp3."
        )

    yt = YouTube(url)
    stream = (yt.streams
                .filter(only_audio=True)
                .order_by("abr")
                .desc()
                .first())
    if stream is None:
        raise RuntimeError("pytubefix found no audio-only stream for this video.")

    video_id = yt.video_id or youtube_video_id(url)
    if not video_id:
        raise RuntimeError(f"pytubefix could not determine video ID for {url}")

    # pytubefix picks the file extension from the stream itself.
    raw_path = Path(stream.download(output_path=str(out_dir),
                                    filename=f"{video_id}.{stream.subtype}"))
    mp3_path = out_dir / f"{video_id}.mp3"

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(raw_path),
           "-vn", "-codec:a", "libmp3lame", "-b:a", f"{bitrate}k",
           str(mp3_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if raw_path.exists():
        raw_path.unlink()
    if result.returncode != 0 or not mp3_path.exists():
        raise RuntimeError(f"ffmpeg transcode failed: {result.stderr[-500:]}")
    return str(mp3_path.resolve())


def download_youtube_audio(url: str, output_dir: str | os.PathLike,
                           bitrate: str = "192",
                           cookies_file: Optional[str] = None,
                           cookies_from_browser: Optional[str] = "auto") -> str:
    """Download `url` and produce an mp3 in `output_dir`.

    Tries yt-dlp first, automatically falls back to pytubefix if yt-dlp hits
    YouTube's bot challenge. If both fail, raises a RuntimeError explaining
    that real cookies are needed.

    Args:
        url: YouTube URL (any standard form).
        output_dir: Directory to write the mp3 into. Created if missing.
        bitrate: mp3 bitrate in kbps as a string.
        cookies_file: Path to Netscape cookies.txt for stubborn videos. When
            None, well-known locations are probed automatically ($TAPA_YT_COOKIES,
            then cookies.txt / youtube_cookies.txt in the current directory,
            /content — Colab's upload dir — and the home directory).
        cookies_from_browser: Browser name to read cookies from (local machines
            only). "auto" (the default) uses the first installed browser found,
            or no cookies when none is installed (e.g. Colab). None or "none"
            disables browser cookies entirely.

    Returns:
        Absolute path to the resulting `.mp3` file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    auto_cookies_file = False
    if not cookies_file:
        discovered = _discover_cookies_file()
        if discovered:
            cookies_file = discovered
            auto_cookies_file = True
            print(f"[TAPA] Using auto-discovered YouTube cookies file: {discovered}",
                  flush=True)

    auto_browser = False
    if cookies_from_browser == "none":
        cookies_from_browser = None
    elif cookies_from_browser == "auto":
        cookies_from_browser = None if cookies_file else _detect_cookie_browser()
        auto_browser = cookies_from_browser is not None
        if auto_browser:
            print(f"[TAPA] Using YouTube cookies from detected browser: "
                  f"{cookies_from_browser} (pass cookies_from_browser='none' to disable).",
                  flush=True)

    ytdlp_error: Optional[BaseException] = None
    try:
        return _download_with_ytdlp(url, out_dir, bitrate,
                                    cookies_file, cookies_from_browser)
    except ImportError:
        # yt-dlp itself isn't installed — let the import error propagate from
        # the fallback path, since pytubefix won't be there either if neither
        # was installed.
        raise
    except Exception as e:
        ytdlp_error = e
        bot_check = _is_bot_check_error(e)
        # Auto-detected cookies (browser or discovered cookies.txt) must never
        # make things worse than the old no-cookies default: if the failure
        # looks cookie-related (locked Chrome DB, keyring, corrupt/malformed
        # cookies.txt), retry without the auto-picked cookies. A bot-check
        # error is not cookie-caused — retrying cookieless there is pointless,
        # so go straight to the pytubefix fallback below.
        if (auto_browser or auto_cookies_file) and not bot_check:
            print(f"[TAPA] yt-dlp failed with auto-detected cookies "
                  f"({e}) — retrying without them...", flush=True)
            try:
                return _download_with_ytdlp(
                    url, out_dir, bitrate,
                    None if auto_cookies_file else cookies_file,
                    None if auto_browser else cookies_from_browser)
            except Exception as retry_err:
                ytdlp_error = retry_err
                bot_check = _is_bot_check_error(retry_err)
        # If the user explicitly supplied cookies, they're authoritative —
        # there's no point in falling back to pytubefix, which doesn't use them.
        if (cookies_file and not auto_cookies_file) or \
                (cookies_from_browser and not auto_browser):
            raise
        # Only fall back on bot-check errors. Other yt-dlp failures (private
        # video, region-blocked, network down) won't be fixed by switching libs.
        if not bot_check:
            raise ytdlp_error
        print(f"[TAPA] yt-dlp blocked by YouTube bot check — trying pytubefix fallback...",
              flush=True)

    try:
        path = _download_with_pytubefix(url, out_dir, bitrate)
        print(f"[TAPA] pytubefix fallback succeeded.", flush=True)
        return path
    except Exception as pytube_err:
        raise RuntimeError(
            f"Both yt-dlp and pytubefix failed to download {url}.\n"
            f"  yt-dlp error: {ytdlp_error}\n"
            f"  pytubefix error: {pytube_err}\n\n"
            f"YouTube is hard-blocking this machine's IP for this video (very common\n"
            f"on Colab / cloud runtimes). The reliable fix is real cookies from a\n"
            f"logged-in browser:\n"
            f"  1. On your own computer, install the 'Get cookies.txt LOCALLY'\n"
            f"     extension (Chrome web store / Firefox add-ons).\n"
            f"  2. Open youtube.com while logged in and click the extension to\n"
            f"     export a cookies.txt file.\n"
            f"  3. Upload it here keeping the name cookies.txt — on Colab: folder\n"
            f"     icon in the left sidebar -> upload (it lands in /content/).\n"
            f"  4. Re-run the same cell. TAPA auto-discovers cookies.txt in\n"
            f"     /content/, the working directory, ~, or $TAPA_YT_COOKIES —\n"
            f"     no config change needed.\n"
            f"If you already uploaded cookies and still see this, your yt-dlp is\n"
            f"likely outdated: run `pip install -U yt-dlp` and restart the runtime."
        ) from pytube_err
