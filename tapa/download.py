"""YouTube → mp3 downloader for the TAPA pipeline.

Wraps `yt-dlp` to fetch a video's audio stream and transcode to mp3 via ffmpeg.
The resulting filename uses the YouTube video ID as the stem so downstream
results (e.g. `<id>_diarization.csv`) are easy to trace back to the source.
"""

from __future__ import annotations

import os
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
    # /shorts/<id>, /embed/<id>, /v/<id>
    parts = [p for p in u.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in ("shorts", "embed", "v"):
        return parts[1]
    return None


def download_youtube_audio(url: str, output_dir: str | os.PathLike,
                           bitrate: str = "192",
                           cookies_file: Optional[str] = None,
                           cookies_from_browser: Optional[str] = None) -> str:
    """Download `url` and transcode to mp3 in `output_dir`.

    Args:
        url: YouTube URL (any of the standard forms).
        output_dir: Directory to write the mp3 into. Created if missing.
        bitrate: mp3 bitrate in kbps as a string (yt-dlp's preferredquality).
        cookies_file: Path to a Netscape-format cookies.txt exported from a
            logged-in browser. Use this on Colab / cloud IPs that still
            hit "Sign in to confirm you're not a bot" even with the default
            alternate-client workaround.
        cookies_from_browser: Browser name (e.g. "chrome", "firefox", "edge")
            to read cookies from directly. Only works when that browser's
            profile is on the same machine — has no effect on Colab.

    Returns:
        Absolute path to the resulting `.mp3` file.

    Raises:
        ImportError: yt-dlp not installed.
        RuntimeError: ffmpeg missing or download/transcode failed.
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError(
            "yt-dlp is required for YouTube downloads. Install with `pip install yt-dlp`."
        ) from e

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default workaround for YouTube's "Sign in to confirm you're not a bot"
    # challenge on datacenter IPs: ask yt-dlp for non-web player clients,
    # which return playable streams without the web bot check. Users can
    # supply real cookies via cookies_file / cookies_from_browser if needed.
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
        raise RuntimeError(f"Could not determine video ID for {url}")

    mp3_path = out_dir / f"{video_id}.mp3"
    if not mp3_path.exists():
        raise RuntimeError(
            f"Expected {mp3_path} after download, but it does not exist. "
            "ffmpeg may have failed to transcode."
        )
    return str(mp3_path.resolve())
