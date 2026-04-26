"""YouTube → mp3 downloader for the TAPA pipeline.

Wraps `yt-dlp` to fetch a video's audio stream and transcode to mp3 via ffmpeg.
The resulting filename uses the YouTube video ID as the stem so downstream
results (e.g. `<id>_diarization.csv`) are easy to trace back to the source.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse


YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be",
}


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
                           bitrate: str = "192") -> str:
    """Download `url` and transcode to mp3 in `output_dir`.

    Args:
        url: YouTube URL (any of the standard forms).
        output_dir: Directory to write the mp3 into. Created if missing.
        bitrate: mp3 bitrate in kbps as a string (yt-dlp's preferredquality).

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

    # Use the video ID as the stem so downstream results are traceable.
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": bitrate,
        }],
    }

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
