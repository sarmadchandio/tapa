"""Command-line interface for the TAPA pipeline."""

import argparse
import sys

from .config import TAPAConfig
from .pipeline import TAPAPipeline


def main():
    parser = argparse.ArgumentParser(
        prog="tapa",
        description="TAPA: Speaker diarization + phonetic analysis pipeline",
    )
    parser.add_argument("audio", nargs="+",
                        help="Audio file(s) or YouTube URL(s). "
                             "Audio formats: .mp3, .wav, .flac. URLs are "
                             "downloaded to --audio-dir as mp3 first.")
    parser.add_argument("-o", "--output", default="results/", help="Output directory (default: results/)")
    parser.add_argument("--whisper-model", default="small.en", help="Whisper model name (default: small.en)")
    parser.add_argument("--num-speakers", type=int, default=None, help="Number of speakers (default: auto-detect)")
    parser.add_argument("--mfa-bin", default=None, help="Path to MFA binary (default: auto-detect)")
    parser.add_argument("--audio-dir", default="audio/",
                        help="Where to save audio downloaded from URLs (default: audio/)")
    parser.add_argument("--mp3-bitrate", default="192",
                        help="mp3 bitrate (kbps) for downloaded audio (default: 192)")
    parser.add_argument("--yt-cookies", default=None,
                        help="Path to a Netscape cookies.txt exported from a logged-in "
                             "browser. Use if YouTube serves the bot challenge on "
                             "Colab/cloud IPs. Optional — alternate player_client "
                             "workaround is applied by default.")
    parser.add_argument("--yt-cookies-from-browser", default=None,
                        help="Read YouTube cookies directly from a local browser "
                             "(e.g. chrome, firefox, edge). Only works on machines "
                             "where that browser is installed; not useful on Colab.")
    parser.add_argument("--vot-backend", choices=["tapa", "drvot"], default="tapa",
                        help="Stop-VOT backend: 'tapa' (Praat-based) or 'drvot' "
                             "(Dr.VOT CNN with per-token TAPA fallback). Default: tapa")
    parser.add_argument("--drvot-repo", default=None,
                        help="Path to a local Dr.VOT clone (required when --vot-backend=drvot). "
                             "Use `python -m tapa.drvot setup <dir>` to clone and patch.")
    parser.add_argument("--drvot-python", default=None,
                        help="Python interpreter to invoke Dr.VOT scripts with "
                             "(default: the current Python). Useful when Dr.VOT lives in a sibling venv.")
    parser.add_argument("--drvot-keep-temp", action="store_true",
                        help="Keep the per-recording Dr.VOT temp dir for inspection.")
    args = parser.parse_args()

    if args.vot_backend == "drvot" and not args.drvot_repo:
        parser.error("--vot-backend=drvot requires --drvot-repo PATH")

    cfg = TAPAConfig(
        results_dir=args.output,
        audio_dir=args.audio_dir,
        whisper_model=args.whisper_model,
        num_speakers=args.num_speakers,
        mfa_bin=args.mfa_bin,
        mp3_bitrate=args.mp3_bitrate,
        youtube_cookies_file=args.yt_cookies,
        youtube_cookies_from_browser=args.yt_cookies_from_browser,
        vot_backend=args.vot_backend,
        drvot_repo_dir=args.drvot_repo,
        drvot_python=args.drvot_python,
        drvot_keep_temp=args.drvot_keep_temp,
    )
    pipeline = TAPAPipeline(config=cfg)

    # pipeline.run() handles URL-vs-path dispatch internally — just pass through.
    for inp in args.audio:
        pipeline.run(inp, results_dir=args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
