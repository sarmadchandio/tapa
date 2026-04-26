"""Command-line interface for the TAPA pipeline."""

import argparse
import sys

from .config import TAPAConfig
from .download import download_youtube_audio, is_youtube_url
from .pipeline import TAPAPipeline


def main():
    parser = argparse.ArgumentParser(
        prog="tapa",
        description="TAPA: Speaker diarization + phonetic analysis pipeline",
    )
    parser.add_argument("audio", nargs="+",
                        help="Audio file(s) or YouTube URL(s). Audio: .mp3, .wav, .flac.")
    parser.add_argument("-o", "--output", default="results/", help="Output directory (default: results/)")
    parser.add_argument("--whisper-model", default="small.en", help="Whisper model name (default: small.en)")
    parser.add_argument("--num-speakers", type=int, default=None, help="Number of speakers (default: auto-detect)")
    parser.add_argument("--mfa-bin", default=None, help="Path to MFA binary (default: auto-detect)")
    parser.add_argument("--audio-dir", default="audio/",
                        help="Where to save audio downloaded from URLs (default: audio/)")
    parser.add_argument("--mp3-bitrate", default="192",
                        help="mp3 bitrate (kbps) for downloaded audio (default: 192)")
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
        whisper_model=args.whisper_model,
        num_speakers=args.num_speakers,
        mfa_bin=args.mfa_bin,
        vot_backend=args.vot_backend,
        drvot_repo_dir=args.drvot_repo,
        drvot_python=args.drvot_python,
        drvot_keep_temp=args.drvot_keep_temp,
    )
    pipeline = TAPAPipeline(config=cfg)

    for inp in args.audio:
        if is_youtube_url(inp):
            print(f"Downloading {inp} -> {args.audio_dir} (mp3 @ {args.mp3_bitrate}k)")
            audio_path = download_youtube_audio(inp, args.audio_dir, bitrate=args.mp3_bitrate)
            print(f"Saved {audio_path}")
        else:
            audio_path = inp
        pipeline.run(audio_path, results_dir=args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
