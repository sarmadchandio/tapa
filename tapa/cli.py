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
    parser.add_argument("audio", nargs="+", help="Audio file(s) to process (.mp3, .wav, .flac)")
    parser.add_argument("-o", "--output", default="results/", help="Output directory (default: results/)")
    parser.add_argument("--whisper-model", default="small.en", help="Whisper model name (default: small.en)")
    parser.add_argument("--num-speakers", type=int, default=None, help="Number of speakers (default: auto-detect)")
    parser.add_argument("--mfa-bin", default=None, help="Path to MFA binary (default: auto-detect)")
    args = parser.parse_args()

    cfg = TAPAConfig(
        results_dir=args.output,
        whisper_model=args.whisper_model,
        num_speakers=args.num_speakers,
        mfa_bin=args.mfa_bin,
    )
    pipeline = TAPAPipeline(config=cfg)

    for audio_path in args.audio:
        pipeline.run(audio_path, results_dir=args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
