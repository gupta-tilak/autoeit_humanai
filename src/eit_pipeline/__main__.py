"""CLI entry point for the EIT pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import PipelineConfig
from .pipeline_runner import EITPipeline, run_pipeline
from .stimulus_extractor import extract_stimuli_from_recording
from .utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EIT Stimulus-Alignment Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # ── run ─────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to YAML config file",
    )
    run_parser.add_argument(
        "--compare-methods", action="store_true",
        help="Run all 3 alignment methods for comparison",
    )
    run_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    # ── extract-stimuli ────────────────────────────────────
    extract_parser = subparsers.add_parser(
        "extract-stimuli",
        help="Extract stimulus audio from a reference recording",
    )
    extract_parser.add_argument(
        "audio_path", type=str,
        help="Path to reference EIT recording",
    )
    extract_parser.add_argument(
        "--output-dir", "-o", type=str, default="data/stimulus_audio",
        help="Output directory for stimulus WAV files",
    )
    extract_parser.add_argument(
        "--skip-seconds", type=float, default=150.0,
        help="Seconds to skip from beginning",
    )
    extract_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    # ── export-config ──────────────────────────────────────
    export_parser = subparsers.add_parser(
        "export-config",
        help="Export default configuration to YAML",
    )
    export_parser.add_argument(
        "--output", "-o", type=str, default="configs/stimulus_alignment.yaml",
        help="Output path",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        level = logging.DEBUG if args.verbose else logging.INFO
        setup_logging(level=level, log_file="output/pipeline.log")

        config = None
        if args.config:
            config = PipelineConfig.from_yaml(args.config)
        else:
            config = PipelineConfig()

        if args.compare_methods:
            config.stimulus_alignment.compare_all_methods = True

        results = run_pipeline(config=config)

        success = sum(1 for r in results if "error" not in r)
        print(f"\nPipeline complete: {success}/{len(results)} participants processed.")

    elif args.command == "extract-stimuli":
        level = logging.DEBUG if args.verbose else logging.INFO
        setup_logging(level=level)

        extracted = extract_stimuli_from_recording(
            audio_path=args.audio_path,
            output_dir=args.output_dir,
            skip_seconds=args.skip_seconds,
        )
        print(f"\nExtracted {len(extracted)} stimuli to {args.output_dir}")

    elif args.command == "export-config":
        config = PipelineConfig()
        config.to_yaml(args.output)
        print(f"Config exported to {args.output}")


if __name__ == "__main__":
    main()
