"""CLI entry point for the AutoEIT pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="AutoEIT — Automated Elicited Imitation Task Transcription Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config (MLX-Whisper small)
  python -m autoeit run

  # Run with a specific config file
  python -m autoeit run --config configs/whisper_cpp.yaml

  # Run only one participant
  python -m autoeit run --participants 038010

  # List previous experiment runs
  python -m autoeit list-runs

  # Compare two runs
  python -m autoeit compare --runs run_id_1 run_id_2

  # Export default config
  python -m autoeit export-config --output my_config.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run the transcription pipeline")
    run_parser.add_argument("--config", "-c", help="Path to YAML config file")
    run_parser.add_argument("--name", "-n", default="", help="Experiment name")
    run_parser.add_argument("--description", "-d", default="", help="Experiment description")
    run_parser.add_argument("--tags", nargs="*", default=[], help="Tags for this run")
    run_parser.add_argument("--participants", nargs="*", help="Only process these participant IDs")
    run_parser.add_argument("--backend", choices=["mlx_whisper", "whisper_cpp", "faster_whisper"],
                            help="Override ASR backend")
    run_parser.add_argument("--model-size", choices=["tiny", "base", "small", "medium", "large"],
                            help="Override model size")
    run_parser.add_argument("--base-dir", default=None, help="Project root directory")

    # --- list-runs ---
    list_parser = subparsers.add_parser("list-runs", help="List all experiment runs")
    list_parser.add_argument("--experiments-dir", default="experiments", help="Experiments directory")

    # --- compare ---
    compare_parser = subparsers.add_parser("compare", help="Compare experiment runs")
    compare_parser.add_argument("--runs", nargs="+", required=True, help="Run IDs to compare")
    compare_parser.add_argument("--experiments-dir", default="experiments", help="Experiments directory")

    # --- export-config ---
    export_parser = subparsers.add_parser("export-config", help="Export default configuration")
    export_parser.add_argument("--output", "-o", default="config.yaml", help="Output path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "list-runs":
        _cmd_list_runs(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "export-config":
        _cmd_export_config(args)


def _cmd_run(args):
    from .config import PipelineConfig
    from .pipeline import AutoEITPipeline

    # Load config
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    # Apply overrides
    if args.backend:
        config.asr.backend = args.backend
    if args.model_size:
        config.asr.model_size = args.model_size
    if args.name:
        config.experiment_name = args.name
    if args.description:
        config.experiment_description = args.description
    if args.tags:
        config.tags = args.tags

    base_dir = args.base_dir or str(Path.cwd())

    # Run pipeline
    pipeline = AutoEITPipeline(config, base_dir=base_dir)
    run = pipeline.run(
        participant_ids=args.participants,
        name=config.experiment_name,
        description=config.experiment_description,
        tags=config.tags,
    )

    print(f"\nExperiment completed: {run.run_id}")
    print(f"  Sentences:  {run.total_sentences}")
    print(f"  Flagged:    {run.total_flagged}")
    print(f"  Duration:   {run.duration_s:.1f}s")
    print(f"  Saved to:   experiments/{run.run_id}/")


def _cmd_list_runs(args):
    from .experiment_logger import ExperimentLogger

    logger = ExperimentLogger(args.experiments_dir)
    runs = logger.list_runs()

    if not runs:
        print("No experiment runs found.")
        return

    print(f"{'Run ID':<40} {'Name':<20} {'Backend':<15} {'Model':<10} {'Sentences':>10} {'Flagged':>8} {'Duration':>10}")
    print("-" * 113)
    for r in runs:
        print(
            f"{r['run_id']:<40} {r.get('name', ''):<20} {r.get('backend', ''):<15} "
            f"{r.get('model', ''):<10} {r.get('total_sentences', 0):>10} "
            f"{r.get('total_flagged', 0):>8} {r.get('duration_s', 0):>9.1f}s"
        )


def _cmd_compare(args):
    from .experiment_logger import ExperimentLogger

    logger = ExperimentLogger(args.experiments_dir)
    comparison = logger.compare_runs(args.runs)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


def _cmd_export_config(args):
    from .config import PipelineConfig

    config = PipelineConfig()
    config.to_yaml(args.output)
    print(f"Default config exported to: {args.output}")


if __name__ == "__main__":
    main()
