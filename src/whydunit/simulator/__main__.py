"""CLI entry point: ``python -m whydunit.simulator``.

Example:

    python -m whydunit.simulator --scenario retrieval_degradation \\
        --days 7 --frequency 1min --seed 42 --output data/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from whydunit.models import IncidentCategory
from whydunit.simulator.generator import generate_dataset, write_dataset

DEFAULT_START = "2026-09-01T00:00:00+00:00"


def _parse_start(text: str) -> datetime:
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError(f"start must be timezone-aware ISO-8601, got {text!r}")
    return moment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m whydunit.simulator",
        description="Generate a synthetic AI-pipeline incident dataset.",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=[category.value for category in IncidentCategory],
        help="incident scenario to inject",
    )
    parser.add_argument("--days", type=float, default=7.0, help="data window length (default 7)")
    parser.add_argument(
        "--frequency", default="1min", help="sampling interval, pandas offset (default 1min)"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument(
        "--start",
        type=_parse_start,
        default=DEFAULT_START,
        help=f"window start, tz-aware ISO-8601 (default {DEFAULT_START} — fixed for determinism)",
    )
    parser.add_argument(
        "--incident-start-frac",
        type=float,
        default=0.5,
        help="fraction of the window at which the incident begins (default 0.5)",
    )
    parser.add_argument(
        "--incident-hours",
        type=float,
        default=2.0,
        help="incident duration in hours (default 2)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data"), help="output directory (default data/)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = args.start if isinstance(args.start, datetime) else _parse_start(args.start)

    dataset = generate_dataset(
        IncidentCategory(args.scenario),
        start=start,
        days=args.days,
        freq=args.frequency,
        seed=args.seed,
        incident_start_frac=args.incident_start_frac,
        incident_duration=timedelta(hours=args.incident_hours),
    )
    telemetry_path, scenario_path, truth_path = write_dataset(dataset, args.output)

    scenario = dataset.spec.scenario
    print(f"scenario   : {scenario.id} ({scenario.category.value})")
    print(f"window     : {args.days} days at {args.frequency}, seed {args.seed}")
    print(f"incident   : {scenario.start.isoformat()} for {scenario.duration}")
    print(f"telemetry  : {telemetry_path} ({len(dataset.telemetry)} rows)")
    print(f"scenario   : {scenario_path}")
    print(f"groundtruth: {truth_path} (eval-only — do not use while investigating)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
