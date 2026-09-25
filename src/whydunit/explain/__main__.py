"""CLI: generate a scenario, investigate it, and print a validated narrative.

Usage:
    python -m whydunit.explain --scenario retrieval_degradation --seed 42

Requires the ``llm`` extra and ``ANTHROPIC_API_KEY``. Everything before
the narrative is the deterministic v0.1 pipeline; only the final prose
step calls the model.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from whydunit.explain.narrative import DEFAULT_MODEL
from whydunit.explain.report import NarrativeValidationError, narrate
from whydunit.forensics import case_file, investigate
from whydunit.models import IncidentCategory
from whydunit.simulator import generate_dataset, to_wide

DEFAULT_START = datetime(2026, 9, 1, tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m whydunit.explain",
        description="Narrate a Whydunit investigation via the Anthropic API.",
    )
    parser.add_argument(
        "--scenario",
        default=IncidentCategory.RETRIEVAL_DEGRADATION.value,
        choices=[category.value for category in IncidentCategory],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=float, default=1.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Also write the narrative to this file (UTF-8 Markdown).",
    )
    args = parser.parse_args(argv)

    print(f"Generating scenario '{args.scenario}' (seed {args.seed})...")
    dataset = generate_dataset(
        IncidentCategory(args.scenario),
        start=DEFAULT_START,
        days=args.days,
        seed=args.seed,
    )

    print("Running the deterministic forensic engine...")
    result = investigate(to_wide(dataset.telemetry))
    case = case_file(result)
    print(f"Top hypothesis: {case.hypotheses[0].statement} [{case.hypotheses[0].confidence.value}]")

    print(f"Requesting narrative from {args.model}...")
    try:
        report = narrate(case, model=args.model, max_attempts=args.max_attempts)
    except NarrativeValidationError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1

    print(
        f"Validated on attempt {report.attempts}; cites: {', '.join(report.validation.cited_ids)}"
    )
    print()
    print(report.text)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(report.text + "\n", encoding="utf-8")
        print(f"\nSaved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
