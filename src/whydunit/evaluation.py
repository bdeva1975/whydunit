"""Evaluation harness: score the forensic engine against ground truth.

This module — and only this module, besides the simulator itself — reads
:class:`GroundTruth`. It exists to answer, with numbers: *does the
forensic engine actually work?* A test enforces that nothing under
``whydunit.forensics`` touches ground truth; this harness is the
consumer on the other side of that firewall.

Grading rules, stated in code rather than prose:

* ordinary scenarios — CORRECT if the true category is the top-1
  hypothesis, PARTIAL if it appears in the top 3, else MISSED;
* NORMAL — CORRECT only if the engine's top hypothesis is "no
  incident"; anything else is a false positive;
* MULTI_FACTOR — there is deliberately no multi-factor signature, so
  the engine is graded on surfacing BOTH component faults in the top 4:
  both → CORRECT, one → PARTIAL, none → MISSED;
* CASCADING_FAILURE — top-1 CORRECT, top-3 PARTIAL, and top-1
  ``schema_drift`` also PARTIAL: same ingestion origin, coarser label.

Metric limitations, stated plainly: one seed, one incident placement,
twelve scenarios — this measures "the engine distinguishes the failure
modes it was designed for", not production accuracy. The CLI accepts
``--seed`` so the claim can be checked under other draws.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from whydunit.forensics import investigate
from whydunit.models import IncidentCategory
from whydunit.simulator import generate_dataset, to_wide

DEFAULT_START = datetime(2026, 9, 1, tzinfo=UTC)
TOP_K = 3
MULTI_TOP_K = 4
MULTI_FACTOR_COMPONENTS = (
    IncidentCategory.RETRIEVAL_DEGRADATION,
    IncidentCategory.RESOURCE_EXHAUSTION,
)
CASCADING_COARSE_TWIN = IncidentCategory.SCHEMA_DRIFT


class Outcome(StrEnum):
    CORRECT = "correct"
    PARTIAL = "partial"
    MISSED = "missed"


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """The engine's graded performance on one scenario."""

    category: IncidentCategory
    predicted: tuple[IncidentCategory, ...]
    outcome: Outcome
    detection_delay_minutes: float | None
    top_score: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate results across all scenarios."""

    results: tuple[ScenarioResult, ...]
    days: float
    seed: int

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.outcome is Outcome.CORRECT)

    @property
    def partial(self) -> int:
        return sum(1 for r in self.results if r.outcome is Outcome.PARTIAL)

    @property
    def missed(self) -> int:
        return sum(1 for r in self.results if r.outcome is Outcome.MISSED)

    @property
    def top1_accuracy(self) -> float:
        return self.correct / len(self.results)

    @property
    def false_positive(self) -> bool:
        """Did the engine call an incident on the NORMAL scenario?"""
        normal = next(r for r in self.results if r.category is IncidentCategory.NORMAL)
        return normal.outcome is not Outcome.CORRECT

    @property
    def false_negatives(self) -> tuple[IncidentCategory, ...]:
        """Incident scenarios the engine called healthy."""
        return tuple(
            r.category
            for r in self.results
            if r.category is not IncidentCategory.NORMAL
            and r.predicted[:1] == (IncidentCategory.NORMAL,)
        )

    def to_markdown(self) -> str:
        lines = [
            "| Scenario | Top-1 prediction | Outcome | Detection delay |",
            "|---|---|---|---|",
        ]
        for r in self.results:
            top1 = r.predicted[0].value if r.predicted else "—"
            delay = (
                f"{r.detection_delay_minutes:.0f} min"
                if r.detection_delay_minutes is not None
                else "—"
            )
            lines.append(f"| {r.category.value} | {top1} | {r.outcome.value} | {delay} |")
        lines += [
            "",
            f"Top-1 diagnostic accuracy: **{self.correct}/{len(self.results)}** "
            f"({self.top1_accuracy:.0%}); partial: {self.partial}; missed: {self.missed}.",
            f"False positive on normal operation: {'yes' if self.false_positive else 'no'}. "
            f"False negatives: {len(self.false_negatives)}.",
        ]
        return "\n".join(lines)


def _judge(category: IncidentCategory, predicted: tuple[IncidentCategory, ...]) -> Outcome:
    if category is IncidentCategory.NORMAL:
        top1_is_normal = bool(predicted) and predicted[0] is IncidentCategory.NORMAL
        return Outcome.CORRECT if top1_is_normal else Outcome.MISSED

    if category is IncidentCategory.MULTI_FACTOR:
        hits = sum(1 for c in MULTI_FACTOR_COMPONENTS if c in predicted[:MULTI_TOP_K])
        if hits == len(MULTI_FACTOR_COMPONENTS):
            return Outcome.CORRECT
        return Outcome.PARTIAL if hits == 1 else Outcome.MISSED

    if predicted and predicted[0] is category:
        return Outcome.CORRECT
    if category in predicted[:TOP_K]:
        return Outcome.PARTIAL
    if (
        category is IncidentCategory.CASCADING_FAILURE
        and predicted
        and predicted[0] is CASCADING_COARSE_TWIN
    ):
        return Outcome.PARTIAL
    return Outcome.MISSED


def evaluate_scenario(
    category: IncidentCategory,
    *,
    start: datetime = DEFAULT_START,
    days: float = 1.0,
    seed: int = 42,
) -> ScenarioResult:
    """Generate one scenario, investigate it blind, grade against truth."""
    dataset = generate_dataset(category, start=start, days=days, seed=seed)
    result = investigate(to_wide(dataset.telemetry))
    predicted = tuple(h.category for h in result.hypotheses)

    delay: float | None = None
    truth = dataset.truth
    if category is not IncidentCategory.NORMAL and result.is_incident:
        delay = (result.window_start - truth.incident_start) / timedelta(minutes=1)

    return ScenarioResult(
        category=category,
        predicted=predicted,
        outcome=_judge(category, predicted),
        detection_delay_minutes=delay,
        top_score=result.hypotheses[0].score if result.hypotheses else 0.0,
    )


def run_evaluation(
    *,
    start: datetime = DEFAULT_START,
    days: float = 1.0,
    seed: int = 42,
) -> EvalReport:
    """Run the engine against every scenario category."""
    results = tuple(
        evaluate_scenario(category, start=start, days=days, seed=seed)
        for category in IncidentCategory
    )
    return EvalReport(results=results, days=days, seed=seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m whydunit.evaluation",
        description="Score the forensic engine against synthetic ground truth.",
    )
    parser.add_argument("--days", type=float, default=1.0, help="data window (default 1)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    args = parser.parse_args(argv)

    report = run_evaluation(days=args.days, seed=args.seed)
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())
