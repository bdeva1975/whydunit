"""Tests for the hypothesis engine."""

from datetime import UTC, datetime

import pytest

from whydunit.forensics.anomaly import detect_anomalies
from whydunit.forensics.hypotheses import SIGNATURES, generate_hypotheses
from whydunit.models import ConfidenceBand, IncidentCategory
from whydunit.simulator import generate_dataset, to_wide
from whydunit.simulator.telemetry import generate_baseline

START = datetime(2026, 9, 1, tzinfo=UTC)

CLEAR_CUT = [
    IncidentCategory.SCHEMA_DRIFT,
    IncidentCategory.DATA_QUALITY_DEGRADATION,
    IncidentCategory.FEATURE_DRIFT,
    IncidentCategory.RETRIEVAL_DEGRADATION,
    IncidentCategory.LLM_LATENCY_SPIKE,
    IncidentCategory.MODEL_REGRESSION,
    IncidentCategory.PROMPT_REGRESSION,
    IncidentCategory.API_RATE_LIMITING,
    IncidentCategory.RESOURCE_EXHAUSTION,
]


def _hypotheses_for(category: IncidentCategory):
    wide = to_wide(generate_dataset(category, start=START, days=1.0, seed=42).telemetry)
    return generate_hypotheses(detect_anomalies(wide))


def test_signature_signals_exist_in_catalogue() -> None:
    from whydunit.simulator.pipeline import metric_spec

    for signature in SIGNATURES:
        for move in signature.expected:
            metric_spec(move.signal)  # raises KeyError if unknown
        for ref in signature.quiet:
            metric_spec(ref)


def test_no_anomalies_yields_normal() -> None:
    hypotheses = generate_hypotheses([])
    assert len(hypotheses) == 1
    assert hypotheses[0].category is IncidentCategory.NORMAL
    assert hypotheses[0].confidence is ConfidenceBand.STRONG


def test_healthy_telemetry_diagnosed_normal() -> None:
    wide = to_wide(generate_baseline(START, periods=1440, seed=42))
    hypotheses = generate_hypotheses(detect_anomalies(wide))
    assert hypotheses[0].category is IncidentCategory.NORMAL


@pytest.mark.parametrize("category", CLEAR_CUT, ids=lambda c: c.value)
def test_clear_cut_scenarios_top1(category: IncidentCategory) -> None:
    hypotheses = _hypotheses_for(category)
    assert hypotheses, f"{category.value}: no hypotheses generated"
    assert hypotheses[0].category is category, (
        f"{category.value}: top hypothesis was {hypotheses[0].category.value} "
        f"(score {hypotheses[0].score})"
    )


def test_cascading_failure_in_top3() -> None:
    hypotheses = _hypotheses_for(IncidentCategory.CASCADING_FAILURE)
    top3 = [h.category for h in hypotheses[:3]]
    assert IncidentCategory.CASCADING_FAILURE in top3


def test_multi_factor_surfaces_both_faults() -> None:
    hypotheses = _hypotheses_for(IncidentCategory.MULTI_FACTOR)
    top = [h.category for h in hypotheses[:4]]
    assert IncidentCategory.RETRIEVAL_DEGRADATION in top
    assert IncidentCategory.RESOURCE_EXHAUSTION in top


def test_hypotheses_carry_evidence_both_ways() -> None:
    # Clean single-fault case: the sole survivor needs no contradictions.
    retrieval = _hypotheses_for(IncidentCategory.RETRIEVAL_DEGRADATION)
    assert retrieval[0].supporting, "top hypothesis has no supporting evidence"
    assert retrieval[0].next_step

    # Ambiguous multi-factor case: several hypotheses survive, and the
    # weaker ones must carry the evidence that speaks against them.
    multi = _hypotheses_for(IncidentCategory.MULTI_FACTOR)
    assert len(multi) > 1, "multi-factor should not produce a single clean diagnosis"
    assert multi[-1].contradicting, "weakest surviving hypothesis lacks contradicting evidence"


def test_scores_sorted_descending() -> None:
    hypotheses = _hypotheses_for(IncidentCategory.SCHEMA_DRIFT)
    scores = [h.score for h in hypotheses]
    assert scores == sorted(scores, reverse=True)
