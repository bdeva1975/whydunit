"""Tests for the evaluation harness — the engine's report card."""

import pytest

from whydunit.evaluation import Outcome, run_evaluation
from whydunit.models import IncidentCategory

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


@pytest.fixture(scope="module")
def report():
    return run_evaluation(days=1.0, seed=42)


def test_covers_every_category(report) -> None:
    assert {r.category for r in report.results} == set(IncidentCategory)


def test_clear_cut_scenarios_all_correct(report) -> None:
    by_category = {r.category: r for r in report.results}
    wrong = [c.value for c in CLEAR_CUT if by_category[c].outcome is not Outcome.CORRECT]
    assert wrong == [], f"clear-cut scenarios not top-1: {wrong}"


def test_no_false_positive_on_normal(report) -> None:
    assert not report.false_positive


def test_no_false_negatives(report) -> None:
    assert report.false_negatives == ()


def test_hard_scenarios_at_least_partial(report) -> None:
    by_category = {r.category: r for r in report.results}
    for category in (IncidentCategory.CASCADING_FAILURE, IncidentCategory.MULTI_FACTOR):
        assert by_category[category].outcome in (Outcome.CORRECT, Outcome.PARTIAL), (
            f"{category.value} was missed entirely"
        )


def test_detection_delays_are_reasonable(report) -> None:
    for r in report.results:
        if r.detection_delay_minutes is not None:
            assert -15.0 <= r.detection_delay_minutes <= 60.0, (
                f"{r.category.value}: delay {r.detection_delay_minutes:.0f} min"
            )


def test_markdown_report_renders(report) -> None:
    text = report.to_markdown()
    assert "| Scenario |" in text
    assert "retrieval_degradation" in text
    assert "Top-1 diagnostic accuracy" in text
