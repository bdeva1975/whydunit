"""Tests for incident injection."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from whydunit.models import IncidentCategory, Stage
from whydunit.simulator.pipeline import metric_spec
from whydunit.simulator.scenarios import build_scenario, ground_truth_for, inject
from whydunit.simulator.telemetry import generate_baseline

START = datetime(2026, 9, 1, tzinfo=UTC)
PERIODS = 12 * 60  # 12 hours at 1min
INCIDENT_START = START + timedelta(hours=6)
DURATION = timedelta(hours=2)


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    return generate_baseline(START, periods=PERIODS, seed=42)


@pytest.fixture(scope="module")
def schema_drift(baseline: pd.DataFrame) -> pd.DataFrame:
    return inject(baseline, build_scenario(IncidentCategory.SCHEMA_DRIFT, INCIDENT_START, DURATION))


def _series(frame: pd.DataFrame, stage: Stage, metric: str) -> pd.Series:
    rows = frame[(frame["stage"] == stage.value) & (frame["metric"] == metric)]
    return rows.set_index("timestamp")["value"]


def test_injection_is_deterministic(baseline: pd.DataFrame) -> None:
    spec = build_scenario(IncidentCategory.RETRIEVAL_DEGRADATION, INCIDENT_START, DURATION)
    pd.testing.assert_frame_equal(inject(baseline, spec), inject(baseline, spec))


def test_input_frame_is_not_mutated(baseline: pd.DataFrame) -> None:
    copy = baseline.copy(deep=True)
    inject(baseline, build_scenario(IncidentCategory.SCHEMA_DRIFT, INCIDENT_START, DURATION))
    pd.testing.assert_frame_equal(baseline, copy)


def test_effects_confined_to_window(baseline: pd.DataFrame, schema_drift: pd.DataFrame) -> None:
    before = baseline[baseline["timestamp"] < INCIDENT_START]
    after_start_mask = schema_drift["timestamp"] < INCIDENT_START
    pd.testing.assert_frame_equal(
        before.reset_index(drop=True),
        schema_drift[after_start_mask].reset_index(drop=True),
    )
    end = INCIDENT_START + DURATION
    pd.testing.assert_frame_equal(
        baseline[baseline["timestamp"] >= end].reset_index(drop=True),
        schema_drift[schema_drift["timestamp"] >= end].reset_index(drop=True),
    )


def test_primary_signal_moves_up(baseline: pd.DataFrame, schema_drift: pd.DataFrame) -> None:
    window = slice(INCIDENT_START + timedelta(minutes=30), INCIDENT_START + DURATION)
    healthy = _series(baseline, Stage.INGESTION, "schema_violations")[window]
    faulty = _series(schema_drift, Stage.INGESTION, "schema_violations")[window]
    spec = metric_spec_for(Stage.INGESTION, "schema_violations")
    assert faulty.mean() - healthy.mean() > 5 * spec.noise_std


def test_downstream_quality_moves_down(baseline: pd.DataFrame, schema_drift: pd.DataFrame) -> None:
    window = slice(INCIDENT_START + timedelta(minutes=75), INCIDENT_START + DURATION)
    healthy = _series(baseline, Stage.EVALUATION, "eval_score")[window]
    faulty = _series(schema_drift, Stage.EVALUATION, "eval_score")[window]
    assert faulty.mean() < healthy.mean()


def test_temporal_ordering_primary_before_downstream(
    baseline: pd.DataFrame, schema_drift: pd.DataFrame
) -> None:
    def first_divergence(stage: Stage, metric: str) -> pd.Timestamp:
        healthy = _series(baseline, stage, metric)
        faulty = _series(schema_drift, stage, metric)
        diverged = (healthy - faulty).abs() > 1e-12
        return diverged[diverged].index[0]

    primary_at = first_divergence(Stage.INGESTION, "schema_violations")
    downstream_at = first_divergence(Stage.EVALUATION, "eval_score")
    assert primary_at < downstream_at


def test_bounds_still_respected(schema_drift: pd.DataFrame) -> None:
    for (stage_value, metric), group in schema_drift.groupby(["stage", "metric"]):
        spec = metric_spec_for(Stage(stage_value), metric)
        assert group["value"].min() >= spec.low, f"{stage_value}.{metric}"
        assert group["value"].max() <= spec.high, f"{stage_value}.{metric}"


def test_ground_truth_alignment() -> None:
    spec = build_scenario(IncidentCategory.RETRIEVAL_DEGRADATION, INCIDENT_START, DURATION)
    truth = ground_truth_for(spec)
    assert truth.category is IncidentCategory.RETRIEVAL_DEGRADATION
    assert truth.root_cause_stage is Stage.RETRIEVAL
    assert truth.incident_end == INCIDENT_START + DURATION
    assert truth.injected_signals[0] == spec.scenario.primary_signal


def test_unknown_category_raises() -> None:
    with pytest.raises(KeyError):
        build_scenario(IncidentCategory.MULTI_FACTOR, INCIDENT_START, DURATION)


def metric_spec_for(stage: Stage, metric: str):
    from whydunit.models import SignalRef

    return metric_spec(SignalRef(stage=stage, metric=metric))
