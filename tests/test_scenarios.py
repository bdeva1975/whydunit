"""Tests for incident scenarios and injection."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from whydunit.models import IncidentCategory, SignalRef, Stage
from whydunit.simulator.pipeline import metric_spec
from whydunit.simulator.scenarios import (
    SCENARIO_BUILDERS,
    build_scenario,
    ground_truth_for,
    inject,
)
from whydunit.simulator.telemetry import generate_baseline

START = datetime(2026, 9, 1, tzinfo=UTC)
PERIODS = 12 * 60  # 12 hours at 1min
INCIDENT_START = START + timedelta(hours=6)
DURATION = timedelta(hours=2)

NON_NORMAL = [c for c in IncidentCategory if c is not IncidentCategory.NORMAL]


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    return generate_baseline(START, periods=PERIODS, seed=42)


@pytest.fixture(scope="module")
def schema_drift(baseline: pd.DataFrame) -> pd.DataFrame:
    return inject(baseline, build_scenario(IncidentCategory.SCHEMA_DRIFT, INCIDENT_START, DURATION))


def _series(frame: pd.DataFrame, ref: SignalRef) -> pd.Series:
    rows = frame[(frame["stage"] == ref.stage.value) & (frame["metric"] == ref.metric)]
    return rows.set_index("timestamp")["value"]


def test_every_category_has_a_builder() -> None:
    assert set(SCENARIO_BUILDERS) == set(IncidentCategory)


def test_injection_is_deterministic(baseline: pd.DataFrame) -> None:
    spec = build_scenario(IncidentCategory.RETRIEVAL_DEGRADATION, INCIDENT_START, DURATION)
    pd.testing.assert_frame_equal(inject(baseline, spec), inject(baseline, spec))


def test_input_frame_is_not_mutated(baseline: pd.DataFrame) -> None:
    copy = baseline.copy(deep=True)
    inject(baseline, build_scenario(IncidentCategory.SCHEMA_DRIFT, INCIDENT_START, DURATION))
    pd.testing.assert_frame_equal(baseline, copy)


def test_normal_scenario_changes_nothing(baseline: pd.DataFrame) -> None:
    spec = build_scenario(IncidentCategory.NORMAL, INCIDENT_START, DURATION)
    pd.testing.assert_frame_equal(inject(baseline, spec), baseline)
    truth = ground_truth_for(spec)
    assert truth.root_cause_stage is None
    assert truth.injected_signals == ()


def test_effects_confined_to_window(baseline: pd.DataFrame, schema_drift: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        baseline[baseline["timestamp"] < INCIDENT_START].reset_index(drop=True),
        schema_drift[schema_drift["timestamp"] < INCIDENT_START].reset_index(drop=True),
    )
    end = INCIDENT_START + DURATION
    pd.testing.assert_frame_equal(
        baseline[baseline["timestamp"] >= end].reset_index(drop=True),
        schema_drift[schema_drift["timestamp"] >= end].reset_index(drop=True),
    )


def test_temporal_ordering_primary_before_downstream(
    baseline: pd.DataFrame, schema_drift: pd.DataFrame
) -> None:
    def first_divergence(ref: SignalRef) -> pd.Timestamp:
        healthy = _series(baseline, ref)
        faulty = _series(schema_drift, ref)
        diverged = (healthy - faulty).abs() > 1e-12
        return diverged[diverged].index[0]

    primary_at = first_divergence(SignalRef(Stage.INGESTION, "schema_violations"))
    downstream_at = first_divergence(SignalRef(Stage.EVALUATION, "eval_score"))
    assert primary_at < downstream_at


@pytest.mark.parametrize("category", NON_NORMAL, ids=lambda c: c.value)
def test_primary_signal_moves_as_designed(
    baseline: pd.DataFrame, category: IncidentCategory
) -> None:
    spec = build_scenario(category, INCIDENT_START, DURATION)
    injected = inject(baseline, spec)
    primary_effect = spec.effects[0]

    # Second half of the incident: every effect is past onset and ramp.
    window = slice(INCIDENT_START + DURATION / 2, INCIDENT_START + DURATION)
    healthy = _series(baseline, primary_effect.signal)[window]
    faulty = _series(injected, primary_effect.signal)[window]
    delta = faulty.mean() - healthy.mean()

    sigma = metric_spec(primary_effect.signal).noise_std
    expected_sign = 1.0 if primary_effect.shift_sigmas > 0 else -1.0
    assert delta * expected_sign > 2 * sigma, f"{category.value}: primary barely moved"


@pytest.mark.parametrize("category", NON_NORMAL, ids=lambda c: c.value)
def test_bounds_respected_for_every_scenario(
    baseline: pd.DataFrame, category: IncidentCategory
) -> None:
    injected = inject(baseline, build_scenario(category, INCIDENT_START, DURATION))
    for (stage_value, metric), group in injected.groupby(["stage", "metric"]):
        spec = metric_spec(SignalRef(Stage(stage_value), metric))
        assert group["value"].min() >= spec.low, f"{stage_value}.{metric}"
        assert group["value"].max() <= spec.high, f"{stage_value}.{metric}"


def test_ground_truth_alignment() -> None:
    spec = build_scenario(IncidentCategory.RETRIEVAL_DEGRADATION, INCIDENT_START, DURATION)
    truth = ground_truth_for(spec)
    assert truth.category is IncidentCategory.RETRIEVAL_DEGRADATION
    assert truth.root_cause_stage is Stage.RETRIEVAL
    assert truth.incident_end == INCIDENT_START + DURATION
    assert truth.injected_signals[0] == spec.scenario.primary_signal


def test_multi_factor_declares_both_fault_stages() -> None:
    spec = build_scenario(IncidentCategory.MULTI_FACTOR, INCIDENT_START, DURATION)
    assert Stage.RETRIEVAL in spec.scenario.affected_stages
    assert Stage.MODEL_INFERENCE in spec.scenario.affected_stages
