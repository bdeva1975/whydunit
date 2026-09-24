"""Tests for the anomaly detectors."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from whydunit.forensics.anomaly import (
    EWMA,
    ROBUST_ZSCORE,
    detect_anomalies,
    robust_zscores,
)
from whydunit.models import Direction, IncidentCategory, SignalRef, Stage
from whydunit.simulator import generate_dataset, to_wide
from whydunit.simulator.telemetry import generate_baseline

START = datetime(2026, 9, 1, tzinfo=UTC)
INCIDENT_START = START + timedelta(hours=12)  # generate_dataset default frac 0.5 of 1 day
INCIDENT_END = INCIDENT_START + timedelta(hours=2)


@pytest.fixture(scope="module")
def wide_healthy() -> pd.DataFrame:
    return to_wide(generate_baseline(START, periods=1440, seed=42))


def _wide_for(category: IncidentCategory) -> pd.DataFrame:
    return to_wide(generate_dataset(category, start=START, days=1.0, seed=42).telemetry)


@pytest.fixture(scope="module")
def wide_retrieval() -> pd.DataFrame:
    return _wide_for(IncidentCategory.RETRIEVAL_DEGRADATION)


@pytest.fixture(scope="module")
def wide_schema() -> pd.DataFrame:
    return _wide_for(IncidentCategory.SCHEMA_DRIFT)


def test_zscores_stay_small_on_healthy_data(wide_healthy: pd.DataFrame) -> None:
    z = robust_zscores(wide_healthy["retrieval.top_k_similarity"])
    assert z.abs().max() < 5.0


def test_no_anomalies_on_healthy_data(wide_healthy: pd.DataFrame) -> None:
    assert detect_anomalies(wide_healthy) == []


def test_retrieval_primary_detected_down(wide_retrieval: pd.DataFrame) -> None:
    ref = SignalRef(Stage.RETRIEVAL, "top_k_similarity")
    found = [
        a for a in detect_anomalies(wide_retrieval) if a.signal == ref and a.method == ROBUST_ZSCORE
    ]
    assert found, "primary signal not detected"
    first = found[0]
    assert first.direction is Direction.DOWN
    assert first.score >= 5.0
    assert INCIDENT_START <= first.start <= INCIDENT_START + timedelta(minutes=30)


def test_schema_violations_detected_up(wide_schema: pd.DataFrame) -> None:
    ref = SignalRef(Stage.INGESTION, "schema_violations")
    found = [
        a for a in detect_anomalies(wide_schema) if a.signal == ref and a.method == ROBUST_ZSCORE
    ]
    assert found
    assert found[0].direction is Direction.UP


def test_ewma_catches_slow_feature_drift() -> None:
    wide = _wide_for(IncidentCategory.FEATURE_DRIFT)
    ref = SignalRef(Stage.FEATURE_ENGINEERING, "feature_drift_score")
    found = [
        a
        for a in detect_anomalies(wide)
        if a.signal == ref and a.method == EWMA and a.start >= INCIDENT_START
    ]
    assert found, "EWMA detector missed the slow drift"


def test_incident_yields_few_merged_windows(wide_retrieval: pd.DataFrame) -> None:
    ref = SignalRef(Stage.RETRIEVAL, "top_k_similarity")
    found = [
        a for a in detect_anomalies(wide_retrieval) if a.signal == ref and a.method == ROBUST_ZSCORE
    ]
    assert 1 <= len(found) <= 3, f"expected merged windows, got {len(found)}"


def test_results_sorted_by_start(wide_schema: pd.DataFrame) -> None:
    anomalies = detect_anomalies(wide_schema)
    starts = [a.start for a in anomalies]
    assert starts == sorted(starts)
    assert all(a.end >= a.start for a in anomalies)
