"""Tests for change-point detection."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from whydunit.forensics.changepoint import (
    ChangePointConfig,
    detect_changepoints,
    shift_statistic,
)
from whydunit.models import IncidentCategory, SignalRef, Stage
from whydunit.simulator import generate_dataset, to_wide
from whydunit.simulator.telemetry import generate_baseline

START = datetime(2026, 9, 1, tzinfo=UTC)
INCIDENT_START = START + timedelta(hours=12)


@pytest.fixture(scope="module")
def wide_healthy() -> pd.DataFrame:
    return to_wide(generate_baseline(START, periods=1440, seed=42))


@pytest.fixture(scope="module")
def wide_retrieval() -> pd.DataFrame:
    telemetry = generate_dataset(
        IncidentCategory.RETRIEVAL_DEGRADATION, start=START, days=1.0, seed=42
    ).telemetry
    return to_wide(telemetry)


def test_statistic_flat_series_is_quiet() -> None:
    rng = np.random.default_rng(0)
    index = pd.date_range(START, periods=600, freq="1min", tz="UTC")
    series = pd.Series(rng.normal(10.0, 1.0, size=600), index=index)
    stat = shift_statistic(series, window=60)
    assert np.nanmax(stat.to_numpy()) < 4.0


def test_statistic_peaks_at_synthetic_step() -> None:
    rng = np.random.default_rng(1)
    index = pd.date_range(START, periods=600, freq="1min", tz="UTC")
    values = rng.normal(10.0, 1.0, size=600)
    values[300:] += 8.0
    stat = shift_statistic(pd.Series(values, index=index), window=60)
    peak_pos = int(np.nanargmax(stat.to_numpy()))
    assert abs(peak_pos - 300) <= 5
    assert float(np.nanmax(stat.to_numpy())) >= 4.0


def test_no_changepoints_on_healthy_data(wide_healthy: pd.DataFrame) -> None:
    assert detect_changepoints(wide_healthy) == []


def test_retrieval_changepoint_near_incident_start(wide_retrieval: pd.DataFrame) -> None:
    ref = SignalRef(Stage.RETRIEVAL, "top_k_similarity")
    found = [cp for cp in detect_changepoints(wide_retrieval) if cp.signal == ref]
    assert found, "no change point on the primary signal"
    first = found[0]
    assert (
        INCIDENT_START - timedelta(minutes=10) <= first.at <= INCIDENT_START + timedelta(minutes=45)
    )
    assert first.relative_delta < 0  # similarity fell


def test_min_separation_dedupes(wide_retrieval: pd.DataFrame) -> None:
    ref = SignalRef(Stage.RETRIEVAL, "top_k_similarity")
    found = [cp for cp in detect_changepoints(wide_retrieval) if cp.signal == ref]
    assert len(found) <= 3


def test_short_series_returns_nan_statistic() -> None:
    index = pd.date_range(START, periods=50, freq="1min", tz="UTC")
    series = pd.Series(np.ones(50), index=index)
    stat = shift_statistic(series, window=60)
    assert stat.isna().all()


def test_results_sorted(wide_retrieval: pd.DataFrame) -> None:
    found = detect_changepoints(wide_retrieval)
    ats = [cp.at for cp in found]
    assert ats == sorted(ats)


def test_config_threshold_respected(wide_retrieval: pd.DataFrame) -> None:
    strict = ChangePointConfig(threshold=1e9)
    assert detect_changepoints(wide_retrieval, strict) == []
