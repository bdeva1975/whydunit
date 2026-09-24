"""Tests for baseline telemetry generation."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from whydunit.simulator.pipeline import STAGE_METRICS, all_signals, metric_spec
from whydunit.simulator.telemetry import generate_baseline, make_time_index, to_wide

START = datetime(2026, 9, 1, tzinfo=UTC)


def test_naive_start_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_time_index(datetime(2026, 9, 1), periods=10)


def test_time_index_is_utc() -> None:
    index = make_time_index(START, periods=5, freq="1min")
    assert str(index.tz) == "UTC"
    assert len(index) == 5


def test_generation_is_deterministic() -> None:
    first = generate_baseline(START, periods=120, seed=42)
    second = generate_baseline(START, periods=120, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_differ() -> None:
    first = generate_baseline(START, periods=120, seed=42)
    second = generate_baseline(START, periods=120, seed=43)
    assert not first["value"].equals(second["value"])


def test_shape_and_columns() -> None:
    periods = 60
    telemetry = generate_baseline(START, periods=periods, seed=1)
    assert list(telemetry.columns) == ["timestamp", "stage", "metric", "value"]
    assert len(telemetry) == periods * len(all_signals())
    assert telemetry["timestamp"].dt.tz is not None


def test_values_respect_bounds() -> None:
    telemetry = generate_baseline(START, periods=240, seed=7)
    for (stage_value, metric), group in telemetry.groupby(["stage", "metric"]):
        specs = {spec.name: spec for specs in STAGE_METRICS.values() for spec in specs}
        spec = specs[metric]
        assert group["value"].min() >= spec.low, f"{stage_value}.{metric}"
        assert group["value"].max() <= spec.high, f"{stage_value}.{metric}"


def test_values_hover_near_baseline() -> None:
    telemetry = generate_baseline(START, periods=1440, seed=5)  # one day
    for signal in all_signals():
        spec = metric_spec(signal)
        series = telemetry[
            (telemetry["stage"] == signal.stage.value) & (telemetry["metric"] == signal.metric)
        ]["value"]
        # Mean within 3 noise_std of baseline: loose enough for seasonality,
        # tight enough to catch unit mistakes.
        assert abs(series.mean() - spec.baseline) < 3 * spec.noise_std, str(signal)


def test_wide_pivot_shape() -> None:
    periods = 30
    telemetry = generate_baseline(START, periods=periods, seed=2)
    wide = to_wide(telemetry)
    assert wide.shape == (periods, len(all_signals()))
    assert "retrieval.top_k_similarity" in wide.columns
