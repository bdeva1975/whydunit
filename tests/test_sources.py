"""Tests for the telemetry-source seam."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from whydunit.forensics import investigate
from whydunit.models import IncidentCategory
from whydunit.simulator import generate_dataset, to_wide
from whydunit.simulator.generator import write_dataset
from whydunit.sources import (
    CSVTelemetrySource,
    SyntheticTelemetrySource,
    TelemetrySource,
    validate_telemetry,
)

START = datetime(2026, 9, 1, tzinfo=UTC)


def test_implementations_satisfy_protocol() -> None:
    assert isinstance(SyntheticTelemetrySource(), TelemetrySource)
    assert isinstance(CSVTelemetrySource(path=Path("x.csv")), TelemetrySource)


def test_synthetic_source_matches_generator() -> None:
    source = SyntheticTelemetrySource(
        category=IncidentCategory.SCHEMA_DRIFT, start=START, days=0.5, seed=7
    )
    direct = generate_dataset(
        IncidentCategory.SCHEMA_DRIFT, start=START, days=0.5, seed=7
    ).telemetry
    pd.testing.assert_frame_equal(source.load(), direct)


def test_csv_roundtrip_preserves_investigation(tmp_path: Path) -> None:
    dataset = generate_dataset(
        IncidentCategory.RETRIEVAL_DEGRADATION, start=START, days=1.0, seed=42
    )
    telemetry_path, _, _ = write_dataset(dataset, tmp_path)

    loaded = CSVTelemetrySource(path=telemetry_path).load()
    original_result = investigate(to_wide(dataset.telemetry))
    loaded_result = investigate(to_wide(loaded))

    assert loaded_result.hypotheses[0].category is original_result.hypotheses[0].category
    assert len(loaded_result.anomalies) == len(original_result.anomalies)


def test_csv_source_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        CSVTelemetrySource(path=Path("does-not-exist.csv")).load()


def test_csv_source_rejects_naive_timestamps(tmp_path: Path) -> None:
    bad = tmp_path / "naive.csv"
    bad.write_text(
        "timestamp,stage,metric,value\n2026-09-01 00:00:00,ingestion,null_pct,0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        CSVTelemetrySource(path=bad).load()


def test_validate_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_telemetry(pd.DataFrame({"timestamp": [], "value": []}))
