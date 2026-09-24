"""Tests for the dataset generation façade and CLI."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from whydunit.models import IncidentCategory
from whydunit.simulator.__main__ import main
from whydunit.simulator.generator import (
    GROUND_TRUTH_FILENAME,
    SCENARIO_FILENAME,
    TELEMETRY_FILENAME,
    generate_dataset,
    periods_for,
    write_dataset,
)

START = datetime(2026, 9, 1, tzinfo=UTC)


def test_periods_for_basic_math() -> None:
    assert periods_for(1.0, "1min") == 1440
    assert periods_for(0.5, "5min") == 144


def test_periods_for_rejects_tiny_windows() -> None:
    with pytest.raises(ValueError):
        periods_for(0.001, "1h")


def test_generate_dataset_is_deterministic() -> None:
    kwargs = dict(start=START, days=0.5, seed=7)
    first = generate_dataset(IncidentCategory.SCHEMA_DRIFT, **kwargs)
    second = generate_dataset(IncidentCategory.SCHEMA_DRIFT, **kwargs)
    pd.testing.assert_frame_equal(first.telemetry, second.telemetry)
    assert first.spec.scenario.id == second.spec.scenario.id


def test_incident_must_fit_inside_window() -> None:
    with pytest.raises(ValueError):
        generate_dataset(
            IncidentCategory.SCHEMA_DRIFT,
            start=START,
            days=0.25,
            incident_start_frac=0.9,
            incident_duration=timedelta(hours=4),
        )


def test_write_dataset_produces_three_files(tmp_path: Path) -> None:
    dataset = generate_dataset(IncidentCategory.RETRIEVAL_DEGRADATION, start=START, days=0.5)
    telemetry_path, scenario_path, truth_path = write_dataset(dataset, tmp_path / "out")

    assert telemetry_path.name == TELEMETRY_FILENAME
    assert scenario_path.name == SCENARIO_FILENAME
    assert truth_path.name == GROUND_TRUTH_FILENAME

    reloaded = pd.read_csv(telemetry_path)
    assert len(reloaded) == len(dataset.telemetry)
    assert list(reloaded.columns) == ["timestamp", "stage", "metric", "value"]

    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    assert scenario["category"] == "retrieval_degradation"
    assert truth["root_cause_stage"] == "retrieval"
    assert truth["injected_signals"][0] == "retrieval.top_k_similarity"
    # The investigator-facing file must not leak the answer key.
    assert "root_cause_stage" not in scenario
    assert "injected_signals" not in scenario


def test_normal_truth_has_no_root_cause(tmp_path: Path) -> None:
    dataset = generate_dataset(IncidentCategory.NORMAL, start=START, days=0.5)
    _, _, truth_path = write_dataset(dataset, tmp_path)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    assert truth["root_cause_stage"] is None
    assert truth["injected_signals"] == []


def test_cli_end_to_end(tmp_path: Path) -> None:
    code = main(
        [
            "--scenario",
            "llm_latency_spike",
            "--days",
            "0.5",
            "--seed",
            "11",
            "--output",
            str(tmp_path / "cli-out"),
        ]
    )
    assert code == 0
    for name in (TELEMETRY_FILENAME, SCENARIO_FILENAME, GROUND_TRUTH_FILENAME):
        assert (tmp_path / "cli-out" / name).exists()


def test_cli_rejects_unknown_scenario() -> None:
    with pytest.raises(SystemExit):
        main(["--scenario", "no_such_scenario"])
