"""Tests for the investigation façade — and the ground-truth firewall."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import whydunit.forensics as forensics_pkg
from whydunit.forensics import case_file, investigate
from whydunit.models import IncidentCategory, Severity
from whydunit.simulator import generate_dataset, to_wide
from whydunit.simulator.telemetry import generate_baseline

START = datetime(2026, 9, 1, tzinfo=UTC)
INCIDENT_START = START + timedelta(hours=12)


@pytest.fixture(scope="module")
def healthy_result():
    wide = to_wide(generate_baseline(START, periods=1440, seed=42))
    return investigate(wide)


@pytest.fixture(scope="module")
def retrieval_result():
    wide = to_wide(
        generate_dataset(
            IncidentCategory.RETRIEVAL_DEGRADATION, start=START, days=1.0, seed=42
        ).telemetry
    )
    return investigate(wide)


def test_forensics_never_mentions_ground_truth() -> None:
    package_dir = Path(forensics_pkg.__file__).parent
    offenders = [
        source.name
        for source in package_dir.glob("*.py")
        if "GroundTruth" in source.read_text(encoding="utf-8")
        or "ground_truth" in source.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"forensics modules reference ground truth: {offenders}"


def test_healthy_investigation(healthy_result) -> None:
    assert not healthy_result.is_incident
    assert healthy_result.severity is Severity.NONE
    assert healthy_result.hypotheses[0].category is IncidentCategory.NORMAL
    assert healthy_result.window_start == START.replace(tzinfo=UTC)
    assert healthy_result.symptoms == ()


def test_retrieval_investigation_diagnosis(retrieval_result) -> None:
    assert retrieval_result.is_incident
    top = retrieval_result.hypotheses[0]
    assert top.category is IncidentCategory.RETRIEVAL_DEGRADATION
    assert retrieval_result.severity in (Severity.HIGH, Severity.CRITICAL)


def test_retrieval_window_near_incident(retrieval_result) -> None:
    assert (
        INCIDENT_START - timedelta(minutes=15)
        <= retrieval_result.window_start
        <= INCIDENT_START + timedelta(minutes=30)
    )
    assert retrieval_result.window_end > retrieval_result.window_start


def test_symptoms_mention_primary_signal(retrieval_result) -> None:
    assert retrieval_result.symptoms
    joined = " ".join(retrieval_result.symptoms)
    assert "retrieval.top_k_similarity" in joined


def test_case_file_roundtrip(retrieval_result) -> None:
    case = case_file(retrieval_result, notes="Confirmed with search team.")
    payload = json.loads(case.to_json())
    assert payload["pipeline_name"] == "RAG Production Pipeline"
    assert payload["notes"] == "Confirmed with search team."
    markdown = case.to_markdown()
    assert "WHYDUNIT FORENSIC CASE FILE" in markdown
    assert "Root-cause candidates" in markdown
    assert "IST" in markdown


def test_case_file_recommended_actions(retrieval_result) -> None:
    case = case_file(retrieval_result)
    assert case.recommended_actions
    assert any("vector index" in action for action in case.recommended_actions)


def test_healthy_case_file_renders(healthy_result) -> None:
    case = case_file(healthy_result)
    assert "operating normally" in case.summary
    assert case.severity.value == "none"


def test_investigation_is_deterministic() -> None:
    wide = to_wide(
        generate_dataset(IncidentCategory.SCHEMA_DRIFT, start=START, days=1.0, seed=42).telemetry
    )
    first = investigate(wide)
    second = investigate(wide)
    assert first.hypotheses == second.hypotheses
    assert first.anomalies == second.anomalies


def test_wide_frame_required_shape() -> None:
    with pytest.raises((KeyError, ValueError, AttributeError)):
        investigate(pd.DataFrame({"not_a_signal": [1, 2, 3]}))
