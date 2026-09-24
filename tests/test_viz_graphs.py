"""Smoke tests for the pipeline view and evidence graph."""

from datetime import UTC, datetime

import plotly.graph_objects as go
import pytest

from whydunit.forensics import investigate, stage_health
from whydunit.models import IncidentCategory, Stage
from whydunit.simulator import generate_dataset, to_wide
from whydunit.viz import evidence_graph_figure, pipeline_figure

START = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def result():
    wide = to_wide(
        generate_dataset(
            IncidentCategory.RETRIEVAL_DEGRADATION, start=START, days=1.0, seed=42
        ).telemetry
    )
    return investigate(wide)


def test_pipeline_figure_all_healthy_builds() -> None:
    figure = pipeline_figure()
    assert isinstance(figure, go.Figure)
    node_trace = figure.data[-1]
    assert len(node_trace.x) == len(Stage)


def test_pipeline_figure_colours_track_health(result) -> None:
    health = stage_health(list(result.anomalies))
    figure = pipeline_figure(health)
    node_trace = figure.data[-1]
    colors = set(node_trace.marker.color)
    assert len(colors) > 1, "incident should produce mixed health colours"


def test_pipeline_layout_is_deterministic(result) -> None:
    health = stage_health(list(result.anomalies))
    first = pipeline_figure(health)
    second = pipeline_figure(health)
    assert first.data[-1].x == second.data[-1].x
    assert first.data[-1].y == second.data[-1].y


def test_evidence_graph_builds(result) -> None:
    figure = evidence_graph_figure(result)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) >= 3  # edges + at least signals/evidence/hypotheses


def test_evidence_graph_labels_confidence(result) -> None:
    figure = evidence_graph_figure(result)
    hypothesis_trace = figure.data[-1]
    assert any("retrieval_degradation" in text for text in hypothesis_trace.text)
    assert any("[strong]" in text for text in hypothesis_trace.text)


def test_evidence_graph_healthy_result_builds() -> None:
    wide = to_wide(
        generate_dataset(IncidentCategory.NORMAL, start=START, days=1.0, seed=42).telemetry
    )
    healthy = investigate(wide)
    figure = evidence_graph_figure(healthy)
    assert isinstance(figure, go.Figure)  # single NORMAL hypothesis, no signals
