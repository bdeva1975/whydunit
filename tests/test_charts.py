"""Smoke tests for chart builders — figures build headless without error."""

from datetime import UTC, datetime

import plotly.graph_objects as go
import pytest

from whydunit.forensics import investigate
from whydunit.models import IncidentCategory, SignalRef, Stage
from whydunit.simulator import generate_dataset, to_wide
from whydunit.viz.charts import compare_signal, correlation_heatmap, signal_timeline

START = datetime(2026, 9, 1, tzinfo=UTC)
PRIMARY = SignalRef(Stage.RETRIEVAL, "top_k_similarity")
DOWNSTREAM = SignalRef(Stage.EVALUATION, "groundedness")


@pytest.fixture(scope="module")
def wide():
    return to_wide(
        generate_dataset(
            IncidentCategory.RETRIEVAL_DEGRADATION, start=START, days=1.0, seed=42
        ).telemetry
    )


@pytest.fixture(scope="module")
def result(wide):
    return investigate(wide)


def test_timeline_builds_with_markers(wide, result) -> None:
    figure = signal_timeline(
        wide,
        [PRIMARY, DOWNSTREAM],
        anomalies=result.anomalies,
        changepoints=result.changepoints,
    )
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert figure.layout.shapes, "anomaly shading / changepoint lines missing"


def test_timeline_rejects_empty_signal_list(wide) -> None:
    with pytest.raises(ValueError):
        signal_timeline(wide, [])


def test_timeline_axis_names_display_tz(wide) -> None:
    figure = signal_timeline(wide, [PRIMARY], display_tz="Asia/Kolkata")
    axis_titles = [
        axis.title.text
        for axis in figure.select_xaxes()
        if axis.title is not None and axis.title.text
    ]
    assert any("Asia/Kolkata" in title for title in axis_titles)


def test_heatmap_builds(wide, result) -> None:
    cluster = result.clusters[0]
    figure = correlation_heatmap(wide, list(cluster.signals), start=cluster.start, end=cluster.end)
    assert isinstance(figure, go.Figure)
    assert figure.data[0].type == "heatmap"


def test_heatmap_rejects_single_signal(wide) -> None:
    with pytest.raises(ValueError):
        correlation_heatmap(wide, [PRIMARY])


def test_compare_signal_two_traces(wide) -> None:
    healthy = to_wide(
        generate_dataset(IncidentCategory.NORMAL, start=START, days=1.0, seed=42).telemetry
    )
    figure = compare_signal(wide, healthy, PRIMARY, labels=("incident", "healthy"))
    assert len(figure.data) == 2
    assert {trace.name for trace in figure.data} == {"incident", "healthy"}
