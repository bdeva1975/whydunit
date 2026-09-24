"""Tests for the static pipeline definition."""

import pytest

from whydunit.models import SignalRef, Stage
from whydunit.simulator.pipeline import (
    PIPELINE_EDGES,
    STAGE_METRICS,
    all_signals,
    metric_spec,
)


def test_every_stage_declares_metrics() -> None:
    for stage in Stage:
        assert len(STAGE_METRICS[stage]) >= 2, f"{stage.value} has too few metrics"


def test_baselines_sit_inside_bounds() -> None:
    for specs in STAGE_METRICS.values():
        for spec in specs:
            assert spec.low <= spec.baseline <= spec.high, spec.name
            assert spec.noise_std > 0, spec.name


def test_edges_form_a_chain_over_all_stages() -> None:
    stages = list(Stage)
    assert len(PIPELINE_EDGES) == len(stages) - 1
    for (upstream, downstream), expected_up, expected_down in zip(
        PIPELINE_EDGES, stages[:-1], stages[1:], strict=True
    ):
        assert upstream is expected_up
        assert downstream is expected_down


def test_signal_catalogue_and_lookup_roundtrip() -> None:
    signals = all_signals()
    assert len(signals) == sum(len(specs) for specs in STAGE_METRICS.values())
    ref = SignalRef(stage=Stage.RETRIEVAL, metric="top_k_similarity")
    assert ref in signals
    assert metric_spec(ref).concern is not None


def test_metric_spec_rejects_unknown_metric() -> None:
    with pytest.raises(KeyError):
        metric_spec(SignalRef(stage=Stage.RETRIEVAL, metric="no_such_metric"))
