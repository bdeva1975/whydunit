"""Tests for cross-signal correlation clustering."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from whydunit.forensics.anomaly import detect_anomalies
from whydunit.forensics.correlation import CorrelationConfig, find_correlation_clusters
from whydunit.models import IncidentCategory, SignalRef, Stage
from whydunit.simulator import generate_dataset, to_wide
from whydunit.simulator.telemetry import generate_baseline

START = datetime(2026, 9, 1, tzinfo=UTC)


def _wide_for(category: IncidentCategory) -> pd.DataFrame:
    return to_wide(generate_dataset(category, start=START, days=1.0, seed=42).telemetry)


@pytest.fixture(scope="module")
def wide_retrieval() -> pd.DataFrame:
    return _wide_for(IncidentCategory.RETRIEVAL_DEGRADATION)


@pytest.fixture(scope="module")
def retrieval_clusters(wide_retrieval: pd.DataFrame):
    return find_correlation_clusters(wide_retrieval, detect_anomalies(wide_retrieval))


def test_healthy_data_yields_no_clusters() -> None:
    wide = to_wide(generate_baseline(START, periods=1440, seed=42))
    assert find_correlation_clusters(wide, detect_anomalies(wide)) == []


def test_retrieval_incident_forms_one_cluster(retrieval_clusters) -> None:
    assert len(retrieval_clusters) == 1, f"expected one cluster, got {len(retrieval_clusters)}"


def test_cluster_contains_primary_and_downstream(retrieval_clusters) -> None:
    cluster = retrieval_clusters[0]
    signals = set(cluster.signals)
    assert SignalRef(Stage.RETRIEVAL, "top_k_similarity") in signals
    assert SignalRef(Stage.EVALUATION, "groundedness") in signals


def test_ordering_puts_primary_before_downstream(retrieval_clusters) -> None:
    ordering = list(retrieval_clusters[0].ordering)
    primary = SignalRef(Stage.RETRIEVAL, "top_k_similarity")
    downstream = SignalRef(Stage.EVALUATION, "hallucination_rate")
    assert primary in ordering and downstream in ordering
    assert ordering.index(primary) < ordering.index(downstream)


def test_cluster_strength_is_meaningful(retrieval_clusters) -> None:
    assert retrieval_clusters[0].strength >= 0.5


def test_multi_factor_merges_concurrent_faults() -> None:
    wide = _wide_for(IncidentCategory.MULTI_FACTOR)
    clusters = find_correlation_clusters(wide, detect_anomalies(wide))
    assert len(clusters) == 1
    signals = set(clusters[0].signals)
    assert SignalRef(Stage.RETRIEVAL, "top_k_similarity") in signals
    assert SignalRef(Stage.MODEL_INFERENCE, "memory_utilization_pct") in signals


def test_min_signals_respected(wide_retrieval: pd.DataFrame) -> None:
    config = CorrelationConfig(min_signals=100)
    assert find_correlation_clusters(wide_retrieval, detect_anomalies(wide_retrieval)) != []
    assert find_correlation_clusters(wide_retrieval, detect_anomalies(wide_retrieval), config) == []


def test_clusters_sorted_by_start(retrieval_clusters) -> None:
    starts = [cluster.start for cluster in retrieval_clusters]
    assert starts == sorted(starts)
