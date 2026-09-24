"""Tests for dependency-graph reasoning."""

from datetime import UTC, datetime

import pytest

from whydunit.forensics.anomaly import detect_anomalies
from whydunit.forensics.dependency import (
    build_stage_graph,
    can_explain,
    downstream_of,
    is_upstream_of,
    rank_origin_candidates,
    stage_health,
    upstream_of,
)
from whydunit.models import HealthState, IncidentCategory, Stage
from whydunit.simulator import generate_dataset, to_wide

START = datetime(2026, 9, 1, tzinfo=UTC)


def _anomalies_for(category: IncidentCategory):
    wide = to_wide(generate_dataset(category, start=START, days=1.0, seed=42).telemetry)
    return detect_anomalies(wide)


@pytest.fixture(scope="module")
def schema_anomalies():
    return _anomalies_for(IncidentCategory.SCHEMA_DRIFT)


def test_graph_is_dag_over_all_stages() -> None:
    graph = build_stage_graph()
    assert set(graph.nodes) == set(Stage)
    assert graph.number_of_edges() == len(Stage) - 1


def test_upstream_downstream_relations() -> None:
    assert Stage.INGESTION in upstream_of(Stage.RETRIEVAL)
    assert Stage.EVALUATION in downstream_of(Stage.RETRIEVAL)
    assert is_upstream_of(Stage.INGESTION, Stage.DELIVERY)
    assert not is_upstream_of(Stage.DELIVERY, Stage.INGESTION)
    assert not is_upstream_of(Stage.RETRIEVAL, Stage.RETRIEVAL)


def test_can_explain_respects_reachability() -> None:
    assert can_explain(Stage.INGESTION, {Stage.VALIDATION, Stage.EVALUATION})
    assert can_explain(Stage.RETRIEVAL, {Stage.RETRIEVAL, Stage.EVALUATION})
    assert not can_explain(Stage.RETRIEVAL, {Stage.INGESTION, Stage.EVALUATION})


def test_origin_ranking_puts_ingestion_first(schema_anomalies) -> None:
    ranked = rank_origin_candidates(schema_anomalies)
    assert ranked, "no anomalous stages found"
    assert ranked[0] is Stage.INGESTION


def test_origin_ranking_retrieval_scenario() -> None:
    ranked = rank_origin_candidates(_anomalies_for(IncidentCategory.RETRIEVAL_DEGRADATION))
    assert ranked[0] is Stage.RETRIEVAL


def test_stage_health_rollup(schema_anomalies) -> None:
    health = stage_health(schema_anomalies)
    assert health[Stage.INGESTION] in (HealthState.DEGRADED, HealthState.FAILED)
    assert health[Stage.DELIVERY] is HealthState.HEALTHY
    assert set(health) == set(Stage)


def test_stage_health_empty_input_is_all_healthy() -> None:
    health = stage_health([])
    assert all(state is HealthState.HEALTHY for state in health.values())
