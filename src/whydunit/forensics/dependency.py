"""Pipeline dependency reasoning over the stage DAG.

Built from :data:`whydunit.simulator.pipeline.PIPELINE_EDGES` with
``networkx``. Nothing here assumes the chain is linear — a branching
pipeline only needs different edges.

Forensic roles of this module:

* **reachability** — ``is_upstream_of(a, b)``: could a fault at stage A
  plausibly propagate to stage B? Used to test whether one root cause
  can explain all affected stages;
* **origin ranking** — among the stages showing anomalies, which is the
  most upstream (by topological depth, tie-broken by first-anomaly
  time)? Faults propagate downstream, so the earliest/most-upstream
  anomalous stage is the natural origin candidate;
* **health rollup** — collapse signal-level anomalies to per-stage
  :class:`HealthState` for the pipeline view.

Health thresholds (documented, not hidden): a stage with no anomalies is
HEALTHY; max |z| below 7 → WARNING; below 12 → DEGRADED; 12 and above,
or three-plus distinct anomalous signals, → FAILED. The z-values are
robust scores from the detectors, so these bands read as "5–7σ is
notable, 7–12σ is serious, 12σ+ or broad damage is failure".
"""

from __future__ import annotations

from datetime import datetime

import networkx as nx

from whydunit.models import Anomaly, HealthState, Stage
from whydunit.simulator.pipeline import PIPELINE_EDGES

_WARNING_MAX = 7.0
_DEGRADED_MAX = 12.0
_FAILED_SIGNAL_COUNT = 3


def build_stage_graph() -> nx.DiGraph:
    """The pipeline dependency graph: edges point downstream."""
    graph = nx.DiGraph()
    graph.add_nodes_from(Stage)
    graph.add_edges_from(PIPELINE_EDGES)
    if not nx.is_directed_acyclic_graph(graph):  # defensive: edges are data
        raise ValueError("pipeline edges must form a DAG")
    return graph


_GRAPH = build_stage_graph()
_DEPTH: dict[Stage, int] = {stage: index for index, stage in enumerate(nx.topological_sort(_GRAPH))}


def upstream_of(stage: Stage) -> frozenset[Stage]:
    """All stages whose faults could propagate INTO ``stage``."""
    return frozenset(nx.ancestors(_GRAPH, stage))


def downstream_of(stage: Stage) -> frozenset[Stage]:
    """All stages a fault AT ``stage`` could propagate into."""
    return frozenset(nx.descendants(_GRAPH, stage))


def is_upstream_of(candidate: Stage, other: Stage) -> bool:
    """True if ``candidate`` is strictly upstream of ``other``."""
    return other in downstream_of(candidate)


def can_explain(origin: Stage, affected: set[Stage]) -> bool:
    """Could a single fault at ``origin`` explain every affected stage?

    True when every affected stage is the origin itself or downstream of
    it. False means at least one affected stage cannot have been caused
    by ``origin`` — evidence for multiple faults.
    """
    reachable = downstream_of(origin) | {origin}
    return affected <= reachable


def first_anomaly_by_stage(anomalies: list[Anomaly]) -> dict[Stage, datetime]:
    """Earliest anomaly start per stage."""
    firsts: dict[Stage, datetime] = {}
    for anomaly in anomalies:
        stage = anomaly.signal.stage
        if stage not in firsts or anomaly.start < firsts[stage]:
            firsts[stage] = anomaly.start
    return firsts


def rank_origin_candidates(anomalies: list[Anomaly]) -> list[Stage]:
    """Anomalous stages, most plausible origin first.

    Ordered by topological depth (most upstream first), tie-broken by
    first-anomaly time. Depth outranks time because propagation delays
    are minutes while sampling jitter is seconds — but this remains a
    heuristic and is treated as evidence, not proof, downstream.
    """
    firsts = first_anomaly_by_stage(anomalies)
    return sorted(firsts, key=lambda stage: (_DEPTH[stage], firsts[stage]))


def stage_health(anomalies: list[Anomaly]) -> dict[Stage, HealthState]:
    """Roll signal-level anomalies up to per-stage health."""
    peak: dict[Stage, float] = {}
    signals: dict[Stage, set[str]] = {}
    for anomaly in anomalies:
        stage = anomaly.signal.stage
        peak[stage] = max(peak.get(stage, 0.0), anomaly.score)
        signals.setdefault(stage, set()).add(anomaly.signal.metric)

    health: dict[Stage, HealthState] = {}
    for stage in Stage:
        if stage not in peak:
            health[stage] = HealthState.HEALTHY
        elif len(signals[stage]) >= _FAILED_SIGNAL_COUNT or peak[stage] >= _DEGRADED_MAX:
            health[stage] = HealthState.FAILED
        elif peak[stage] >= _WARNING_MAX:
            health[stage] = HealthState.DEGRADED
        else:
            health[stage] = HealthState.WARNING
    return health
