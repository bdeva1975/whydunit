"""The forensic engine façade: one call from telemetry to case file.

``investigate(wide)`` runs the full chain — anomaly detection,
change-point detection, correlation clustering, stage-health rollup,
hypothesis ranking — and returns everything as one
:class:`InvestigationResult`. ``case_file()`` renders that result as an
exportable :class:`CaseFile`.

Severity here is *derived from evidence* (per-stage health), not copied
from the scenario: the investigator sees what the telemetry supports.
Heuristic, documented: two or more FAILED stages → CRITICAL; one FAILED
→ HIGH; any DEGRADED → MEDIUM; any WARNING → LOW; all healthy → NONE.

This module never sees ground truth. A test enforces that no module in
``whydunit.forensics`` even mentions it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from whydunit.forensics.anomaly import DetectorConfig, detect_anomalies
from whydunit.forensics.changepoint import ChangePointConfig, detect_changepoints
from whydunit.forensics.correlation import CorrelationConfig, find_correlation_clusters
from whydunit.forensics.dependency import rank_origin_candidates, stage_health
from whydunit.forensics.hypotheses import generate_hypotheses
from whydunit.forensics.timeline import first_anomaly, incident_window, symptom_statements
from whydunit.models import (
    Anomaly,
    CaseFile,
    ChangePoint,
    CorrelationCluster,
    HealthState,
    Hypothesis,
    Severity,
    Stage,
)

DEFAULT_PIPELINE_NAME = "RAG Production Pipeline"


@dataclass(frozen=True, slots=True)
class InvestigationResult:
    """Everything one investigation produced, for UI and export."""

    pipeline_name: str
    window_start: datetime
    window_end: datetime
    anomalies: tuple[Anomaly, ...]
    changepoints: tuple[ChangePoint, ...]
    clusters: tuple[CorrelationCluster, ...]
    health: dict[Stage, HealthState]
    origin_ranking: tuple[Stage, ...]
    hypotheses: tuple[Hypothesis, ...]
    severity: Severity
    symptoms: tuple[str, ...] = field(default=())

    @property
    def is_incident(self) -> bool:
        return bool(self.anomalies)


def _derive_severity(health: dict[Stage, HealthState]) -> Severity:
    failed = sum(1 for state in health.values() if state is HealthState.FAILED)
    degraded = sum(1 for state in health.values() if state is HealthState.DEGRADED)
    warning = sum(1 for state in health.values() if state is HealthState.WARNING)
    if failed >= 2:
        return Severity.CRITICAL
    if failed == 1:
        return Severity.HIGH
    if degraded >= 1:
        return Severity.MEDIUM
    if warning >= 1:
        return Severity.LOW
    return Severity.NONE


def investigate(
    wide: pd.DataFrame,
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
    detector_config: DetectorConfig | None = None,
    changepoint_config: ChangePointConfig | None = None,
    correlation_config: CorrelationConfig | None = None,
) -> InvestigationResult:
    """Run the full forensic chain over wide telemetry."""
    anomalies = detect_anomalies(wide, detector_config)
    changepoints = detect_changepoints(wide, changepoint_config)
    clusters = find_correlation_clusters(wide, anomalies, correlation_config)
    health = stage_health(anomalies)
    hypotheses = generate_hypotheses(anomalies)

    window = incident_window(anomalies)
    if window is None:
        window = (
            wide.index[0].to_pydatetime(),
            wide.index[-1].to_pydatetime(),
        )

    return InvestigationResult(
        pipeline_name=pipeline_name,
        window_start=window[0],
        window_end=window[1],
        anomalies=tuple(anomalies),
        changepoints=tuple(changepoints),
        clusters=tuple(clusters),
        health=health,
        origin_ranking=tuple(rank_origin_candidates(anomalies)),
        hypotheses=tuple(hypotheses),
        severity=_derive_severity(health),
        symptoms=symptom_statements(anomalies, changepoints),
    )


def _summary(result: InvestigationResult) -> str:
    if not result.is_incident:
        return (
            f"No anomalies detected on {result.pipeline_name}; the pipeline is operating normally."
        )
    top = result.hypotheses[0]
    affected = sorted({anomaly.signal.stage.value for anomaly in result.anomalies})
    first = first_anomaly(list(result.anomalies))
    assert first is not None  # is_incident guarantees it
    return (
        f"{len(result.anomalies)} anomaly windows across {len(affected)} stages "
        f"({', '.join(affected)}), beginning with `{first.signal}` at "
        f"{first.start.isoformat()}. Leading hypothesis: {top.statement} "
        f"[{top.confidence.value} evidence]."
    )


def case_file(
    result: InvestigationResult,
    notes: str = "",
    case_id: str | None = None,
    incident_id: str | None = None,
) -> CaseFile:
    """Render an investigation as an exportable case file."""
    stamp = f"{result.window_start:%Y%m%d-%H%M}"
    actions = tuple(
        f"[{h.confidence.value}] {h.next_step}" for h in result.hypotheses[:3] if h.next_step
    )
    return CaseFile(
        case_id=case_id or f"CASE-{stamp}",
        incident_id=incident_id or f"INC-{stamp}",
        pipeline_name=result.pipeline_name,
        window_start=result.window_start,
        window_end=result.window_end,
        severity=result.severity,
        summary=_summary(result),
        symptoms=result.symptoms,
        anomalies=result.anomalies,
        correlations=result.clusters,
        hypotheses=result.hypotheses,
        notes=notes,
        recommended_actions=actions,
    )
