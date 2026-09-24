"""Deterministic forensic engine: detectors, correlation, DAG, hypotheses."""

from whydunit.forensics.anomaly import DetectorConfig, detect_anomalies
from whydunit.forensics.changepoint import ChangePointConfig, detect_changepoints
from whydunit.forensics.correlation import CorrelationConfig, find_correlation_clusters
from whydunit.forensics.dependency import (
    build_stage_graph,
    can_explain,
    rank_origin_candidates,
    stage_health,
)
from whydunit.forensics.engine import InvestigationResult, case_file, investigate
from whydunit.forensics.hypotheses import SIGNATURES, generate_hypotheses
from whydunit.forensics.timeline import first_anomaly, incident_window, symptom_statements

__all__ = [
    "SIGNATURES",
    "ChangePointConfig",
    "CorrelationConfig",
    "DetectorConfig",
    "InvestigationResult",
    "build_stage_graph",
    "can_explain",
    "case_file",
    "detect_anomalies",
    "detect_changepoints",
    "find_correlation_clusters",
    "first_anomaly",
    "generate_hypotheses",
    "incident_window",
    "investigate",
    "rank_origin_candidates",
    "stage_health",
    "symptom_statements",
]
