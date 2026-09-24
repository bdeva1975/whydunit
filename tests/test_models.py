"""Tests for the core model layer."""

from datetime import datetime, timedelta, timezone

import json
import pytest

from whydunit.models.casefile import CaseFile
from whydunit.models.enums import (
    ConfidenceBand,
    Direction,
    EvidenceKind,
    IncidentCategory,
    Severity,
    Stage,
)
from whydunit.models.hypothesis import Evidence, Hypothesis
from whydunit.models.signal import Anomaly, CorrelationCluster, SignalRef

UTC = timezone.utc


def test_stage_declaration_order_is_topological() -> None:
    stages = list(Stage)
    assert stages[0] is Stage.INGESTION
    assert stages[-1] is Stage.DELIVERY
    assert stages.index(Stage.RETRIEVAL) < stages.index(Stage.MODEL_INFERENCE)


def test_signal_ref_roundtrip() -> None:
    ref = SignalRef(stage=Stage.RETRIEVAL, metric="top_k_similarity")
    assert str(ref) == "retrieval.top_k_similarity"
    assert SignalRef.parse(str(ref)) == ref


@pytest.mark.parametrize("bad", ["", "retrieval", "no_such_stage.metric", "."])
def test_signal_ref_parse_rejects_bad_input(bad: str) -> None:
    with pytest.raises(ValueError):
        SignalRef.parse(bad)


def _sample_casefile() -> CaseFile:
    start = datetime(2026, 9, 21, 5, 12, tzinfo=UTC)
    end = start + timedelta(minutes=36)
    signal = SignalRef(stage=Stage.RETRIEVAL, metric="top_k_similarity")
    anomaly = Anomaly(
        signal=signal,
        start=start,
        end=end,
        method="robust_zscore",
        score=6.4,
        direction=Direction.DOWN,
    )
    cluster = CorrelationCluster(
        signals=(signal,),
        start=start,
        end=end,
        strength=0.91,
        ordering=(signal,),
    )
    evidence = Evidence(
        id="EV-1",
        kind=EvidenceKind.OBSERVED,
        statement="Retrieval similarity dropped 38% below baseline",
        signals=(signal,),
        weight=1.0,
    )
    hypothesis = Hypothesis(
        id="HYP-1",
        category=IncidentCategory.RETRIEVAL_DEGRADATION,
        statement="Retrieval quality degraded upstream of the LLM",
        affected_stages=(Stage.RETRIEVAL, Stage.MODEL_INFERENCE),
        supporting=(evidence,),
        contradicting=(),
        score=3.2,
        confidence=ConfidenceBand.STRONG,
        next_step="Inspect the vector index build that shipped before the window",
    )
    return CaseFile(
        case_id="CASE-1",
        incident_id="INC-20260921-0042",
        pipeline_name="RAG Production Pipeline",
        window_start=start,
        window_end=end,
        severity=Severity.HIGH,
        summary="Retrieval degradation propagated to answer quality.",
        symptoms=("Groundedness down 21%",),
        anomalies=(anomaly,),
        correlations=(cluster,),
        hypotheses=(hypothesis,),
        notes="Confirmed with the search team.",
        recommended_actions=("Roll back the index build",),
        exported_at=end,
    )


def test_casefile_json_roundtrips_through_stdlib() -> None:
    payload = json.loads(_sample_casefile().to_json())
    assert payload["severity"] == "high"
    assert payload["hypotheses"][0]["confidence"] == "strong"
    assert payload["anomalies"][0]["signal"] == {
        "stage": "retrieval",
        "metric": "top_k_similarity",
    }


def test_casefile_markdown_renders_ist_and_sections() -> None:
    text = _sample_casefile().to_markdown()
    assert text.startswith("# WHYDUNIT FORENSIC CASE FILE")
    assert "IST" in text  # Asia/Kolkata default display timezone
    assert "2026-09-21 10:42 IST" in text  # 05:12 UTC + 05:30
    assert "## Root-cause candidates" in text
    assert "not proof of causality" in text