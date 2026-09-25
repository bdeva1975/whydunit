"""Tests for the LLM explanation layer — stubbed client, no network, no key."""

from datetime import UTC, datetime, timedelta

import pytest

from whydunit.explain.narrative import build_prompt, evidence_registry, generate_narrative
from whydunit.models import (
    CaseFile,
    ConfidenceBand,
    Direction,
    Evidence,
    EvidenceKind,
    Hypothesis,
    IncidentCategory,
    Severity,
    SignalRef,
    Stage,
)
from whydunit.models.signal import Anomaly

START = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def _evidence(evidence_id: str, statement: str) -> Evidence:
    return Evidence(
        id=evidence_id,
        kind=EvidenceKind.OBSERVED,
        statement=statement,
        signals=(SignalRef(stage=Stage.RETRIEVAL, metric="top_k_similarity"),),
        weight=1.0,
    )


def _case(with_evidence: bool = True) -> CaseFile:
    shared = _evidence("EV-1", "Retrieval similarity dropped 38% below baseline")
    second = _evidence("EV-2", "Groundedness fell in the same window")
    contra = _evidence("EV-3", "Token throughput stayed flat")
    hypotheses = (
        Hypothesis(
            id="HYP-1",
            category=IncidentCategory.RETRIEVAL_DEGRADATION,
            statement="Retrieval quality degraded upstream of the LLM",
            affected_stages=(Stage.RETRIEVAL,),
            supporting=(shared, second) if with_evidence else (),
            contradicting=(contra,) if with_evidence else (),
            score=6.2,
            confidence=ConfidenceBand.STRONG,
            next_step="Inspect the vector index build",
        ),
        Hypothesis(
            id="HYP-2",
            category=IncidentCategory.MODEL_REGRESSION,
            statement="Model output quality regressed",
            affected_stages=(Stage.MODEL_INFERENCE,),
            supporting=(shared,) if with_evidence else (),
            score=2.0,
            confidence=ConfidenceBand.WEAK,
        ),
    )
    signal = SignalRef(stage=Stage.RETRIEVAL, metric="top_k_similarity")
    anomaly = Anomaly(
        signal=signal,
        start=START,
        end=START + timedelta(minutes=30),
        method="robust_zscore",
        score=6.4,
        direction=Direction.DOWN,
    )
    return CaseFile(
        case_id="CASE-TEST",
        incident_id="INC-TEST",
        pipeline_name="RAG Production Pipeline",
        window_start=START,
        window_end=START + timedelta(minutes=30),
        severity=Severity.HIGH,
        summary="Retrieval degradation propagated to answer quality.",
        symptoms=("Groundedness down 21%",),
        anomalies=(anomaly,),
        hypotheses=hypotheses,
    )


class _FakeBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **kwargs) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self.reply)


class _FakeClient:
    def __init__(self, reply: str = "## What happened\nRetrieval fell [EV-1].") -> None:
        self.messages = _FakeMessages(reply)


def test_registry_deduplicates_across_hypotheses() -> None:
    registry = evidence_registry(_case())
    assert sorted(registry) == ["EV-1", "EV-2", "EV-3"]  # EV-1 shared, kept once


def test_prompt_carries_facts_evidence_and_rules() -> None:
    prompt = build_prompt(_case())
    assert "[EV-1] (observed) Retrieval similarity dropped 38%" in prompt
    assert "retrieval.top_k_similarity" in prompt
    assert "supporting: EV-1, EV-2, contradicting: EV-3" in prompt
    assert START.isoformat() in prompt
    assert "ONLY IDs from the evidence registry above" in prompt
    assert "for\n   example [EV-1]" in prompt  # dynamic example uses a real ID
    assert "confidence: strong" in prompt


def test_generate_narrative_uses_injected_client() -> None:
    client = _FakeClient()
    text = generate_narrative(_case(), client=client)
    assert text.startswith("## What happened")
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-4-6"
    assert "Evidence registry" in call["messages"][0]["content"]


def test_generate_narrative_refuses_empty_evidence() -> None:
    with pytest.raises(ValueError, match="no evidence"):
        generate_narrative(_case(with_evidence=False), client=_FakeClient())


def test_generate_narrative_rejects_empty_reply() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        generate_narrative(_case(), client=_FakeClient(reply="   "))
