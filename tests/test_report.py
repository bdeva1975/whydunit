"""Tests for narrate(): the generate-validate-retry loop."""

from datetime import UTC, datetime, timedelta

import pytest

from whydunit.explain import NarrativeValidationError, narrate
from whydunit.models import (
    CaseFile,
    ConfidenceBand,
    Evidence,
    EvidenceKind,
    Hypothesis,
    IncidentCategory,
    Severity,
    SignalRef,
    Stage,
)

START = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)

GOOD = "## What happened\n\nRetrieval fell hard [EV-1]."
BAD_UNKNOWN = "## What happened\n\nRetrieval fell hard [EV-99]."
BAD_UNCITED = "## What happened\n\nA network partition caused everything."


def _case() -> CaseFile:
    signal = SignalRef(stage=Stage.RETRIEVAL, metric="top_k_similarity")
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
        statement="Retrieval quality degraded",
        affected_stages=(Stage.RETRIEVAL,),
        supporting=(evidence,),
        score=5.0,
        confidence=ConfidenceBand.STRONG,
    )
    return CaseFile(
        case_id="CASE-R",
        incident_id="INC-R",
        pipeline_name="RAG Production Pipeline",
        window_start=START,
        window_end=START + timedelta(minutes=30),
        severity=Severity.HIGH,
        summary="Test case.",
        hypotheses=(hypothesis,),
    )


class _FakeBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _SeqMessages:
    """Returns queued replies in order; records every request."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def create(self, **kwargs) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self.replies.pop(0))


class _SeqClient:
    def __init__(self, replies: list[str]) -> None:
        self.messages = _SeqMessages(replies)


def test_valid_first_attempt_no_retry() -> None:
    client = _SeqClient([GOOD])
    report = narrate(_case(), client=client)
    assert report.attempts == 1
    assert report.text == GOOD
    assert report.validation.valid
    assert len(client.messages.calls) == 1


def test_invalid_then_valid_retries_with_feedback() -> None:
    client = _SeqClient([BAD_UNKNOWN, GOOD])
    report = narrate(_case(), client=client)
    assert report.attempts == 2
    assert report.text == GOOD
    retry_prompt = client.messages.calls[1]["messages"][0]["content"]
    assert "Correction required" in retry_prompt
    assert "EV-99" in retry_prompt
    assert BAD_UNKNOWN.splitlines()[-1] in retry_prompt  # previous attempt included


def test_still_invalid_after_retry_raises() -> None:
    client = _SeqClient([BAD_UNKNOWN, BAD_UNCITED])
    with pytest.raises(NarrativeValidationError) as excinfo:
        narrate(_case(), client=client)
    assert excinfo.value.attempts == 2
    assert any("uncited claim" in problem for problem in excinfo.value.problems)
    assert len(client.messages.calls) == 2


def test_single_attempt_mode() -> None:
    client = _SeqClient([BAD_UNKNOWN])
    with pytest.raises(NarrativeValidationError) as excinfo:
        narrate(_case(), client=client, max_attempts=1)
    assert excinfo.value.attempts == 1
    assert len(client.messages.calls) == 1
