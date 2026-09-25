"""Tests for narrative validation — the 'zero authority' enforcement."""

from datetime import UTC, datetime, timedelta

from whydunit.explain.validator import validate_narrative
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


def _case() -> CaseFile:
    signal = SignalRef(stage=Stage.RETRIEVAL, metric="top_k_similarity")
    evidence = tuple(
        Evidence(
            id=f"EV-{index}",
            kind=EvidenceKind.OBSERVED,
            statement=f"Fact number {index}",
            signals=(signal,),
            weight=1.0,
        )
        for index in (1, 2)
    )
    hypothesis = Hypothesis(
        id="HYP-1",
        category=IncidentCategory.RETRIEVAL_DEGRADATION,
        statement="Retrieval quality degraded",
        affected_stages=(Stage.RETRIEVAL,),
        supporting=evidence,
        score=5.0,
        confidence=ConfidenceBand.STRONG,
    )
    return CaseFile(
        case_id="CASE-V",
        incident_id="INC-V",
        pipeline_name="RAG Production Pipeline",
        window_start=START,
        window_end=START + timedelta(minutes=30),
        severity=Severity.HIGH,
        summary="Test case.",
        hypotheses=(hypothesis,),
    )


GOOD = """## What happened

Retrieval similarity collapsed during the window [EV-1].

## Why the engine believes this

Groundedness fell in lockstep [EV-1][EV-2].

## What to do next

- Inspect the vector index build that shipped before the window.
"""


def test_valid_narrative_passes() -> None:
    result = validate_narrative(GOOD, _case())
    assert result.valid
    assert result.cited_ids == ("EV-1", "EV-2")
    assert result.problems == ()


def test_unknown_id_is_rejected() -> None:
    bad = GOOD.replace("[EV-2]", "[EV-99]")
    result = validate_narrative(bad, _case())
    assert not result.valid
    assert result.unknown_ids == ("EV-99",)
    assert any("EV-99" in problem for problem in result.problems)


def test_uncited_paragraph_is_rejected() -> None:
    bad = GOOD.replace(
        "Retrieval similarity collapsed during the window [EV-1].",
        "Retrieval similarity collapsed because of a network partition.",
    )
    result = validate_narrative(bad, _case())
    assert not result.valid
    assert result.uncited_paragraphs
    assert "network partition" in result.uncited_paragraphs[0]


def test_next_steps_section_is_exempt() -> None:
    result = validate_narrative(GOOD, _case())
    assert all("Inspect the vector" not in item for item in result.uncited_paragraphs)


def test_headings_alone_are_not_claims() -> None:
    result = validate_narrative("## What happened\n\nAll fell down [EV-1].", _case())
    assert result.valid


def test_zero_citations_is_invalid() -> None:
    result = validate_narrative("## What happened\n\nEverything broke badly.", _case())
    assert not result.valid
    assert "cites no evidence at all" in result.problems[0]
