"""Evidence and root-cause hypothesis records.

The forensic engine's output is a ranked list of ``Hypothesis`` objects.
Each carries its supporting AND contradicting evidence — the engine never
hides inconvenient signals — and a qualitative ``ConfidenceBand`` rather
than a fake probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from whydunit.models.enums import ConfidenceBand, EvidenceKind, IncidentCategory, Stage
from whydunit.models.signal import SignalRef


@dataclass(frozen=True, slots=True)
class Evidence:
    """One statement about the telemetry, with its epistemic status.

    ``weight`` expresses how much this item moves a hypothesis score
    (positive numbers only; whether it supports or contradicts is decided
    by which list it sits in on the hypothesis).
    """

    id: str
    kind: EvidenceKind
    statement: str
    signals: tuple[SignalRef, ...]
    weight: float


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A ranked root-cause candidate.

    ``score`` is the raw signature-match score (comparable within one
    investigation only). ``confidence`` is the qualitative band derived
    from score and evidence coverage. ``next_step`` tells the engineer
    what to check to confirm or kill this hypothesis.
    """

    id: str
    category: IncidentCategory
    statement: str
    affected_stages: tuple[Stage, ...]
    supporting: tuple[Evidence, ...] = field(default=())
    contradicting: tuple[Evidence, ...] = field(default=())
    score: float = 0.0
    confidence: ConfidenceBand = ConfidenceBand.INSUFFICIENT
    next_step: str = ""
