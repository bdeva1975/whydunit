"""Signal identity and forensic finding records.

A *signal* is one metric on one pipeline stage (e.g. ``retrieval.top_k_similarity``).
Anomalies, change-points, and correlation clusters are the engine's raw
findings about signals — evidence inputs, not conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from whydunit.models.enums import Direction, Stage


@dataclass(frozen=True, slots=True)
class SignalRef:
    """Identifies one metric on one pipeline stage."""

    stage: Stage
    metric: str

    def __str__(self) -> str:
        return f"{self.stage.value}.{self.metric}"

    @classmethod
    def parse(cls, text: str) -> SignalRef:
        """Parse ``"stage.metric"`` back into a reference.

        Raises ``ValueError`` for unknown stages or malformed input.
        """
        stage_part, sep, metric_part = text.partition(".")
        if not sep or not metric_part:
            raise ValueError(f"not a signal reference: {text!r}")
        return cls(stage=Stage(stage_part), metric=metric_part)


@dataclass(frozen=True, slots=True)
class Anomaly:
    """A window where a signal deviated significantly from its baseline.

    ``score`` is method-specific (e.g. robust z-score); comparable within
    a method, not across methods.
    """

    signal: SignalRef
    start: datetime
    end: datetime
    method: str
    score: float
    direction: Direction


@dataclass(frozen=True, slots=True)
class ChangePoint:
    """A sustained level shift in a signal.

    ``relative_delta`` is ``(after_mean - before_mean) / max(|before_mean|, eps)``,
    so +0.5 reads as "the level rose 50%".
    """

    signal: SignalRef
    at: datetime
    before_mean: float
    after_mean: float
    relative_delta: float


@dataclass(frozen=True, slots=True)
class CorrelationCluster:
    """A group of signals that moved together inside one time window.

    ``ordering`` lists the signals by first-anomaly time (earliest first);
    temporal order is suggestive of propagation, never proof of causality.
    """

    signals: tuple[SignalRef, ...]
    start: datetime
    end: datetime
    strength: float
    ordering: tuple[SignalRef, ...]
