"""Timeline facts: when the incident window was, what the symptoms were.

Pure functions from detector findings to human-readable facts. The
symptom statements pair each signal's peak anomaly with its change-point
delta when one exists ("fell 41% below the prior level") — numbers an
engineer can quote in an incident channel.
"""

from __future__ import annotations

from datetime import datetime

from whydunit.models import Anomaly, ChangePoint, SignalRef


def incident_window(anomalies: list[Anomaly]) -> tuple[datetime, datetime] | None:
    """[earliest anomaly start, latest anomaly end], or None if quiet."""
    if not anomalies:
        return None
    return (
        min(anomaly.start for anomaly in anomalies),
        max(anomaly.end for anomaly in anomalies),
    )


def first_anomaly(anomalies: list[Anomaly]) -> Anomaly | None:
    """The earliest anomaly — where the forensic trail begins."""
    return min(anomalies, key=lambda anomaly: anomaly.start) if anomalies else None


def _peak_by_signal(anomalies: list[Anomaly]) -> dict[SignalRef, Anomaly]:
    peaks: dict[SignalRef, Anomaly] = {}
    for anomaly in anomalies:
        current = peaks.get(anomaly.signal)
        if current is None or anomaly.score > current.score:
            peaks[anomaly.signal] = anomaly
    return peaks


def symptom_statements(
    anomalies: list[Anomaly],
    changepoints: list[ChangePoint],
    limit: int = 8,
) -> tuple[str, ...]:
    """Top symptoms by peak anomaly score, as quotable sentences."""
    deltas: dict[SignalRef, float] = {}
    for cp in changepoints:
        if cp.signal not in deltas or abs(cp.relative_delta) > abs(deltas[cp.signal]):
            deltas[cp.signal] = cp.relative_delta

    peaks = sorted(_peak_by_signal(anomalies).values(), key=lambda a: a.score, reverse=True)
    statements: list[str] = []
    for anomaly in peaks[:limit]:
        line = f"`{anomaly.signal}` moved {anomaly.direction.value} (peak {anomaly.score:.1f}σ)"
        delta = deltas.get(anomaly.signal)
        if delta is not None:
            direction = "above" if delta > 0 else "below"
            line += f", level shifted {abs(delta):.0%} {direction} the prior hour"
        statements.append(line)
    return tuple(statements)
