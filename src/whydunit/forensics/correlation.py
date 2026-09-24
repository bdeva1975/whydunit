"""Cross-signal correlation: which signals moved together, and who moved first.

Two-step method, deliberately transparent:

1. **Time clustering** — each signal's anomalies are collapsed to one
   [first-start, last-end] span; spans that overlap (within a tolerance
   gap) are merged into clusters via interval sweep. Signals that were
   anomalous in the same window *might* be related.
2. **Correlation scoring** — over the cluster's union window, compute
   pairwise Pearson correlation of the raw signals; the cluster's
   ``strength`` is the mean absolute off-diagonal correlation. High
   strength means the signals genuinely co-moved, not merely co-occurred.

``ordering`` ranks the cluster's signals by first-anomaly time. Temporal
precedence is *suggestive* of propagation direction and is treated as
evidence downstream — never as proof of causality.

Pearson on raw values is enough here because incident effects are
sustained level shifts: within an incident window, co-shifted signals
correlate strongly. Lagged cross-correlation would add little at these
magnitudes and would cost explainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from whydunit.models import Anomaly, CorrelationCluster, SignalRef


@dataclass(frozen=True, slots=True)
class CorrelationConfig:
    """Knobs, sized for 1-minute telemetry.

    * ``merge_gap`` 15 min → anomaly spans separated by less than this
      are treated as one time cluster (incident effects have staggered
      onsets but overlap heavily);
    * ``pad`` 10 min → the correlation window extends slightly beyond
      the anomaly union so onset ramps are included;
    * ``min_signals`` 2 → a cluster of one is not a correlation.
    """

    merge_gap: timedelta = timedelta(minutes=15)
    pad: timedelta = timedelta(minutes=10)
    min_signals: int = 2


@dataclass(frozen=True, slots=True)
class _Span:
    signal: SignalRef
    start: datetime
    end: datetime


def _signal_spans(anomalies: list[Anomaly]) -> list[_Span]:
    """Collapse each signal's anomalies to one earliest-to-latest span."""
    by_signal: dict[SignalRef, tuple[datetime, datetime]] = {}
    for anomaly in anomalies:
        current = by_signal.get(anomaly.signal)
        if current is None:
            by_signal[anomaly.signal] = (anomaly.start, anomaly.end)
        else:
            by_signal[anomaly.signal] = (
                min(current[0], anomaly.start),
                max(current[1], anomaly.end),
            )
    spans = [_Span(signal, start, end) for signal, (start, end) in by_signal.items()]
    spans.sort(key=lambda span: (span.start, str(span.signal)))
    return spans


def _cluster_spans(spans: list[_Span], merge_gap: timedelta) -> list[list[_Span]]:
    """Interval sweep: group spans whose windows overlap within the gap."""
    clusters: list[list[_Span]] = []
    current: list[_Span] = []
    current_end: datetime | None = None

    for span in spans:  # already sorted by start
        if current_end is None or span.start <= current_end + merge_gap:
            current.append(span)
            current_end = span.end if current_end is None else max(current_end, span.end)
        else:
            clusters.append(current)
            current = [span]
            current_end = span.end
    if current:
        clusters.append(current)
    return clusters


def _cluster_strength(
    wide: pd.DataFrame, signals: list[SignalRef], start: datetime, end: datetime
) -> float:
    """Mean absolute pairwise Pearson correlation over the window."""
    columns = [str(signal) for signal in signals]
    window = wide.loc[(wide.index >= start) & (wide.index <= end), columns]
    if len(window) < 3:
        return 0.0
    corr = window.corr().to_numpy()
    n = corr.shape[0]
    off_diagonal = corr[~np.eye(n, dtype=bool)]
    if off_diagonal.size == 0 or np.isnan(off_diagonal).all():
        return 0.0
    return float(np.nanmean(np.abs(off_diagonal)))


def find_correlation_clusters(
    wide: pd.DataFrame,
    anomalies: list[Anomaly],
    config: CorrelationConfig | None = None,
) -> list[CorrelationCluster]:
    """Group co-anomalous signals and score how strongly they co-moved.

    Returns clusters sorted by start time. ``ordering`` lists each
    cluster's signals by first-anomaly time (earliest first).
    """
    cfg = config or CorrelationConfig()
    spans = _signal_spans(anomalies)
    clusters: list[CorrelationCluster] = []

    for group in _cluster_spans(spans, cfg.merge_gap):
        if len(group) < cfg.min_signals:
            continue
        start = min(span.start for span in group) - cfg.pad
        end = max(span.end for span in group) + cfg.pad
        ordering = tuple(span.signal for span in group)  # group is start-sorted
        clusters.append(
            CorrelationCluster(
                signals=tuple(sorted(ordering, key=str)),
                start=start,
                end=end,
                strength=_cluster_strength(wide, list(ordering), start, end),
                ordering=ordering,
            )
        )

    clusters.sort(key=lambda cluster: cluster.start)
    return clusters
