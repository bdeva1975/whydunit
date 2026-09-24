"""The evidence graph: signals → evidence → hypotheses, layered.

The visually distinctive view of the console. Three deterministic
columns: anomalous signals on the left, evidence items in the middle,
ranked hypotheses on the right. Green edges support, red edges
contradict; hypothesis nodes are sized by score and labelled with their
confidence band. Everything is derived from the investigation result —
the graph *is* the engine's reasoning, drawn.
"""

from __future__ import annotations

import plotly.graph_objects as go

from whydunit.forensics import InvestigationResult
from whydunit.models import ConfidenceBand, Evidence, Hypothesis, SignalRef

_SUPPORT_COLOR = "rgba(46, 139, 87, 0.55)"
_CONTRADICT_COLOR = "rgba(214, 69, 65, 0.55)"
_SIGNAL_COLOR = "#4c78a8"
_EVIDENCE_COLOR = "#8c8c8c"
_HYPOTHESIS_COLORS: dict[ConfidenceBand, str] = {
    ConfidenceBand.STRONG: "#2e8b57",
    ConfidenceBand.MODERATE: "#e6b800",
    ConfidenceBand.WEAK: "#e67e22",
    ConfidenceBand.INSUFFICIENT: "#777777",
}
_TEMPLATE = "plotly_dark"

_X_SIGNAL, _X_EVIDENCE, _X_HYPOTHESIS = 0.0, 1.0, 2.0


def _spread(count: int) -> list[float]:
    """Symmetric y positions for ``count`` nodes in one column."""
    if count == 0:
        return []
    offset = (count - 1) / 2.0
    return [float(i) - offset for i in range(count)]


def evidence_graph_figure(
    result: InvestigationResult,
    max_hypotheses: int = 4,
    title: str = "Evidence graph",
) -> go.Figure:
    """Draw the layered signals → evidence → hypotheses graph."""
    hypotheses: list[Hypothesis] = list(result.hypotheses[:max_hypotheses])

    evidence_items: list[tuple[Evidence, Hypothesis, bool]] = []
    for hypothesis in hypotheses:
        for item in hypothesis.supporting:
            evidence_items.append((item, hypothesis, True))
        for item in hypothesis.contradicting:
            evidence_items.append((item, hypothesis, False))

    signals: list[SignalRef] = sorted(
        {ref for item, _, _ in evidence_items for ref in item.signals}, key=str
    )

    signal_y = dict(zip(signals, _spread(len(signals)), strict=True))
    evidence_y = {
        item.id: y
        for (item, _, _), y in zip(evidence_items, _spread(len(evidence_items)), strict=True)
    }
    hypothesis_y = dict(zip([h.id for h in hypotheses], _spread(len(hypotheses)), strict=True))

    figure = go.Figure()

    for item, hypothesis, supports in evidence_items:
        color = _SUPPORT_COLOR if supports else _CONTRADICT_COLOR
        for ref in item.signals:
            figure.add_trace(
                go.Scatter(
                    x=[_X_SIGNAL, _X_EVIDENCE],
                    y=[signal_y[ref], evidence_y[item.id]],
                    mode="lines",
                    line={"color": color, "width": 1.0},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        figure.add_trace(
            go.Scatter(
                x=[_X_EVIDENCE, _X_HYPOTHESIS],
                y=[evidence_y[item.id], hypothesis_y[hypothesis.id]],
                mode="lines",
                line={"color": color, "width": 1.4},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if signals:
        figure.add_trace(
            go.Scatter(
                x=[_X_SIGNAL] * len(signals),
                y=[signal_y[ref] for ref in signals],
                mode="markers+text",
                marker={"size": 12, "color": _SIGNAL_COLOR},
                text=[str(ref) for ref in signals],
                textposition="middle left",
                textfont={"size": 10},
                hovertemplate="%{text}<extra>signal</extra>",
                showlegend=False,
            )
        )

    if evidence_items:
        figure.add_trace(
            go.Scatter(
                x=[_X_EVIDENCE] * len(evidence_items),
                y=[evidence_y[item.id] for item, _, _ in evidence_items],
                mode="markers",
                marker={"size": 8, "color": _EVIDENCE_COLOR, "symbol": "diamond"},
                customdata=[
                    f"[{item.kind.value}] {item.statement}" for item, _, _ in evidence_items
                ],
                hovertemplate="%{customdata}<extra>evidence</extra>",
                showlegend=False,
            )
        )

    if hypotheses:
        figure.add_trace(
            go.Scatter(
                x=[_X_HYPOTHESIS] * len(hypotheses),
                y=[hypothesis_y[h.id] for h in hypotheses],
                mode="markers+text",
                marker={
                    "size": [max(14.0, 10.0 + 2.0 * h.score) for h in hypotheses],
                    "color": [_HYPOTHESIS_COLORS[h.confidence] for h in hypotheses],
                    "line": {"color": "#dddddd", "width": 1},
                },
                text=[f"{h.category.value} [{h.confidence.value}]" for h in hypotheses],
                textposition="middle right",
                textfont={"size": 10},
                customdata=[h.statement for h in hypotheses],
                hovertemplate="%{customdata}<extra>hypothesis</extra>",
                showlegend=False,
            )
        )

    height = max(420, 30 * max(len(evidence_items), 1) + 160)
    figure.update_layout(
        template=_TEMPLATE,
        title=title,
        height=height,
        margin={"l": 160, "r": 200, "t": 60, "b": 20},
        xaxis={"visible": False, "range": [-0.8, 3.2]},
        yaxis={"visible": False},
    )
    return figure
