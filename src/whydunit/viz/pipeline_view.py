"""The pipeline health view: stages left-to-right, coloured by health.

Layout is deterministic: stages sit at their topological depth on the
x-axis (works for branching DAGs too — parallel branches stack on y).
No spring layouts anywhere in this project: screenshots and demos must
be reproducible.
"""

from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go

from whydunit.forensics import build_stage_graph
from whydunit.models import HealthState, Stage

HEALTH_COLORS: dict[HealthState, str] = {
    HealthState.HEALTHY: "#2e8b57",
    HealthState.WARNING: "#e6b800",
    HealthState.DEGRADED: "#e67e22",
    HealthState.FAILED: "#d64541",
}
_EDGE_COLOR = "rgba(150, 150, 150, 0.6)"
_TEMPLATE = "plotly_dark"


def _positions(graph: nx.DiGraph) -> dict[Stage, tuple[float, float]]:
    """x = topological depth; y spreads stages sharing a depth."""
    depth: dict[Stage, int] = {}
    for stage in nx.topological_sort(graph):
        parents = list(graph.predecessors(stage))
        depth[stage] = 1 + max((depth[p] for p in parents), default=-1)

    by_depth: dict[int, list[Stage]] = {}
    for stage, d in depth.items():
        by_depth.setdefault(d, []).append(stage)

    positions: dict[Stage, tuple[float, float]] = {}
    for d, stages in by_depth.items():
        stages.sort(key=lambda s: s.value)
        offset = (len(stages) - 1) / 2.0
        for i, stage in enumerate(stages):
            positions[stage] = (float(d), float(i) - offset)
    return positions


def pipeline_figure(
    health: dict[Stage, HealthState] | None = None,
    title: str = "Pipeline health",
) -> go.Figure:
    """Draw the stage DAG; ``health`` colours the nodes (default: all healthy)."""
    graph = build_stage_graph()
    positions = _positions(graph)
    states = health or dict.fromkeys(Stage, HealthState.HEALTHY)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for upstream, downstream in graph.edges:
        x0, y0 = positions[upstream]
        x1, y1 = positions[downstream]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": _EDGE_COLOR, "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    ordered = list(nx.topological_sort(graph))
    figure.add_trace(
        go.Scatter(
            x=[positions[stage][0] for stage in ordered],
            y=[positions[stage][1] for stage in ordered],
            mode="markers+text",
            marker={
                "size": 26,
                "color": [HEALTH_COLORS[states[stage]] for stage in ordered],
                "line": {"color": "#dddddd", "width": 1},
            },
            text=[stage.value for stage in ordered],
            textposition="bottom center",
            textfont={"size": 11},
            customdata=[states[stage].value for stage in ordered],
            hovertemplate="%{text}: %{customdata}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        template=_TEMPLATE,
        title=title,
        height=300,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False, "range": [-2.0, 2.0]},
    )
    return figure
