"""Plotly chart builders for the forensic console.

Pure functions: wide telemetry + findings in, ``go.Figure`` out. No
Streamlit imports, so everything here is testable headless.

Every chart answers a diagnostic question, per the product spec:

* :func:`signal_timeline` — what did these signals do, when, and where
  are the detected anomaly windows and level shifts?
* :func:`correlation_heatmap` — which signals co-moved inside the
  incident window?
* :func:`compare_signal` — how does this signal differ between two
  datasets (two incidents, or incident vs healthy)?

Timezone rule: convert to the display timezone (IST by default), then
strip tzinfo. Plotly's timezone handling differs between renderers;
naive-after-conversion is deterministic everywhere, and the timezone is
named on the axis so nothing is hidden.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from whydunit.models import DEFAULT_DISPLAY_TZ, Anomaly, ChangePoint, SignalRef

ANOMALY_FILL = "rgba(214, 69, 65, 0.18)"
CHANGEPOINT_LINE = "rgba(230, 126, 34, 0.9)"
TRACE_COLOR = "#4c78a8"
COMPARE_COLORS = ("#4c78a8", "#d64541")
_TEMPLATE = "plotly_dark"


def _to_display(index: pd.DatetimeIndex, display_tz: str) -> pd.DatetimeIndex:
    return index.tz_convert(ZoneInfo(display_tz)).tz_localize(None)


def _moment_to_display(moment: datetime, display_tz: str) -> datetime:
    return moment.astimezone(ZoneInfo(display_tz)).replace(tzinfo=None)


def signal_timeline(
    wide: pd.DataFrame,
    signals: Sequence[SignalRef],
    anomalies: Sequence[Anomaly] = (),
    changepoints: Sequence[ChangePoint] = (),
    display_tz: str = DEFAULT_DISPLAY_TZ,
    title: str | None = None,
) -> go.Figure:
    """Stacked time-series panels with anomaly shading and shift markers."""
    if not signals:
        raise ValueError("signal_timeline needs at least one signal")

    rows = len(signals)
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.08, 0.3 / rows),
        subplot_titles=[str(signal) for signal in signals],
    )
    display_index = _to_display(wide.index, display_tz)

    for row, signal in enumerate(signals, start=1):
        column = str(signal)
        figure.add_trace(
            go.Scatter(
                x=display_index,
                y=wide[column],
                mode="lines",
                line={"color": TRACE_COLOR, "width": 1.2},
                name=column,
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        for anomaly in anomalies:
            if anomaly.signal == signal:
                figure.add_vrect(
                    x0=_moment_to_display(anomaly.start, display_tz),
                    x1=_moment_to_display(anomaly.end, display_tz),
                    fillcolor=ANOMALY_FILL,
                    line_width=0,
                    row=row,
                    col=1,
                )
        for changepoint in changepoints:
            if changepoint.signal == signal:
                figure.add_vline(
                    x=_moment_to_display(changepoint.at, display_tz),
                    line={"color": CHANGEPOINT_LINE, "width": 1.5, "dash": "dot"},
                    row=row,
                    col=1,
                )

    figure.update_layout(
        template=_TEMPLATE,
        height=max(240, 170 * rows),
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        title=title,
    )
    figure.update_xaxes(title_text=f"time ({display_tz})", row=rows, col=1)
    return figure


def correlation_heatmap(
    wide: pd.DataFrame,
    signals: Sequence[SignalRef],
    start: datetime | None = None,
    end: datetime | None = None,
    title: str | None = None,
) -> go.Figure:
    """Pairwise Pearson correlation of the given signals over a window."""
    if len(signals) < 2:
        raise ValueError("correlation_heatmap needs at least two signals")

    columns = [str(signal) for signal in signals]
    window = wide
    if start is not None:
        window = window[window.index >= start]
    if end is not None:
        window = window[window.index <= end]

    corr = window[columns].corr()
    figure = go.Figure(
        data=go.Heatmap(
            z=corr.to_numpy(),
            x=columns,
            y=columns,
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "r"},
        )
    )
    figure.update_layout(
        template=_TEMPLATE,
        height=max(360, 40 * len(columns) + 160),
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        title=title or "Signal correlation",
    )
    return figure


def compare_signal(
    wide_a: pd.DataFrame,
    wide_b: pd.DataFrame,
    signal: SignalRef,
    labels: tuple[str, str] = ("A", "B"),
    display_tz: str = DEFAULT_DISPLAY_TZ,
) -> go.Figure:
    """Overlay one signal from two datasets (incident vs incident/healthy)."""
    column = str(signal)
    figure = go.Figure()
    for wide, label, color in zip((wide_a, wide_b), labels, COMPARE_COLORS, strict=True):
        figure.add_trace(
            go.Scatter(
                x=_to_display(wide.index, display_tz),
                y=wide[column],
                mode="lines",
                line={"color": color, "width": 1.2},
                name=label,
            )
        )
    figure.update_layout(
        template=_TEMPLATE,
        height=320,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        title=str(signal),
        xaxis_title=f"time ({display_tz})",
    )
    return figure
