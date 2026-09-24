"""The telemetry-source seam: how real data will one day replace synthetic.

Anything that can produce the canonical long-form telemetry frame
(``timestamp`` tz-aware UTC, ``stage``, ``metric``, ``value``) can feed
the forensic engine. The engine itself never cares where telemetry came
from — it takes a wide frame; sources produce the long frame that
``to_wide`` pivots.

Future implementations (OpenTelemetry, Prometheus, Langfuse, Postgres…)
implement this same Protocol: fetch, map their metric names onto the
``stage.metric`` catalogue, return the long frame. ``validate_telemetry``
is the contract check every source can run on its own output.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "stage", "metric", "value")


@runtime_checkable
class TelemetrySource(Protocol):
    """Anything that yields canonical long-form telemetry."""

    def load(self) -> pd.DataFrame:
        """Return a long-form telemetry frame (see ``REQUIRED_COLUMNS``)."""
        ...


def validate_telemetry(frame: pd.DataFrame) -> pd.DataFrame:
    """Check a frame against the telemetry contract; return it unchanged.

    Raises ``ValueError`` with a precise message on the first violation:
    missing columns, naive or non-datetime timestamps, or non-numeric
    values. Unknown stage/metric names are permitted — a real source may
    carry extra signals; the engine simply has no signatures for them.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"telemetry frame missing columns: {missing}")

    timestamps = frame["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        raise ValueError("timestamp column must be datetime64")
    if timestamps.dt.tz is None:
        raise ValueError("timestamps must be timezone-aware (UTC)")

    if not pd.api.types.is_numeric_dtype(frame["value"]):
        raise ValueError("value column must be numeric")

    return frame
