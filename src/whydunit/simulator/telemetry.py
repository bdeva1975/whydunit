"""Baseline (healthy-state) telemetry generation.

Produces the canonical long-form telemetry frame:

    timestamp (tz-aware UTC) | stage | metric | value

Design points:

* Deterministic: one ``numpy`` Generator seeded once drives everything.
* Mild diurnal seasonality (24h sine) is added to every metric at the
  scale of its own noise, so baselines wobble realistically instead of
  being flat lines — a detector tuned on flat lines would be a toy.
* Values are clipped to each metric's hard bounds, so rates stay in
  [0, 1] and counts stay non-negative.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from whydunit.models import Stage
from whydunit.simulator.pipeline import STAGE_METRICS

TELEMETRY_COLUMNS = ("timestamp", "stage", "metric", "value")

_SEASONALITY_PERIOD_HOURS = 24.0
_SEASONALITY_AMPLITUDE_FACTOR = 1.0  # seasonal swing ≈ one noise_std


def make_time_index(start: datetime, periods: int, freq: str = "1min") -> pd.DatetimeIndex:
    """Build the shared UTC time index.

    ``start`` must be timezone-aware; it is converted to UTC. Rejecting
    naive datetimes at the boundary keeps the naive/aware mix-up class of
    bugs out of the entire codebase.
    """
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware (use datetime(..., tzinfo=UTC))")
    return pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")


def generate_baseline(
    start: datetime,
    periods: int,
    freq: str = "1min",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate healthy telemetry for every signal in the catalogue.

    Returns a long-form frame with ``TELEMETRY_COLUMNS``, sorted by
    timestamp then stage order. Identical inputs yield identical output.
    """
    index = make_time_index(start, periods, freq)
    rng = np.random.default_rng(seed)

    hours = (index - index[0]).total_seconds() / 3600.0
    season = np.sin(2.0 * np.pi * hours.to_numpy() / _SEASONALITY_PERIOD_HOURS)

    frames: list[pd.DataFrame] = []
    for stage in Stage:
        for spec in STAGE_METRICS[stage]:
            # Per-signal phase offset so all metrics don't peak together.
            phase = rng.uniform(0.0, 2.0 * np.pi)
            seasonal = (
                _SEASONALITY_AMPLITUDE_FACTOR
                * spec.noise_std
                * np.sin(2.0 * np.pi * hours.to_numpy() / _SEASONALITY_PERIOD_HOURS + phase)
            )
            noise = rng.normal(loc=0.0, scale=spec.noise_std, size=periods)
            values = np.clip(spec.baseline + seasonal + noise, spec.low, spec.high)
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": index,
                        "stage": stage.value,
                        "metric": spec.name,
                        "value": values,
                    }
                )
            )

    _ = season  # base curve kept for readability; per-signal phases used above
    telemetry = pd.concat(frames, ignore_index=True)
    return telemetry.sort_values(["timestamp", "stage", "metric"], ignore_index=True)


def to_wide(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form telemetry to wide: one ``stage.metric`` column per signal.

    The forensic engine works on this shape; the long form stays the
    storage/interchange format.
    """
    wide = telemetry.pivot_table(
        index="timestamp",
        columns=["stage", "metric"],
        values="value",
        aggfunc="mean",
    )
    wide.columns = [f"{stage}.{metric}" for stage, metric in wide.columns]
    return wide.sort_index()
