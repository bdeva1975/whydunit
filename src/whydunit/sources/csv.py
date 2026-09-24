"""CSV source: load telemetry the CLI generator (or anything else) wrote.

The first proof that the source seam is real: a dataset written by
``python -m whydunit.simulator`` — or exported from any external system
into the same four-column shape — loads back and investigates
identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from whydunit.sources.base import validate_telemetry


@dataclass(frozen=True, slots=True)
class CSVTelemetrySource:
    """Reads canonical long-form telemetry from a CSV file."""

    path: Path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"telemetry file not found: {self.path}")
        frame = pd.read_csv(self.path, parse_dates=["timestamp"])
        if (
            pd.api.types.is_datetime64_any_dtype(frame["timestamp"])
            and frame["timestamp"].dt.tz is None
        ):
            # ISO strings with offsets normally parse tz-aware; a naive
            # result means bare timestamps — reject rather than guess.
            raise ValueError(f"{self.path}: timestamps are not timezone-aware")
        return validate_telemetry(frame)
