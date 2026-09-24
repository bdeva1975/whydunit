"""The default source: seeded synthetic telemetry with one injected incident."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from whydunit.models import IncidentCategory
from whydunit.simulator import generate_dataset
from whydunit.sources.base import validate_telemetry

_DEFAULT_START = datetime.fromisoformat("2026-09-01T00:00:00+00:00")


@dataclass(frozen=True, slots=True)
class SyntheticTelemetrySource:
    """Wraps the simulator behind the ``TelemetrySource`` Protocol."""

    category: IncidentCategory = IncidentCategory.RETRIEVAL_DEGRADATION
    start: datetime = _DEFAULT_START
    days: float = 1.0
    freq: str = "1min"
    seed: int = 42
    incident_start_frac: float = 0.5
    incident_duration: timedelta = field(default=timedelta(hours=2))

    def load(self) -> pd.DataFrame:
        dataset = generate_dataset(
            self.category,
            start=self.start,
            days=self.days,
            freq=self.freq,
            seed=self.seed,
            incident_start_frac=self.incident_start_frac,
            incident_duration=self.incident_duration,
        )
        return validate_telemetry(dataset.telemetry)
