"""Telemetry sources: the seam where real data replaces synthetic."""

from whydunit.sources.base import REQUIRED_COLUMNS, TelemetrySource, validate_telemetry
from whydunit.sources.csv import CSVTelemetrySource
from whydunit.sources.synthetic import SyntheticTelemetrySource

__all__ = [
    "REQUIRED_COLUMNS",
    "CSVTelemetrySource",
    "SyntheticTelemetrySource",
    "TelemetrySource",
    "validate_telemetry",
]
