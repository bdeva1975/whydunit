"""Dataset generation façade and file output.

One call produces the three artefacts of a synthetic incident dataset:

* ``telemetry.csv`` — long-form telemetry (the investigation input),
* ``scenario.json`` — the incident metadata shown to the investigator,
* ``ground_truth.json`` — the answer key. EVAL-ONLY: the forensic engine
  never reads this file; it exists so the engine can be scored.

JSON schemas are written out field-by-field on purpose: these files are
a documented interchange format, not an accidental object dump.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from whydunit.models import GroundTruth, IncidentCategory
from whydunit.simulator.scenarios import (
    ScenarioSpec,
    build_scenario,
    ground_truth_for,
    inject,
)
from whydunit.simulator.telemetry import generate_baseline

TELEMETRY_FILENAME = "telemetry.csv"
SCENARIO_FILENAME = "scenario.json"
GROUND_TRUTH_FILENAME = "ground_truth.json"


@dataclass(frozen=True, slots=True)
class GeneratedDataset:
    """Everything one generation run produces, still in memory."""

    telemetry: pd.DataFrame
    spec: ScenarioSpec
    truth: GroundTruth


def periods_for(days: float, freq: str) -> int:
    """Number of samples covering ``days`` at ``freq`` (e.g. '1min')."""
    step_seconds = pd.Timedelta(freq).total_seconds()
    if step_seconds <= 0:
        raise ValueError(f"frequency must be positive, got {freq!r}")
    periods = int(days * 86400.0 / step_seconds)
    if periods < 10:
        raise ValueError(f"{days} days at {freq!r} yields only {periods} samples")
    return periods


def generate_dataset(
    category: IncidentCategory,
    *,
    start: datetime,
    days: float = 7.0,
    freq: str = "1min",
    seed: int = 42,
    incident_start_frac: float = 0.5,
    incident_duration: timedelta = timedelta(hours=2),
) -> GeneratedDataset:
    """Generate baseline telemetry with one scenario injected.

    The incident begins at ``incident_start_frac`` of the data window and
    must end inside it. Identical arguments yield identical output.
    """
    if not 0.0 <= incident_start_frac < 1.0:
        raise ValueError("incident_start_frac must be in [0, 1)")
    periods = periods_for(days, freq)
    window = pd.Timedelta(freq) * periods
    incident_start = start + incident_start_frac * window
    if incident_start + incident_duration > start + window:
        raise ValueError("incident does not fit inside the data window")

    baseline = generate_baseline(start, periods=periods, freq=freq, seed=seed)
    spec = build_scenario(category, incident_start, incident_duration)
    return GeneratedDataset(
        telemetry=inject(baseline, spec),
        spec=spec,
        truth=ground_truth_for(spec),
    )


def scenario_to_dict(spec: ScenarioSpec) -> dict[str, Any]:
    """Investigator-facing scenario metadata (no ground truth inside)."""
    scenario = spec.scenario
    return {
        "id": scenario.id,
        "name": scenario.name,
        "category": scenario.category.value,
        "description": scenario.description,
        "severity": scenario.severity.value,
        "affected_stages": [stage.value for stage in scenario.affected_stages],
        "start": scenario.start.isoformat(),
        "duration_seconds": scenario.duration.total_seconds(),
    }


def truth_to_dict(truth: GroundTruth) -> dict[str, Any]:
    """Answer-key schema. Consumed by the eval harness only."""
    return {
        "scenario_id": truth.scenario_id,
        "category": truth.category.value,
        "root_cause_stage": (
            truth.root_cause_stage.value if truth.root_cause_stage is not None else None
        ),
        "incident_start": truth.incident_start.isoformat(),
        "incident_end": truth.incident_end.isoformat(),
        "injected_signals": [str(signal) for signal in truth.injected_signals],
    }


def write_dataset(dataset: GeneratedDataset, output_dir: Path) -> tuple[Path, Path, Path]:
    """Write the three artefacts into ``output_dir`` (created if missing).

    Existing files are overwritten — generation is cheap and seeded, so
    the files are reproducible, not precious.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = output_dir / TELEMETRY_FILENAME
    dataset.telemetry.to_csv(telemetry_path, index=False)

    scenario_path = output_dir / SCENARIO_FILENAME
    scenario_path.write_text(json.dumps(scenario_to_dict(dataset.spec), indent=2), encoding="utf-8")

    truth_path = output_dir / GROUND_TRUTH_FILENAME
    truth_path.write_text(json.dumps(truth_to_dict(dataset.truth), indent=2), encoding="utf-8")

    return telemetry_path, scenario_path, truth_path
