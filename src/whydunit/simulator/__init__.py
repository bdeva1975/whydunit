"""Synthetic pipeline simulator: telemetry, scenarios, dataset generation."""

from whydunit.simulator.generator import (
    GeneratedDataset,
    generate_dataset,
    periods_for,
    write_dataset,
)
from whydunit.simulator.scenarios import (
    SCENARIO_BUILDERS,
    Effect,
    ScenarioSpec,
    build_scenario,
    ground_truth_for,
    inject,
)
from whydunit.simulator.telemetry import generate_baseline, make_time_index, to_wide

__all__ = [
    "SCENARIO_BUILDERS",
    "Effect",
    "GeneratedDataset",
    "ScenarioSpec",
    "build_scenario",
    "generate_baseline",
    "generate_dataset",
    "ground_truth_for",
    "inject",
    "make_time_index",
    "periods_for",
    "to_wide",
    "write_dataset",
]
