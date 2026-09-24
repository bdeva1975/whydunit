"""Synthetic incident scenarios and the injection machinery.

An incident is a set of :class:`Effect` perturbations applied to baseline
telemetry. Effects are expressed in units of each signal's own
``noise_std`` (σ), so "+8σ" means the same thing on a latency metric and
a rate metric. Each effect carries an onset delay and a ramp, giving
incidents realistic temporal structure: the root-cause signal moves
first, downstream symptoms follow.

v0.1 simplifications, documented deliberately:

* all effects of a scenario end together at the incident end (no
  staggered recovery),
* effects are additive shifts (no variance changes) — variance-based
  scenarios can be added later without touching the injector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from whydunit.models import (
    GroundTruth,
    IncidentCategory,
    IncidentScenario,
    Severity,
    SignalRef,
    Stage,
)
from whydunit.simulator.pipeline import metric_spec


@dataclass(frozen=True, slots=True)
class Effect:
    """One perturbation of one signal during an incident window.

    ``shift_sigmas``: additive shift in units of the signal's noise_std;
    negative values push the signal down.
    ``onset_frac``: fraction of the incident duration that passes before
    this effect begins (0.0 = at incident start).
    ``ramp_frac``: fraction of the effect's active window spent ramping
    linearly from zero to full strength.
    """

    signal: SignalRef
    shift_sigmas: float
    onset_frac: float = 0.0
    ramp_frac: float = 0.2


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """A fully-specified incident: the scenario record plus its effects."""

    scenario: IncidentScenario
    effects: tuple[Effect, ...]


ScenarioBuilder = Callable[[datetime, timedelta], ScenarioSpec]


def _spec(
    *,
    slug: str,
    name: str,
    category: IncidentCategory,
    description: str,
    severity: Severity,
    start: datetime,
    duration: timedelta,
    expected_root_cause: str,
    effects: tuple[Effect, ...],
) -> ScenarioSpec:
    """Assemble a ScenarioSpec; the first effect is the primary signal."""
    affected_stages: tuple[Stage, ...] = tuple(
        dict.fromkeys(effect.signal.stage for effect in effects)
    )
    scenario = IncidentScenario(
        id=f"INC-{start:%Y%m%d-%H%M}-{slug}",
        name=name,
        category=category,
        description=description,
        severity=severity,
        affected_stages=affected_stages,
        start=start,
        duration=duration,
        primary_signal=effects[0].signal,
        secondary_signals=tuple(effect.signal for effect in effects[1:]),
        expected_root_cause=expected_root_cause,
    )
    return ScenarioSpec(scenario=scenario, effects=effects)


def inject(telemetry: pd.DataFrame, spec: ScenarioSpec) -> pd.DataFrame:
    """Apply a scenario's effects to baseline telemetry.

    Returns a new frame (input is not mutated). Deterministic: the
    envelope is a pure function of timestamps, so identical inputs yield
    identical output. Values are re-clipped to each metric's bounds.
    """
    result = telemetry.copy()
    incident_start = spec.scenario.start
    incident_end = incident_start + spec.scenario.duration

    for effect in spec.effects:
        effect_start = incident_start + effect.onset_frac * spec.scenario.duration
        active_seconds = (incident_end - effect_start).total_seconds()
        if active_seconds <= 0:
            raise ValueError(f"effect on {effect.signal} starts after the incident ends")
        ramp_seconds = max(active_seconds * effect.ramp_frac, 1.0)

        mask = (
            (result["stage"] == effect.signal.stage.value)
            & (result["metric"] == effect.signal.metric)
            & (result["timestamp"] >= effect_start)
            & (result["timestamp"] < incident_end)
        )
        if not mask.any():
            continue

        elapsed = (result.loc[mask, "timestamp"] - effect_start).dt.total_seconds()
        envelope = np.minimum(elapsed.to_numpy() / ramp_seconds, 1.0)

        spec_m = metric_spec(effect.signal)
        shifted = result.loc[mask, "value"].to_numpy() + (
            effect.shift_sigmas * spec_m.noise_std * envelope
        )
        result.loc[mask, "value"] = np.clip(shifted, spec_m.low, spec_m.high)

    return result


def ground_truth_for(spec: ScenarioSpec) -> GroundTruth:
    """Derive the eval-only answer key from a scenario spec."""
    return GroundTruth(
        scenario_id=spec.scenario.id,
        category=spec.scenario.category,
        root_cause_stage=spec.scenario.primary_signal.stage,
        incident_start=spec.scenario.start,
        incident_end=spec.scenario.start + spec.scenario.duration,
        injected_signals=(
            spec.scenario.primary_signal,
            *spec.scenario.secondary_signals,
        ),
    )


# --------------------------------------------------------------------------
# Scenario builders
# --------------------------------------------------------------------------
# Magnitude convention: primary fault 8–10σ, first-order knock-ons 5–7σ,
# far-downstream quality effects 3–5σ with later onsets — strong enough to
# detect reliably, weak enough that ordering (not magnitude alone) matters.


def build_schema_drift(start: datetime, duration: timedelta) -> ScenarioSpec:
    """Upstream schema change: violations spike, quality decays downstream."""
    return _spec(
        slug="schema-drift",
        name="Schema drift at ingestion",
        category=IncidentCategory.SCHEMA_DRIFT,
        description=(
            "An upstream producer shipped a schema change. Schema violations "
            "and nulls spike at ingestion; validation failures follow; feature "
            "quality decays; answer quality degrades minutes later."
        ),
        severity=Severity.HIGH,
        start=start,
        duration=duration,
        expected_root_cause="Producer-side schema change at the ingestion boundary",
        effects=(
            Effect(SignalRef(Stage.INGESTION, "schema_violations"), +10.0, 0.00),
            Effect(SignalRef(Stage.INGESTION, "null_pct"), +6.0, 0.05),
            Effect(SignalRef(Stage.VALIDATION, "validation_failures"), +8.0, 0.10),
            Effect(SignalRef(Stage.VALIDATION, "missing_fields_pct"), +5.0, 0.10),
            Effect(SignalRef(Stage.FEATURE_ENGINEERING, "feature_null_rate"), +6.0, 0.20),
            Effect(SignalRef(Stage.EVALUATION, "eval_score"), -4.0, 0.35),
            Effect(SignalRef(Stage.EVALUATION, "groundedness"), -3.0, 0.40),
        ),
    )


def build_retrieval_degradation(start: datetime, duration: timedelta) -> ScenarioSpec:
    """Retrieval quality collapses; the LLM hallucinates on thin context."""
    return _spec(
        slug="retrieval-degradation",
        name="Retrieval degradation",
        category=IncidentCategory.RETRIEVAL_DEGRADATION,
        description=(
            "Retrieval similarity and context relevance drop (e.g. a bad "
            "index build); empty retrievals rise; groundedness falls and "
            "hallucinations climb as the LLM answers on thin context."
        ),
        severity=Severity.HIGH,
        start=start,
        duration=duration,
        expected_root_cause="Vector index or retrieval configuration regression",
        effects=(
            Effect(SignalRef(Stage.RETRIEVAL, "top_k_similarity"), -8.0, 0.00),
            Effect(SignalRef(Stage.RETRIEVAL, "context_relevance"), -7.0, 0.05),
            Effect(SignalRef(Stage.RETRIEVAL, "empty_retrieval_rate"), +6.0, 0.05),
            Effect(SignalRef(Stage.EVALUATION, "groundedness"), -6.0, 0.20),
            Effect(SignalRef(Stage.EVALUATION, "hallucination_rate"), +6.0, 0.25),
            Effect(SignalRef(Stage.EVALUATION, "eval_score"), -4.0, 0.30),
        ),
    )


SCENARIO_BUILDERS: dict[IncidentCategory, ScenarioBuilder] = {
    IncidentCategory.SCHEMA_DRIFT: build_schema_drift,
    IncidentCategory.RETRIEVAL_DEGRADATION: build_retrieval_degradation,
}


def build_scenario(
    category: IncidentCategory, start: datetime, duration: timedelta
) -> ScenarioSpec:
    """Build a scenario by category. Raises ``KeyError`` if not implemented."""
    try:
        builder = SCENARIO_BUILDERS[category]
    except KeyError as exc:
        raise KeyError(f"no builder registered for {category.value!r}") from exc
    return builder(start, duration)
