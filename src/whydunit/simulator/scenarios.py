"""Synthetic incident scenarios and the injection machinery.

An incident is a set of :class:`Effect` perturbations applied to baseline
telemetry. Effects are expressed in units of each signal's own
``noise_std`` (σ), so "+8σ" means the same thing on a latency metric and
a rate metric. Each effect carries an onset delay and a ramp, giving
incidents realistic temporal structure: the root-cause signal moves
first, downstream symptoms follow.

Magnitude convention: primary faults 8–10σ, first-order knock-ons 5–7σ,
far-downstream quality effects 3–5σ with later onsets — strong enough to
detect reliably, weak enough that ordering (not magnitude alone) matters.

Fingerprint design (what lets the forensic engine tell scenarios apart):

* schema drift vs data quality: violations spike only in schema drift;
* retrieval degradation vs model regression: retrieval signals move only
  in the former — quality collapsing with a healthy upstream points at
  the model;
* model vs prompt regression: prompt regressions break parsing and token
  counts; model regressions degrade quality without parse failures;
* rate limiting vs resource exhaustion vs latency spike: errors dominate
  rate limiting, CPU/memory dominate exhaustion, a pure latency spike
  moves neither;
* cascading failure: one upstream fault, staggered onsets down the whole
  chain;
* multi-factor: two independent primaries start together — root-cause
  ambiguity is the point.

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
        primary_signal=effects[0].signal if effects else None,
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
    primary = spec.scenario.primary_signal
    return GroundTruth(
        scenario_id=spec.scenario.id,
        category=spec.scenario.category,
        root_cause_stage=primary.stage if primary is not None else None,
        incident_start=spec.scenario.start,
        incident_end=spec.scenario.start + spec.scenario.duration,
        injected_signals=(
            (primary, *spec.scenario.secondary_signals) if primary is not None else ()
        ),
    )


# --------------------------------------------------------------------------
# Scenario builders
# --------------------------------------------------------------------------


def build_normal(start: datetime, duration: timedelta) -> ScenarioSpec:
    """Healthy operation: nothing is injected."""
    return _spec(
        slug="normal",
        name="Normal operation",
        category=IncidentCategory.NORMAL,
        description="Healthy pipeline; no fault injected. The correct diagnosis is 'no incident'.",
        severity=Severity.NONE,
        start=start,
        duration=duration,
        expected_root_cause="",
        effects=(),
    )


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


def build_data_quality_degradation(start: datetime, duration: timedelta) -> ScenarioSpec:
    """Dirty data without a schema change: nulls, duplicates, drops."""
    return _spec(
        slug="data-quality",
        name="Data-quality degradation",
        category=IncidentCategory.DATA_QUALITY_DEGRADATION,
        description=(
            "Source data turned dirty without a schema change: nulls and "
            "duplicates rise, validation rejects more rows, feature nulls "
            "climb, and answer quality sags. Schema violations stay flat — "
            "that is what separates this from schema drift."
        ),
        severity=Severity.MEDIUM,
        start=start,
        duration=duration,
        expected_root_cause="Degraded source-data quality (no schema change)",
        effects=(
            Effect(SignalRef(Stage.INGESTION, "null_pct"), +8.0, 0.00),
            Effect(SignalRef(Stage.INGESTION, "records_rejected"), +5.0, 0.05),
            Effect(SignalRef(Stage.VALIDATION, "duplicate_rate"), +6.0, 0.05),
            Effect(SignalRef(Stage.VALIDATION, "missing_fields_pct"), +6.0, 0.10),
            Effect(SignalRef(Stage.PREPROCESSING, "rows_dropped"), +5.0, 0.15),
            Effect(SignalRef(Stage.FEATURE_ENGINEERING, "feature_null_rate"), +5.0, 0.20),
            Effect(SignalRef(Stage.EVALUATION, "eval_score"), -3.0, 0.35),
        ),
    )


def build_feature_drift(start: datetime, duration: timedelta) -> ScenarioSpec:
    """Feature distributions shift; model quality decays quietly."""
    return _spec(
        slug="feature-drift",
        name="Feature drift",
        category=IncidentCategory.FEATURE_DRIFT,
        description=(
            "Feature distributions drift away from what the model was tuned "
            "on. Upstream data checks stay green; the drift score rises first "
            "and answer quality decays slowly behind it."
        ),
        severity=Severity.MEDIUM,
        start=start,
        duration=duration,
        expected_root_cause="Input feature distribution shift (world changed, model did not)",
        effects=(
            Effect(SignalRef(Stage.FEATURE_ENGINEERING, "feature_drift_score"), +9.0, 0.00, 0.4),
            Effect(SignalRef(Stage.EVALUATION, "eval_score"), -4.0, 0.30, 0.4),
            Effect(SignalRef(Stage.EVALUATION, "relevance"), -3.0, 0.35, 0.4),
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


def build_llm_latency_spike(start: datetime, duration: timedelta) -> ScenarioSpec:
    """Provider latency spikes; quality is untouched."""
    return _spec(
        slug="llm-latency",
        name="LLM latency spike",
        category=IncidentCategory.LLM_LATENCY_SPIKE,
        description=(
            "The model provider slows down: latency and timeouts climb, "
            "throughput drops, delivery lags. CPU, memory, and answer "
            "quality stay normal — this is a provider problem, not ours."
        ),
        severity=Severity.HIGH,
        start=start,
        duration=duration,
        expected_root_cause="Upstream LLM provider latency degradation",
        effects=(
            Effect(SignalRef(Stage.MODEL_INFERENCE, "llm_latency_ms"), +9.0, 0.00),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "timeout_rate"), +6.0, 0.10),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "throughput_rpm"), -5.0, 0.10),
            Effect(SignalRef(Stage.DELIVERY, "delivery_latency_ms"), +4.0, 0.20),
        ),
    )


def build_model_regression(start: datetime, duration: timedelta) -> ScenarioSpec:
    """A worse model version ships; quality drops with a healthy upstream."""
    return _spec(
        slug="model-regression",
        name="Model version regression",
        category=IncidentCategory.MODEL_REGRESSION,
        description=(
            "A new model version answers differently and worse: token usage "
            "shifts at inference, then every quality metric degrades. All "
            "upstream stages stay healthy — quality collapsing on a clean "
            "upstream points at the model itself."
        ),
        severity=Severity.HIGH,
        start=start,
        duration=duration,
        expected_root_cause="Model version rollout with regressed answer quality",
        effects=(
            Effect(SignalRef(Stage.MODEL_INFERENCE, "tokens_per_request"), +4.0, 0.00),
            Effect(SignalRef(Stage.EVALUATION, "eval_score"), -6.0, 0.05),
            Effect(SignalRef(Stage.EVALUATION, "groundedness"), -4.0, 0.10),
            Effect(SignalRef(Stage.EVALUATION, "hallucination_rate"), +5.0, 0.10),
            Effect(SignalRef(Stage.EVALUATION, "relevance"), -4.0, 0.10),
        ),
    )


def build_prompt_regression(start: datetime, duration: timedelta) -> ScenarioSpec:
    """A prompt/template change breaks output structure and bloats tokens."""
    return _spec(
        slug="prompt-regression",
        name="Prompt/template regression",
        category=IncidentCategory.PROMPT_REGRESSION,
        description=(
            "A prompt or template change ships: token counts jump, output "
            "shape changes, parsers start failing, relevance sags. Parse "
            "failures plus a token jump separate this from a model "
            "regression."
        ),
        severity=Severity.MEDIUM,
        start=start,
        duration=duration,
        expected_root_cause="Prompt/template change regressing output structure",
        effects=(
            Effect(SignalRef(Stage.MODEL_INFERENCE, "tokens_per_request"), +7.0, 0.00),
            Effect(SignalRef(Stage.POSTPROCESSING, "parse_failure_rate"), +7.0, 0.05),
            Effect(SignalRef(Stage.POSTPROCESSING, "output_length_chars"), +5.0, 0.05),
            Effect(SignalRef(Stage.EVALUATION, "relevance"), -4.0, 0.20),
            Effect(SignalRef(Stage.EVALUATION, "eval_score"), -3.0, 0.25),
        ),
    )


def build_api_rate_limiting(start: datetime, duration: timedelta) -> ScenarioSpec:
    """The provider throttles us: errors dominate, infra stays calm."""
    return _spec(
        slug="rate-limiting",
        name="API rate limiting",
        category=IncidentCategory.API_RATE_LIMITING,
        description=(
            "The model API starts throttling: error and timeout rates jump, "
            "retries inflate latency, throughput and delivery suffer. CPU "
            "and memory stay flat — pressure is external, not ours."
        ),
        severity=Severity.MEDIUM,
        start=start,
        duration=duration,
        expected_root_cause="Provider-side rate limiting / quota exhaustion",
        effects=(
            Effect(SignalRef(Stage.MODEL_INFERENCE, "error_rate"), +8.0, 0.00),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "timeout_rate"), +5.0, 0.05),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "llm_latency_ms"), +4.0, 0.05),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "throughput_rpm"), -6.0, 0.10),
            Effect(SignalRef(Stage.DELIVERY, "delivery_failures"), +4.0, 0.20),
        ),
    )


def build_resource_exhaustion(start: datetime, duration: timedelta) -> ScenarioSpec:
    """Our own infrastructure saturates; everything on the box suffers."""
    return _spec(
        slug="resource-exhaustion",
        name="Resource exhaustion",
        category=IncidentCategory.RESOURCE_EXHAUSTION,
        description=(
            "Memory and CPU on the serving infrastructure saturate: latency "
            "and timeouts climb, throughput collapses, errors follow, "
            "delivery lags. The CPU/memory pair moving first is the "
            "fingerprint."
        ),
        severity=Severity.CRITICAL,
        start=start,
        duration=duration,
        expected_root_cause="Serving-infrastructure resource saturation (memory/CPU)",
        effects=(
            Effect(SignalRef(Stage.MODEL_INFERENCE, "memory_utilization_pct"), +8.0, 0.00),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "cpu_utilization_pct"), +7.0, 0.00),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "llm_latency_ms"), +6.0, 0.10),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "timeout_rate"), +5.0, 0.15),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "throughput_rpm"), -5.0, 0.15),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "error_rate"), +3.0, 0.20),
            Effect(SignalRef(Stage.DELIVERY, "delivery_latency_ms"), +3.0, 0.25),
        ),
    )


def build_cascading_failure(start: datetime, duration: timedelta) -> ScenarioSpec:
    """One upstream fault propagates down the entire chain, stage by stage."""
    return _spec(
        slug="cascading",
        name="Cascading failure from schema change",
        category=IncidentCategory.CASCADING_FAILURE,
        description=(
            "A schema change at ingestion corrupts validation, features, and "
            "embeddings; retrieval degrades on corrupted vectors; the LLM "
            "hallucinates; evaluation and delivery fail last. Looking only "
            "at the final output would misdiagnose this completely."
        ),
        severity=Severity.CRITICAL,
        start=start,
        duration=duration,
        expected_root_cause="Ingestion schema change cascading through every downstream stage",
        effects=(
            Effect(SignalRef(Stage.INGESTION, "schema_violations"), +9.0, 0.00),
            Effect(SignalRef(Stage.INGESTION, "null_pct"), +6.0, 0.02),
            Effect(SignalRef(Stage.VALIDATION, "validation_failures"), +7.0, 0.05),
            Effect(SignalRef(Stage.FEATURE_ENGINEERING, "feature_null_rate"), +6.0, 0.15),
            Effect(SignalRef(Stage.FEATURE_ENGINEERING, "feature_drift_score"), +5.0, 0.20),
            Effect(SignalRef(Stage.EMBEDDING, "embedding_anomaly_rate"), +5.0, 0.25),
            Effect(SignalRef(Stage.RETRIEVAL, "top_k_similarity"), -5.0, 0.35),
            Effect(SignalRef(Stage.RETRIEVAL, "context_relevance"), -5.0, 0.40),
            Effect(SignalRef(Stage.RETRIEVAL, "empty_retrieval_rate"), +4.0, 0.40),
            Effect(SignalRef(Stage.EVALUATION, "groundedness"), -5.0, 0.55),
            Effect(SignalRef(Stage.EVALUATION, "hallucination_rate"), +5.0, 0.60),
            Effect(SignalRef(Stage.EVALUATION, "eval_score"), -5.0, 0.60),
            Effect(SignalRef(Stage.DELIVERY, "delivery_failures"), +3.0, 0.70),
        ),
    )


def build_multi_factor(start: datetime, duration: timedelta) -> ScenarioSpec:
    """Two independent faults at once: retrieval regression + infra pressure."""
    return _spec(
        slug="multi-factor",
        name="Multi-factor failure",
        category=IncidentCategory.MULTI_FACTOR,
        description=(
            "Two unrelated faults land together: a retrieval regression and "
            "infrastructure memory pressure. Symptoms interleave; no single "
            "root cause explains all the evidence, and a partial diagnosis "
            "is the honest outcome."
        ),
        severity=Severity.CRITICAL,
        start=start,
        duration=duration,
        expected_root_cause=(
            "Two concurrent faults: retrieval configuration regression AND "
            "serving-infrastructure memory pressure"
        ),
        effects=(
            Effect(SignalRef(Stage.RETRIEVAL, "top_k_similarity"), -7.0, 0.00),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "memory_utilization_pct"), +7.0, 0.00),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "cpu_utilization_pct"), +6.0, 0.05),
            Effect(SignalRef(Stage.RETRIEVAL, "context_relevance"), -6.0, 0.05),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "llm_latency_ms"), +5.0, 0.10),
            Effect(SignalRef(Stage.MODEL_INFERENCE, "timeout_rate"), +4.0, 0.15),
            Effect(SignalRef(Stage.EVALUATION, "groundedness"), -5.0, 0.25),
            Effect(SignalRef(Stage.EVALUATION, "hallucination_rate"), +5.0, 0.30),
            Effect(SignalRef(Stage.EVALUATION, "eval_score"), -4.0, 0.35),
        ),
    )


SCENARIO_BUILDERS: dict[IncidentCategory, ScenarioBuilder] = {
    IncidentCategory.NORMAL: build_normal,
    IncidentCategory.SCHEMA_DRIFT: build_schema_drift,
    IncidentCategory.DATA_QUALITY_DEGRADATION: build_data_quality_degradation,
    IncidentCategory.FEATURE_DRIFT: build_feature_drift,
    IncidentCategory.RETRIEVAL_DEGRADATION: build_retrieval_degradation,
    IncidentCategory.LLM_LATENCY_SPIKE: build_llm_latency_spike,
    IncidentCategory.MODEL_REGRESSION: build_model_regression,
    IncidentCategory.PROMPT_REGRESSION: build_prompt_regression,
    IncidentCategory.API_RATE_LIMITING: build_api_rate_limiting,
    IncidentCategory.RESOURCE_EXHAUSTION: build_resource_exhaustion,
    IncidentCategory.CASCADING_FAILURE: build_cascading_failure,
    IncidentCategory.MULTI_FACTOR: build_multi_factor,
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
