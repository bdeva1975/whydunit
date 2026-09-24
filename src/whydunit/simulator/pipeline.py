"""Static definition of the synthetic AI pipeline.

Single source of truth for:

* which stages exist and how they depend on each other,
* which metrics each stage emits,
* each metric's healthy baseline, noise level, hard bounds, and which
  direction of movement is *concerning*.

The telemetry generator perturbs these baselines; the forensic engine
interprets deviations against them. Neither hardcodes metric names.
"""

from __future__ import annotations

from dataclasses import dataclass

from whydunit.models import Direction, SignalRef, Stage


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Healthy-state definition of one metric.

    ``baseline`` and ``noise_std`` drive the generator's Gaussian noise;
    ``low``/``high`` are hard clipping bounds (rates stay in [0, 1],
    counts stay non-negative). ``concern`` is the direction that signals
    trouble — ``None`` means a large move either way is suspicious
    (e.g. embedding vector norms).
    """

    name: str
    baseline: float
    noise_std: float
    low: float
    high: float
    unit: str
    concern: Direction | None


_M = MetricSpec

STAGE_METRICS: dict[Stage, tuple[MetricSpec, ...]] = {
    Stage.INGESTION: (
        _M("records_processed", 5000.0, 150.0, 0.0, 1e9, "records/interval", Direction.DOWN),
        _M("records_rejected", 25.0, 6.0, 0.0, 1e9, "records/interval", Direction.UP),
        _M("ingestion_latency_ms", 120.0, 15.0, 1.0, 1e6, "ms", Direction.UP),
        _M("schema_violations", 2.0, 1.2, 0.0, 1e9, "count/interval", Direction.UP),
        _M("null_pct", 0.5, 0.12, 0.0, 100.0, "%", Direction.UP),
    ),
    Stage.VALIDATION: (
        _M("validation_failures", 10.0, 3.0, 0.0, 1e9, "count/interval", Direction.UP),
        _M("duplicate_rate", 0.010, 0.003, 0.0, 1.0, "ratio", Direction.UP),
        _M("missing_fields_pct", 0.4, 0.10, 0.0, 100.0, "%", Direction.UP),
    ),
    Stage.PREPROCESSING: (
        _M("processing_latency_ms", 80.0, 10.0, 1.0, 1e6, "ms", Direction.UP),
        _M("rows_dropped", 15.0, 4.0, 0.0, 1e9, "count/interval", Direction.UP),
    ),
    Stage.FEATURE_ENGINEERING: (
        _M("fe_latency_ms", 140.0, 18.0, 1.0, 1e6, "ms", Direction.UP),
        _M("feature_null_rate", 0.008, 0.002, 0.0, 1.0, "ratio", Direction.UP),
        _M("feature_drift_score", 0.05, 0.015, 0.0, 1.0, "score", Direction.UP),
    ),
    Stage.EMBEDDING: (
        _M("embedding_latency_ms", 60.0, 8.0, 1.0, 1e6, "ms", Direction.UP),
        _M("vector_norm_mean", 1.00, 0.02, 0.0, 10.0, "norm", None),
        _M("embedding_anomaly_rate", 0.005, 0.0015, 0.0, 1.0, "ratio", Direction.UP),
    ),
    Stage.RETRIEVAL: (
        _M("top_k_similarity", 0.82, 0.02, 0.0, 1.0, "cosine", Direction.DOWN),
        _M("retrieval_latency_ms", 45.0, 6.0, 1.0, 1e6, "ms", Direction.UP),
        _M("empty_retrieval_rate", 0.010, 0.003, 0.0, 1.0, "ratio", Direction.UP),
        _M("context_relevance", 0.84, 0.02, 0.0, 1.0, "score", Direction.DOWN),
    ),
    Stage.MODEL_INFERENCE: (
        _M("llm_latency_ms", 900.0, 90.0, 1.0, 1e7, "ms", Direction.UP),
        _M("tokens_per_request", 750.0, 60.0, 1.0, 1e6, "tokens", Direction.UP),
        _M("error_rate", 0.006, 0.002, 0.0, 1.0, "ratio", Direction.UP),
        _M("timeout_rate", 0.002, 0.001, 0.0, 1.0, "ratio", Direction.UP),
        _M("throughput_rpm", 240.0, 18.0, 0.0, 1e6, "req/min", Direction.DOWN),
        _M("cpu_utilization_pct", 55.0, 5.0, 0.0, 100.0, "%", Direction.UP),
        _M("memory_utilization_pct", 60.0, 4.0, 0.0, 100.0, "%", Direction.UP),
    ),
    Stage.POSTPROCESSING: (
        _M("parse_failure_rate", 0.004, 0.0015, 0.0, 1.0, "ratio", Direction.UP),
        _M("output_length_chars", 1400.0, 120.0, 0.0, 1e6, "chars", None),
    ),
    Stage.EVALUATION: (
        _M("eval_score", 0.86, 0.015, 0.0, 1.0, "score", Direction.DOWN),
        _M("groundedness", 0.88, 0.015, 0.0, 1.0, "score", Direction.DOWN),
        _M("hallucination_rate", 0.030, 0.006, 0.0, 1.0, "ratio", Direction.UP),
        _M("relevance", 0.85, 0.015, 0.0, 1.0, "score", Direction.DOWN),
    ),
    Stage.DELIVERY: (
        _M("delivery_latency_ms", 30.0, 4.0, 1.0, 1e6, "ms", Direction.UP),
        _M("delivery_failures", 1.0, 0.8, 0.0, 1e9, "count/interval", Direction.UP),
    ),
}

PIPELINE_EDGES: tuple[tuple[Stage, Stage], ...] = tuple(
    zip(list(Stage)[:-1], list(Stage)[1:], strict=True)
)
"""Stage dependencies as (upstream, downstream) pairs.

v0.1 uses a linear chain matching ``Stage`` declaration order; the
dependency module treats it as a general DAG, so branching pipelines
only require edits here.
"""


def all_signals() -> tuple[SignalRef, ...]:
    """Every signal the pipeline emits, in stage order."""
    return tuple(
        SignalRef(stage=stage, metric=spec.name) for stage in Stage for spec in STAGE_METRICS[stage]
    )


def metric_spec(ref: SignalRef) -> MetricSpec:
    """Look up the spec for one signal. Raises ``KeyError`` if unknown."""
    for spec in STAGE_METRICS[ref.stage]:
        if spec.name == ref.metric:
            return spec
    raise KeyError(f"unknown metric {ref.metric!r} on stage {ref.stage.value!r}")
