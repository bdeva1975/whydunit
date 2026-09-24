"""Core enumerations shared across Whydunit.

Every module in the simulator, forensic engine, and UI speaks in these
terms. Keeping them in one place prevents the string-literal drift that
plagues telemetry codebases.
"""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    """Pipeline stages, declared in topological (execution) order.

    ``list(Stage)`` therefore yields the canonical upstream-to-downstream
    ordering used by the dependency graph.
    """

    INGESTION = "ingestion"
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    MODEL_INFERENCE = "model_inference"
    POSTPROCESSING = "postprocessing"
    EVALUATION = "evaluation"
    DELIVERY = "delivery"


class IncidentCategory(str, Enum):
    """The classes of synthetic incident the simulator can inject.

    ``NORMAL`` is deliberately a category: the forensic engine must be
    able to conclude "nothing is wrong" and be scored on it.
    """

    NORMAL = "normal"
    SCHEMA_DRIFT = "schema_drift"
    DATA_QUALITY_DEGRADATION = "data_quality_degradation"
    FEATURE_DRIFT = "feature_drift"
    RETRIEVAL_DEGRADATION = "retrieval_degradation"
    LLM_LATENCY_SPIKE = "llm_latency_spike"
    MODEL_REGRESSION = "model_regression"
    PROMPT_REGRESSION = "prompt_regression"
    API_RATE_LIMITING = "api_rate_limiting"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CASCADING_FAILURE = "cascading_failure"
    MULTI_FACTOR = "multi_factor"


class Severity(str, Enum):
    """Incident severity. ``NONE`` applies to normal operation."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Direction(str, Enum):
    """Which way a signal moved relative to its baseline."""

    UP = "up"
    DOWN = "down"


class HealthState(str, Enum):
    """Per-stage health shown in the pipeline view."""

    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    FAILED = "failed"


class ConfidenceBand(str, Enum):
    """Qualitative confidence attached to a hypothesis.

    Deliberately not a probability: synthetic telemetry cannot prove
    causality, so the engine reports evidence strength, not certainty.
    """

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"


class EvidenceKind(str, Enum):
    """Epistemic status of a piece of evidence."""

    OBSERVED = "observed"
    DERIVED = "derived"
    CORRELATION = "correlation"