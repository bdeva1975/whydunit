"""Root-cause hypothesis generation via category signatures.

Each :class:`Signature` declares what a fault category *looks like*:

* ``expected`` — signals that should move, with direction;
* ``quiet`` — signals that should NOT move (distinguishing absences:
  e.g. rate limiting moves errors but keeps CPU/memory flat);
* ``origin`` — the stage where the fault originates.

Scoring is deliberately plain arithmetic (documented weights, no
learning): observed expected movements add their weight; expected
movements that are absent subtract a miss penalty; quiet signals that
fired subtract; DAG consistency (origin can explain all affected
stages) and origin-rank agreement add. The result is a ranked list of
:class:`Hypothesis` objects carrying BOTH supporting and contradicting
evidence, mapped to qualitative confidence bands — never probabilities.

Weights (why these numbers): primary-signal match 3.0 — the fault's own
signal moving is the strongest single clue; secondary matches 1.0 each;
missing expected signal −0.75 (absence is weaker evidence than
presence: a detector can miss a small shift); quiet violation −1.5
(a signal that should be flat moving is strong counter-evidence);
DAG-explains-all +1.0 and origin-rank agreement +1.0 (structural
corroboration). Bands: coverage = matched/expected; STRONG needs
score ≥ 6 and coverage ≥ 0.7, MODERATE ≥ 3.5 and ≥ 0.5, WEAK ≥ 1.5,
else INSUFFICIENT. Multi-factor incidents naturally yield two MODERATE
hypotheses rather than one STRONG — which is the honest answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from whydunit.forensics.dependency import can_explain, rank_origin_candidates
from whydunit.models import (
    Anomaly,
    ConfidenceBand,
    Direction,
    Evidence,
    EvidenceKind,
    Hypothesis,
    IncidentCategory,
    SignalRef,
    Stage,
)

_W_PRIMARY = 3.0
_W_SECONDARY = 1.0
_W_MISS = 0.75
_W_QUIET_VIOLATION = 1.5
_W_DAG_CONSISTENT = 1.0
_W_ORIGIN_RANK = 1.0

_STRONG_SCORE, _STRONG_COVERAGE = 6.0, 0.7
_MODERATE_SCORE, _MODERATE_COVERAGE = 3.5, 0.5
_WEAK_SCORE = 1.5


@dataclass(frozen=True, slots=True)
class ExpectedMove:
    signal: SignalRef
    direction: Direction
    primary: bool = False


@dataclass(frozen=True, slots=True)
class Signature:
    """What one incident category looks like in the telemetry."""

    category: IncidentCategory
    origin: Stage
    statement: str
    next_step: str
    expected: tuple[ExpectedMove, ...]
    quiet: tuple[SignalRef, ...] = field(default=())


def _ref(stage: Stage, metric: str) -> SignalRef:
    return SignalRef(stage=stage, metric=metric)


SIGNATURES: tuple[Signature, ...] = (
    Signature(
        category=IncidentCategory.SCHEMA_DRIFT,
        origin=Stage.INGESTION,
        statement="An upstream schema change is corrupting data at the ingestion boundary",
        next_step="Diff the producer's schema/contract against the last known-good version",
        expected=(
            ExpectedMove(_ref(Stage.INGESTION, "schema_violations"), Direction.UP, primary=True),
            ExpectedMove(_ref(Stage.INGESTION, "null_pct"), Direction.UP),
            ExpectedMove(_ref(Stage.VALIDATION, "validation_failures"), Direction.UP),
            ExpectedMove(_ref(Stage.VALIDATION, "missing_fields_pct"), Direction.UP),
            ExpectedMove(_ref(Stage.FEATURE_ENGINEERING, "feature_null_rate"), Direction.UP),
        ),
    ),
    Signature(
        category=IncidentCategory.DATA_QUALITY_DEGRADATION,
        origin=Stage.INGESTION,
        statement="Source data quality degraded without a schema change",
        next_step="Sample rejected/duplicate rows and trace them to the producing system",
        expected=(
            ExpectedMove(_ref(Stage.INGESTION, "null_pct"), Direction.UP, primary=True),
            ExpectedMove(_ref(Stage.INGESTION, "records_rejected"), Direction.UP),
            ExpectedMove(_ref(Stage.VALIDATION, "duplicate_rate"), Direction.UP),
            ExpectedMove(_ref(Stage.VALIDATION, "missing_fields_pct"), Direction.UP),
            ExpectedMove(_ref(Stage.PREPROCESSING, "rows_dropped"), Direction.UP),
        ),
        quiet=(_ref(Stage.INGESTION, "schema_violations"),),
    ),
    Signature(
        category=IncidentCategory.FEATURE_DRIFT,
        origin=Stage.FEATURE_ENGINEERING,
        statement="Feature distributions drifted from what the model expects",
        next_step="Compare current feature distributions against the training snapshot",
        expected=(
            ExpectedMove(
                _ref(Stage.FEATURE_ENGINEERING, "feature_drift_score"), Direction.UP, primary=True
            ),
            ExpectedMove(_ref(Stage.EVALUATION, "eval_score"), Direction.DOWN),
            ExpectedMove(_ref(Stage.EVALUATION, "relevance"), Direction.DOWN),
        ),
        quiet=(
            _ref(Stage.INGESTION, "schema_violations"),
            _ref(Stage.INGESTION, "null_pct"),
        ),
    ),
    Signature(
        category=IncidentCategory.RETRIEVAL_DEGRADATION,
        origin=Stage.RETRIEVAL,
        statement="Retrieval quality collapsed; the LLM is answering on thin context",
        next_step="Check the vector index build/deploy history and retrieval configuration",
        expected=(
            ExpectedMove(_ref(Stage.RETRIEVAL, "top_k_similarity"), Direction.DOWN, primary=True),
            ExpectedMove(_ref(Stage.RETRIEVAL, "context_relevance"), Direction.DOWN),
            ExpectedMove(_ref(Stage.RETRIEVAL, "empty_retrieval_rate"), Direction.UP),
            ExpectedMove(_ref(Stage.EVALUATION, "groundedness"), Direction.DOWN),
            ExpectedMove(_ref(Stage.EVALUATION, "hallucination_rate"), Direction.UP),
        ),
    ),
    Signature(
        category=IncidentCategory.LLM_LATENCY_SPIKE,
        origin=Stage.MODEL_INFERENCE,
        statement="The LLM provider is slow; requests are timing out",
        next_step="Check the provider status page and per-request latency breakdown",
        expected=(
            ExpectedMove(_ref(Stage.MODEL_INFERENCE, "llm_latency_ms"), Direction.UP, primary=True),
            ExpectedMove(_ref(Stage.MODEL_INFERENCE, "timeout_rate"), Direction.UP),
            ExpectedMove(_ref(Stage.MODEL_INFERENCE, "throughput_rpm"), Direction.DOWN),
            ExpectedMove(_ref(Stage.DELIVERY, "delivery_latency_ms"), Direction.UP),
        ),
        quiet=(
            _ref(Stage.MODEL_INFERENCE, "cpu_utilization_pct"),
            _ref(Stage.MODEL_INFERENCE, "memory_utilization_pct"),
            _ref(Stage.EVALUATION, "eval_score"),
        ),
    ),
    Signature(
        category=IncidentCategory.MODEL_REGRESSION,
        origin=Stage.MODEL_INFERENCE,
        statement="A model rollout regressed answer quality with a healthy upstream",
        next_step="Check the model version/deployment history against the incident start",
        expected=(
            ExpectedMove(_ref(Stage.EVALUATION, "eval_score"), Direction.DOWN, primary=True),
            ExpectedMove(_ref(Stage.EVALUATION, "groundedness"), Direction.DOWN),
            ExpectedMove(_ref(Stage.EVALUATION, "hallucination_rate"), Direction.UP),
            ExpectedMove(_ref(Stage.EVALUATION, "relevance"), Direction.DOWN),
        ),
        quiet=(
            _ref(Stage.RETRIEVAL, "top_k_similarity"),
            _ref(Stage.RETRIEVAL, "context_relevance"),
            _ref(Stage.POSTPROCESSING, "parse_failure_rate"),
        ),
    ),
    Signature(
        category=IncidentCategory.PROMPT_REGRESSION,
        origin=Stage.MODEL_INFERENCE,
        statement="A prompt/template change altered output structure and cost",
        next_step="Diff the active prompt/template against the previous version",
        expected=(
            ExpectedMove(
                _ref(Stage.MODEL_INFERENCE, "tokens_per_request"), Direction.UP, primary=True
            ),
            ExpectedMove(_ref(Stage.POSTPROCESSING, "parse_failure_rate"), Direction.UP),
            ExpectedMove(_ref(Stage.POSTPROCESSING, "output_length_chars"), Direction.UP),
            ExpectedMove(_ref(Stage.EVALUATION, "relevance"), Direction.DOWN),
        ),
        quiet=(_ref(Stage.RETRIEVAL, "top_k_similarity"),),
    ),
    Signature(
        category=IncidentCategory.API_RATE_LIMITING,
        origin=Stage.MODEL_INFERENCE,
        statement="The model API is throttling; errors and retries dominate",
        next_step="Check provider quota/429 metrics and recent traffic growth",
        expected=(
            ExpectedMove(_ref(Stage.MODEL_INFERENCE, "error_rate"), Direction.UP, primary=True),
            ExpectedMove(_ref(Stage.MODEL_INFERENCE, "timeout_rate"), Direction.UP),
            ExpectedMove(_ref(Stage.MODEL_INFERENCE, "throughput_rpm"), Direction.DOWN),
            ExpectedMove(_ref(Stage.DELIVERY, "delivery_failures"), Direction.UP),
        ),
        quiet=(
            _ref(Stage.MODEL_INFERENCE, "cpu_utilization_pct"),
            _ref(Stage.MODEL_INFERENCE, "memory_utilization_pct"),
        ),
    ),
    Signature(
        category=IncidentCategory.RESOURCE_EXHAUSTION,
        origin=Stage.MODEL_INFERENCE,
        statement="Serving infrastructure is saturated (CPU/memory pressure)",
        next_step="Inspect host/container metrics, recent load growth, and memory leaks",
        expected=(
            ExpectedMove(
                _ref(Stage.MODEL_INFERENCE, "memory_utilization_pct"), Direction.UP, primary=True
            ),
            ExpectedMove(_ref(Stage.MODEL_INFERENCE, "cpu_utilization_pct"), Direction.UP),
            ExpectedMove(_ref(Stage.MODEL_INFERENCE, "llm_latency_ms"), Direction.UP),
            ExpectedMove(_ref(Stage.MODEL_INFERENCE, "timeout_rate"), Direction.UP),
            ExpectedMove(_ref(Stage.MODEL_INFERENCE, "throughput_rpm"), Direction.DOWN),
        ),
    ),
    Signature(
        category=IncidentCategory.CASCADING_FAILURE,
        origin=Stage.INGESTION,
        statement="An upstream data fault is cascading through every downstream stage",
        next_step="Freeze downstream rollbacks; fix the ingestion-boundary fault first",
        expected=(
            ExpectedMove(_ref(Stage.INGESTION, "schema_violations"), Direction.UP, primary=True),
            ExpectedMove(_ref(Stage.VALIDATION, "validation_failures"), Direction.UP),
            ExpectedMove(_ref(Stage.FEATURE_ENGINEERING, "feature_null_rate"), Direction.UP),
            ExpectedMove(_ref(Stage.EMBEDDING, "embedding_anomaly_rate"), Direction.UP),
            ExpectedMove(_ref(Stage.RETRIEVAL, "top_k_similarity"), Direction.DOWN),
            ExpectedMove(_ref(Stage.EVALUATION, "groundedness"), Direction.DOWN),
            ExpectedMove(_ref(Stage.DELIVERY, "delivery_failures"), Direction.UP),
        ),
    ),
)


def _anomaly_directions(anomalies: list[Anomaly]) -> dict[SignalRef, set[Direction]]:
    """Directions in which each signal was anomalous (any detector)."""
    moved: dict[SignalRef, set[Direction]] = {}
    for anomaly in anomalies:
        moved.setdefault(anomaly.signal, set()).add(anomaly.direction)
    return moved


def _score_signature(
    signature: Signature,
    moved: dict[SignalRef, set[Direction]],
    affected_stages: set[Stage],
    origin_ranked: list[Stage],
    hypothesis_id: str,
) -> Hypothesis:
    supporting: list[Evidence] = []
    contradicting: list[Evidence] = []
    score = 0.0
    matched = 0

    for index, move in enumerate(signature.expected):
        weight = _W_PRIMARY if move.primary else _W_SECONDARY
        directions = moved.get(move.signal)
        if directions and move.direction in directions:
            matched += 1
            score += weight
            supporting.append(
                Evidence(
                    id=f"{hypothesis_id}-E{index}",
                    kind=EvidenceKind.OBSERVED,
                    statement=(
                        f"`{move.signal}` moved {move.direction.value} as this fault predicts"
                    ),
                    signals=(move.signal,),
                    weight=weight,
                )
            )
        else:
            score -= _W_MISS
            contradicting.append(
                Evidence(
                    id=f"{hypothesis_id}-C{index}",
                    kind=EvidenceKind.DERIVED,
                    statement=f"`{move.signal}` did not move {move.direction.value} as expected",
                    signals=(move.signal,),
                    weight=_W_MISS,
                )
            )

    for index, ref in enumerate(signature.quiet):
        if ref in moved:
            score -= _W_QUIET_VIOLATION
            contradicting.append(
                Evidence(
                    id=f"{hypothesis_id}-Q{index}",
                    kind=EvidenceKind.OBSERVED,
                    statement=f"`{ref}` moved, but this fault predicts it stays flat",
                    signals=(ref,),
                    weight=_W_QUIET_VIOLATION,
                )
            )

    if affected_stages and can_explain(signature.origin, affected_stages):
        score += _W_DAG_CONSISTENT
        supporting.append(
            Evidence(
                id=f"{hypothesis_id}-DAG",
                kind=EvidenceKind.DERIVED,
                statement=(
                    f"A fault at `{signature.origin.value}` can reach every affected stage "
                    "through the pipeline graph"
                ),
                signals=(),
                weight=_W_DAG_CONSISTENT,
            )
        )

    if origin_ranked and origin_ranked[0] is signature.origin:
        score += _W_ORIGIN_RANK
        supporting.append(
            Evidence(
                id=f"{hypothesis_id}-ORIGIN",
                kind=EvidenceKind.DERIVED,
                statement=(
                    f"`{signature.origin.value}` is the most upstream anomalous stage "
                    "(anomalies propagate downstream)"
                ),
                signals=(),
                weight=_W_ORIGIN_RANK,
            )
        )

    coverage = matched / len(signature.expected) if signature.expected else 0.0
    if score >= _STRONG_SCORE and coverage >= _STRONG_COVERAGE:
        band = ConfidenceBand.STRONG
    elif score >= _MODERATE_SCORE and coverage >= _MODERATE_COVERAGE:
        band = ConfidenceBand.MODERATE
    elif score >= _WEAK_SCORE:
        band = ConfidenceBand.WEAK
    else:
        band = ConfidenceBand.INSUFFICIENT

    return Hypothesis(
        id=hypothesis_id,
        category=signature.category,
        statement=signature.statement,
        affected_stages=tuple(stage for stage in Stage if stage in affected_stages),
        supporting=tuple(supporting),
        contradicting=tuple(contradicting),
        score=round(score, 2),
        confidence=band,
        next_step=signature.next_step,
    )


def generate_hypotheses(anomalies: list[Anomaly]) -> list[Hypothesis]:
    """Rank every signature against the observed anomalies.

    No anomalies → a single strong "no incident" hypothesis. Otherwise
    all signatures are scored and returned best-first (INSUFFICIENT ones
    are dropped — they add noise, not information).
    """
    if not anomalies:
        return [
            Hypothesis(
                id="HYP-NORMAL",
                category=IncidentCategory.NORMAL,
                statement="No anomalies detected; the pipeline is operating normally",
                affected_stages=(),
                supporting=(),
                contradicting=(),
                score=1.0,
                confidence=ConfidenceBand.STRONG,
                next_step="No action needed; continue routine monitoring",
            )
        ]

    moved = _anomaly_directions(anomalies)
    affected_stages = {anomaly.signal.stage for anomaly in anomalies}
    origin_ranked = rank_origin_candidates(anomalies)

    hypotheses = [
        _score_signature(sig, moved, affected_stages, origin_ranked, f"HYP-{sig.category.value}")
        for sig in SIGNATURES
    ]
    ranked = [h for h in hypotheses if h.confidence is not ConfidenceBand.INSUFFICIENT]
    ranked.sort(key=lambda h: h.score, reverse=True)
    return ranked
