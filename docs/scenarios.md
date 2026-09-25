# Whydunit — Incident Scenarios

Twelve synthetic scenarios, one per `IncidentCategory`. Each is a recipe
of correlated, temporally ordered effects (see `simulator/scenarios.py`
for exact magnitudes and onsets). Magnitude convention: primary faults
8–10σ, first-order knock-ons 5–7σ, far-downstream quality effects 3–5σ
with later onsets — strong enough to detect reliably, weak enough that
*ordering*, not magnitude alone, carries the diagnosis.

Every scenario is reproducible:

```bash
python -m whydunit.simulator --scenario <category> --days 1 --seed 42 --output data/<category>
```

## Fingerprint table

What lets the engine tell near-neighbours apart:

| Scenario | Moves first | Distinguishing presence | Distinguishing absence |
|---|---|---|---|
| `normal` | nothing | — | any anomaly at all |
| `schema_drift` | ingestion.schema_violations ↑ | violations spike | — |
| `data_quality_degradation` | ingestion.null_pct ↑ | duplicates, rejects, drops | **schema violations stay flat** |
| `feature_drift` | feature_drift_score ↑ (slow ramp) | drift score leads quality decay | upstream data checks stay green |
| `retrieval_degradation` | retrieval.top_k_similarity ↓ | retrieval trio + groundedness/hallucination | upstream stages healthy |
| `llm_latency_spike` | llm_latency_ms ↑ | latency, timeouts, throughput, delivery lag | **CPU/memory flat, quality flat** |
| `model_regression` | tokens shift, then eval_score ↓ | all quality metrics degrade | **retrieval healthy, no parse failures** |
| `prompt_regression` | tokens_per_request ↑ | parse failures + output-length change | retrieval healthy |
| `api_rate_limiting` | error_rate ↑ | errors dominate, delivery failures | **CPU/memory flat** |
| `resource_exhaustion` | memory + CPU ↑ together | infra pair moves first, everything on the box suffers | — |
| `cascading_failure` | ingestion.schema_violations ↑ | staggered onsets down the whole chain | — |
| `multi_factor` | retrieval ↓ AND memory ↑ simultaneously | two unrelated symptom families | no single origin explains all evidence |

## Scenario notes

### 1. Normal operation
No fault injected; the correct diagnosis is "no incident". Exists so the
engine is scored on recognising health (false-positive control).

### 2. Schema drift at ingestion — HIGH
An upstream producer ships a schema change. Violations and nulls spike at
ingestion; validation failures follow; feature nulls rise; answer quality
sags minutes later. **Remediation:** diff the producer's contract against
last known-good; coordinate rollback or adapt the parser.

### 3. Data-quality degradation — MEDIUM
Dirty data without a schema change: nulls, duplicates, rejects, dropped
rows. The flat `schema_violations` signal is what separates this from
schema drift. **Remediation:** sample rejected rows, trace to the
producing system.

### 4. Feature drift — MEDIUM
The world changed; the model didn't. Drift score ramps slowly (0.4 ramp
fraction — this is the EWMA detector's showcase), quality decays behind
it, all upstream checks stay green. **Remediation:** compare live feature
d