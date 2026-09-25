# Whydunit — Forensic Methodology

This document explains *how* the engine reasons, why each method was
chosen, every threshold's justification, and — just as important — what
the engine cannot do. All numbers below are defined in code
(`DetectorConfig`, `ChangePointConfig`, `CorrelationConfig`, and the
weight constants in `forensics/hypotheses.py`); this document explains
them, it does not duplicate their authority.

## 1. Anomaly detection

Two deterministic detectors run over every signal.

### Robust rolling z-score (`robust_zscore`)

For each point, the baseline is the **median** of the previous 120
samples (two hours at 1-minute telemetry) and the scale is the rolling
**MAD** (median absolute deviation, ×1.4826 for stddev equivalence),
both **shifted by one step** so a point is never judged against a
baseline containing itself.

- Why median/MAD, not mean/std: a fault inflates the mean and the
  standard deviation of any window it enters, desensitising detection of
  its own tail. Medians resist exactly this contamination.
- Threshold **5.0**: under normality, P(|z| ≥ 5) ≈ 6×10⁻⁷ per point —
  noise essentially never fires — while injected primary faults are
  8–10σ and knock-ons 3–7σ.
- Scale floor: near-constant or clipped series can have a rolling MAD of
  zero; the scale is floored at 10% of the series' global robust scale.
  The global scale is mildly contaminated by any incident, but at a 10%
  floor the bias is immaterial.

Catches: spikes and step changes. Misses: very slow drift (absorbed into
the rolling median) — which is the second detector's job.

### EWMA residual (`ewma`)

The baseline is an exponentially weighted moving average with a
**4-hour halflife**; the residual against it is normalised by the
rolling median absolute residual. A slow drift keeps out-running a
long-memory baseline, so the residual accumulates where the rolling
median would simply follow.

- Threshold **4.0**, lower than the z-score's 5.0: EWMA residuals are
  smoother, so an equally rare threshold sits lower.

### Window merging

Flagged points merge into anomaly *windows*: at least **3** flagged
samples (kills single-sample flukes), tolerating gaps of up to **2**
unflagged samples. One incident therefore yields one window per
signal/detector, not sixty point-alerts. Both detectors' windows are
kept — which detector fired is itself forensic information (sharp break
vs drift).

### Known limitation

Once an incident is underway, its early samples enter the (shifted)
scale windows and mildly desensitise later detection. Onset detection —
what forensics needs — is unaffected; sensitivity late inside a long
incident is reduced. Accepted.

## 2. Change-point detection

A two-window mean-shift scan answers a different question from anomaly
detection: **when did the level shift begin?**

At each candidate index, compare the mean of the **60 samples before**
with the mean of the **60 samples after**, normalised by the
*pre-window's* robust scale (so the shift cannot contaminate its own
denominator). Local maxima above **4.0** are reported; two reports on
one signal must be at least **30 samples apart**.

- Why not `ruptures`/PELT: one dependency fewer, O(n) via cumulative
  sums, deterministic, explainable in two sentences.
- Honest trade-off: tuned for step-plus-ramp shifts (deployments, config
  changes, our incidents). Very gradual drift smears the statistic; the
  EWMA detector owns that case.
- Recovery edges also produce change points (the level shifts back).
  These are real level shifts, correctly reported; symptom rendering
  prefers the shift whose direction agrees with the anomaly, so a
  down-moving fault quotes its drop, not its recovery.

## 3. Cross-signal correlation

Two steps, deliberately separable:

1. **Time clustering** — each signal's anomaly windows collapse to one
   [first-start, last-end] span; spans overlapping within **15 minutes**
   merge into clusters via interval sweep. Co-occurrence *proposes* a
   relationship.
2. **Correlation scoring** — over the