"""Change-point detection: when did a signal's level shift?

Method: a two-window mean-shift scan. At each candidate index t, compare
the mean of the ``window`` samples before t with the mean of the
``window`` samples from t onward, normalised by the *pre-window's*
robust scale (MAD). Local maxima of that statistic above ``threshold``
are reported as change points.

Why this and not ``ruptures``/PELT: one dependency fewer, O(n) with
cumulative sums, deterministic, and fully explainable in two sentences.
The trade-off, stated honestly: it is tuned for step-plus-ramp shifts
(what our incidents produce and what deployments/config changes look
like in real telemetry); very gradual drift smears the statistic — the
EWMA anomaly detector owns that case.

Normalising by the pre-window keeps the scale uncontaminated by the
shift itself, at the cost of over-sensitivity on near-constant series —
mitigated by the same global-scale floor the anomaly module uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from whydunit.models import ChangePoint, SignalRef

_MAD_TO_STD = 1.4826
_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class ChangePointConfig:
    """Knobs, sized for 1-minute telemetry.

    * ``window`` 60 → one hour on each side of the candidate point;
    * ``threshold`` 4.0 → the two window means must differ by ≥4 robust
      stddevs of the pre-window — noise rarely does that, our incidents
      (≥3σ shifts sustained) comfortably do;
    * ``min_separation`` 30 → two reported change points on one signal
      must be at least half an hour apart (one shift, one report).
    """

    window: int = 60
    threshold: float = 4.0
    min_separation: int = 30


def shift_statistic(series: pd.Series, window: int = 60) -> pd.Series:
    """|mean(after) - mean(before)| / robust_scale(before), per index.

    NaN where either window is incomplete (start/end of the series).
    """
    values = series.to_numpy(dtype=float)
    n = values.size
    stat = np.full(n, np.nan)
    if n < 2 * window:
        return pd.Series(stat, index=series.index)

    csum = np.concatenate(([0.0], np.cumsum(values)))
    # means of [t-window, t) and [t, t+window) for t in [window, n-window]
    t = np.arange(window, n - window + 1)
    before_mean = (csum[t] - csum[t - window]) / window
    after_mean = (csum[t + window] - csum[t]) / window

    global_scale = _MAD_TO_STD * float(np.nanmedian(np.abs(values - np.nanmedian(values))))
    scale_floor = max(0.1 * global_scale, _EPS)

    series_pd = pd.Series(values)
    rolling_mad = (
        (series_pd - series_pd.rolling(window).median()).abs().rolling(window).median().to_numpy()
    )
    before_scale = np.maximum(rolling_mad[t - 1] * _MAD_TO_STD, scale_floor)

    stat[t] = np.abs(after_mean - before_mean) / before_scale
    return pd.Series(stat, index=series.index)


def _local_maxima_above(stat: np.ndarray, threshold: float, min_separation: int) -> list[int]:
    """Indices of local maxima of ``stat`` above ``threshold``, greedily
    keeping the largest and suppressing neighbours within ``min_separation``."""
    with np.errstate(invalid="ignore"):
        candidates = np.flatnonzero(stat >= threshold)
    if candidates.size == 0:
        return []

    order = candidates[np.argsort(stat[candidates])[::-1]]
    kept: list[int] = []
    for pos in order:
        if all(abs(pos - existing) >= min_separation for existing in kept):
            kept.append(int(pos))
    return sorted(kept)


def detect_changepoints(
    wide: pd.DataFrame, config: ChangePointConfig | None = None
) -> list[ChangePoint]:
    """Scan every signal of a wide telemetry frame for level shifts.

    Returns change points sorted by time. ``relative_delta`` is the
    before/after mean change relative to |before| (floored), so +0.5
    reads as "the level rose 50%".
    """
    cfg = config or ChangePointConfig()
    found: list[ChangePoint] = []

    for column in wide.columns:
        ref = SignalRef.parse(column)
        series = wide[column]
        stat = shift_statistic(series, window=cfg.window).to_numpy()

        for pos in _local_maxima_above(stat, cfg.threshold, cfg.min_separation):
            values = series.to_numpy(dtype=float)
            before_mean = float(values[pos - cfg.window : pos].mean())
            after_mean = float(values[pos : pos + cfg.window].mean())
            denom = max(abs(before_mean), _EPS)
            found.append(
                ChangePoint(
                    signal=ref,
                    at=series.index[pos].to_pydatetime(),
                    before_mean=before_mean,
                    after_mean=after_mean,
                    relative_delta=(after_mean - before_mean) / denom,
                )
            )

    found.sort(key=lambda cp: (cp.at, str(cp.signal)))
    return found
