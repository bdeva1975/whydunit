"""Statistical anomaly detection over wide-form telemetry.

Two deliberately simple, explainable detectors:

* ``robust_zscore`` — rolling median/MAD z-score. The window is shifted
  one step so a point is never judged against a baseline containing
  itself, and MAD (unlike stddev) resists the incident inflating its own
  scale. Catches spikes and step changes.
* ``ewma`` — residual against a long-memory EWMA baseline (halflife
  measured in hours), normalised by the rolling median absolute
  residual. The long memory means slow drifts keep diverging from the
  baseline instead of being absorbed into it.

Known, accepted limitation: once an incident is underway, its early
samples enter the (shifted) scale windows and mildly desensitise later
detection. Onset detection — what forensics needs — is unaffected.

No Isolation Forest or learned models here: on ~35 univariate series,
robust statistics are transparent, deterministic, and fast, and every
threshold can be justified in a sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from whydunit.models import Anomaly, Direction, SignalRef

_MAD_TO_STD = 1.4826  # scales MAD to stddev-equivalent under normality
_EPS = 1e-9

ROBUST_ZSCORE = "robust_zscore"
EWMA = "ewma"


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Tunable knobs, with defaults sized for 1-minute telemetry.

    * ``zscore_window`` 120 → a two-hour local baseline;
    * ``zscore_threshold`` 5.0 → far above noise (P(|z|>5) ≈ 6e-7 per
      point under normality) yet half the size of our primary faults;
    * ``ewma_halflife`` 240 → a four-hour memory that slow drifts must
      keep out-running;
    * ``min_points`` 3 → a window must contain at least three flagged
      samples, killing single-sample flukes;
    * ``max_gap`` 2 → up to two unflagged samples may sit inside one
      window before it splits.
    """

    zscore_window: int = 120
    zscore_threshold: float = 5.0
    ewma_halflife: int = 240
    ewma_scale_window: int = 240
    ewma_threshold: float = 4.0
    min_points: int = 3
    max_gap: int = 2


def robust_zscores(series: pd.Series, window: int = 120) -> pd.Series:
    """Rolling robust z-score of a series against its own recent past."""
    baseline = series.rolling(window, min_periods=window // 2).median().shift(1)
    abs_dev = (series - baseline).abs()
    scale = abs_dev.rolling(window, min_periods=window // 2).median().shift(1) * _MAD_TO_STD

    # Floor the scale: heavily clipped or near-constant series can have a
    # rolling MAD of zero. The global robust scale is contaminated by any
    # incident, but as a 10% floor that bias is immaterial.
    global_scale = _MAD_TO_STD * float(np.nanmedian(np.abs(series - np.nanmedian(series))))
    scale = scale.clip(lower=max(0.1 * global_scale, _EPS))
    return (series - baseline) / scale


def ewma_zscores(series: pd.Series, halflife: int = 240, scale_window: int = 240) -> pd.Series:
    """Residual z-score against a long-memory EWMA baseline."""
    baseline = series.ewm(halflife=halflife, min_periods=halflife).mean().shift(1)
    residual = series - baseline
    scale = (
        residual.abs().rolling(scale_window, min_periods=scale_window // 2).median().shift(1)
        * _MAD_TO_STD
    )
    global_scale = _MAD_TO_STD * float(np.nanmedian(np.abs(series - np.nanmedian(series))))
    scale = scale.clip(lower=max(0.1 * global_scale, _EPS))
    return residual / scale


def _flag_windows(flags: np.ndarray, max_gap: int, min_points: int) -> list[tuple[int, int]]:
    """Group flagged positions into (start, end) index windows."""
    positions = np.flatnonzero(flags)
    if positions.size == 0:
        return []

    windows: list[tuple[int, int]] = []
    start = prev = int(positions[0])
    count = 1
    for raw in positions[1:]:
        pos = int(raw)
        if pos - prev <= max_gap + 1:
            prev = pos
            count += 1
        else:
            if count >= min_points:
                windows.append((start, prev))
            start = prev = pos
            count = 1
    if count >= min_points:
        windows.append((start, prev))
    return windows


def _detect_in_series(
    zscores: pd.Series,
    ref: SignalRef,
    method: str,
    threshold: float,
    min_points: int,
    max_gap: int,
) -> list[Anomaly]:
    z = zscores.to_numpy()
    with np.errstate(invalid="ignore"):
        flags = np.abs(z) >= threshold  # NaN compares False: warm-up is never flagged

    anomalies: list[Anomaly] = []
    for start_pos, end_pos in _flag_windows(flags, max_gap=max_gap, min_points=min_points):
        segment = z[start_pos : end_pos + 1]
        anomalies.append(
            Anomaly(
                signal=ref,
                start=zscores.index[start_pos].to_pydatetime(),
                end=zscores.index[end_pos].to_pydatetime(),
                method=method,
                score=float(np.nanmax(np.abs(segment))),
                direction=Direction.UP if float(np.nansum(segment)) > 0 else Direction.DOWN,
            )
        )
    return anomalies


def detect_anomalies(wide: pd.DataFrame, config: DetectorConfig | None = None) -> list[Anomaly]:
    """Run both detectors over every signal of a wide telemetry frame.

    ``wide`` is the ``to_wide`` shape: UTC timestamp index, one
    ``stage.metric`` column per signal. Both detectors' findings are
    returned (deliberately not deduplicated — which detector fired is
    itself forensic information), sorted by start time.
    """
    cfg = config or DetectorConfig()
    anomalies: list[Anomaly] = []
    for column in wide.columns:
        ref = SignalRef.parse(column)
        series = wide[column]
        anomalies.extend(
            _detect_in_series(
                robust_zscores(series, window=cfg.zscore_window),
                ref,
                ROBUST_ZSCORE,
                cfg.zscore_threshold,
                cfg.min_points,
                cfg.max_gap,
            )
        )
        anomalies.extend(
            _detect_in_series(
                ewma_zscores(
                    series, halflife=cfg.ewma_halflife, scale_window=cfg.ewma_scale_window
                ),
                ref,
                EWMA,
                cfg.ewma_threshold,
                cfg.min_points,
                cfg.max_gap,
            )
        )
    anomalies.sort(key=lambda anomaly: (anomaly.start, str(anomaly.signal), anomaly.method))
    return anomalies
