"""Error metrics and the cell sleep policy."""
from __future__ import annotations

import numpy as np


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = actual - pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err) / np.maximum(actual, 1e-8)) * 100),
    }


def sleep_policy(actual: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float]:
    """Switch the cell off for the next hour when the forecast load is below threshold.

    hours_asleep_%   : share of hours the cell is switched off (energy saved).
    oracle_asleep_%  : what a perfect forecast would have achieved.
    missed_%         : share of sleeping hours where real load was above threshold,
                       i.e. users were hurt by a wrong decision.
    """
    sleep = pred < threshold
    could_sleep = actual < threshold
    wrong = sleep & ~could_sleep
    return {
        "hours_asleep_%": float(sleep.mean() * 100),
        "oracle_asleep_%": float(could_sleep.mean() * 100),
        "missed_%": float(wrong.sum() / max(sleep.sum(), 1) * 100),
    }
