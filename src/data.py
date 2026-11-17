"""Data loading for cellular traffic forecasting.

Two sources, both returning the same shape:
    DataFrame indexed by hourly timestamps, one column per cell, values = traffic load.

1. load_milan()      -> real Telecom Italia "sms-call-internet-mi" files (Harvard Dataverse).
2. make_synthetic()  -> generated traffic with daily/weekly seasonality, used to test the
                        pipeline before the real data is downloaded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Columns of the raw Milan files (tab separated, no header).
MILAN_COLUMNS = [
    "cell_id", "time_ms", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet",
]


def load_milan(
    data_dir: str | Path,
    cells: list[int] | None = None,
    activity: str = "internet",
) -> pd.DataFrame:
    """Read every sms-call-internet-mi-*.txt file in data_dir and aggregate to hourly load.

    Raw rows are per (cell, 10-minute slot, country code). We sum over country codes and
    over the six 10-minute slots of each hour.
    """
    files = sorted(Path(data_dir).glob("sms-call-internet-mi-*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No Milan files in {data_dir}. See README.md for the download steps."
        )

    parts = []
    for f in files:
        df = pd.read_csv(
            f, sep="\t", header=None, names=MILAN_COLUMNS,
            usecols=["cell_id", "time_ms", activity],
        )
        if cells is not None:
            df = df[df["cell_id"].isin(cells)]
        df = df.dropna(subset=[activity])
        # Timestamps are UTC; Milan is UTC+1 over the whole dataset (Nov-Dec, no DST).
        local = pd.to_datetime(df["time_ms"], unit="ms") + pd.Timedelta(hours=1)
        df["time"] = local.dt.floor("h")
        parts.append(df.groupby(["time", "cell_id"])[activity].sum())

    series = pd.concat(parts).groupby(level=["time", "cell_id"]).sum()
    wide = series.unstack("cell_id").sort_index()
    wide = wide.asfreq("h").interpolate(limit_direction="both")
    wide.columns = [f"cell_{c}" for c in wide.columns]
    return wide


def make_synthetic(n_cells: int = 5, n_days: int = 42, seed: int = 0) -> pd.DataFrame:
    """Generate hourly traffic that behaves like a real cell.

    Each cell has: a daily profile (quiet at night, busy in the evening), a weekend
    effect, a slow trend, random bursts, and noise.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2013-11-01", periods=n_days * 24, freq="h")
    hour = np.asarray(index.hour)
    weekend = np.asarray(index.dayofweek >= 5)
    t = np.arange(len(index))

    data = {}
    for c in range(n_cells):
        base = rng.uniform(80, 400)
        # Two bumps: late morning and evening peak. Peak hours differ per cell.
        morning = np.exp(-0.5 * ((hour - rng.uniform(10, 12)) / 2.5) ** 2)
        evening = np.exp(-0.5 * ((hour - rng.uniform(19, 21)) / 2.0) ** 2)
        daily = 0.15 + 0.6 * morning + rng.uniform(0.7, 1.1) * evening
        weekly = np.where(weekend, rng.uniform(0.6, 1.2), 1.0)
        trend = 1 + rng.uniform(-0.05, 0.15) * t / len(t)
        bursts = np.where(rng.random(len(t)) < 0.01, rng.uniform(1.3, 2.0, len(t)), 1.0)
        noise = rng.normal(1.0, 0.08, len(t))
        data[f"cell_{c + 1}"] = np.clip(base * daily * weekly * trend * bursts * noise, 0, None)

    return pd.DataFrame(data, index=index)


def train_test_split(df: pd.DataFrame, test_days: int = 7):
    """Last test_days days are the test set; everything before is training."""
    cut = len(df) - test_days * 24
    return df.iloc[:cut], df.iloc[cut:]
