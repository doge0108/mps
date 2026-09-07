"""Leak-free rolling / expanding feature helpers.

Every helper shifts by one row *within each group* before aggregating, so the
feature for a game only uses games strictly before it.  Tables must be sorted
chronologically before calling these helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _check_sorted(df: pd.DataFrame) -> None:
    if not df["date"].is_monotonic_increasing:
        raise ValueError("frame must be sorted by date before computing rolling features")


def rolling_mean(df: pd.DataFrame, by: str, cols: list[str], window: int, prefix: str = "") -> pd.DataFrame:
    """Mean of the previous ``window`` rows per group (excluding the current row)."""
    _check_sorted(df)
    shifted = df.groupby(by, sort=False)[cols].shift(1)
    shifted[by] = df[by].to_numpy()
    out = shifted.groupby(by, sort=False)[cols].rolling(window, min_periods=1).mean()
    out = out.reset_index(level=0, drop=True).sort_index()
    out.columns = [f"{prefix}{c}_r{window}" for c in cols]
    return out


def expanding_sum(df: pd.DataFrame, by: str, cols: list[str], prefix: str = "") -> pd.DataFrame:
    """Sum of all previous rows per group (excluding the current row)."""
    _check_sorted(df)
    shifted = df.groupby(by, sort=False)[cols].shift(1)
    shifted[by] = df[by].to_numpy()
    out = shifted.groupby(by, sort=False)[cols].cumsum()
    out.columns = [f"{prefix}{c}_cum" for c in cols]
    return out


def prior_count(df: pd.DataFrame, by: str, name: str) -> pd.Series:
    """Number of previous rows per group."""
    return df.groupby(by, sort=False).cumcount().rename(name)


def days_since_last(df: pd.DataFrame, by: str, name: str) -> pd.Series:
    prev = df.groupby(by, sort=False)["date"].shift(1)
    return (df["date"] - prev).dt.days.rename(name)


def safe_div(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> pd.Series:
    num = pd.Series(np.asarray(num, dtype=float))
    den = pd.Series(np.asarray(den, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return out.where(den > 0, np.nan)
