"""Per-game player stat models (batters and starting pitchers)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..config import BATTING_TARGETS, PITCHING_TARGETS
from .base import MultiTargetBooster

BATTER_OBJECTIVES = {t: "poisson" for t in BATTING_TARGETS}
PITCHER_OBJECTIVES = {t: "poisson" for t in PITCHING_TARGETS}


def make_batter_model(feature_cols: list[str], **kw) -> MultiTargetBooster:
    return MultiTargetBooster(feature_cols, BATTER_OBJECTIVES, **kw)


def make_pitcher_model(feature_cols: list[str], **kw) -> MultiTargetBooster:
    return MultiTargetBooster(feature_cols, PITCHER_OBJECTIVES, **kw)


def poisson_at_least(mean: float, k: int) -> float:
    """P(X >= k) for a Poisson(mean) count."""
    if mean <= 0:
        return 0.0 if k > 0 else 1.0
    return float(sps.poisson.sf(k - 1, mean))


def add_probabilities(pred: pd.DataFrame, stats: list[str] | None = None) -> pd.DataFrame:
    """Append P(stat >= 1) and P(stat >= 2) columns derived from Poisson means."""
    stats = stats or [c for c in pred.columns if c not in ("ab", "outs")]
    out = pred.copy()
    for s in stats:
        mu = out[s].clip(lower=1e-6)
        out[f"p_{s}_ge1"] = sps.poisson.sf(0, mu)
        out[f"p_{s}_ge2"] = sps.poisson.sf(1, mu)
    return out
