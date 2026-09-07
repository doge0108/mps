"""Game outcome model: home win probability and expected runs per side."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .base import MultiTargetBooster

GAME_OBJECTIVES = {"home_win": "binary", "home_runs": "poisson", "away_runs": "poisson"}
# Only a few thousand games per season, so keep the trees shallow and strongly regularised.
GAME_PARAMS = {"num_leaves": 6, "max_depth": 3, "min_child_samples": 150, "feature_fraction": 0.5,
               "learning_rate": 0.02, "lambda_l2": 15.0}


def make_game_model(feature_cols: list[str], **kw) -> MultiTargetBooster:
    kw.setdefault("params", GAME_PARAMS)
    kw.setdefault("max_rounds", 3000)
    return MultiTargetBooster(feature_cols, GAME_OBJECTIVES, **kw)


def summarise_game_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Add total runs, a run-based win probability, and a blended final probability."""
    out = pred.copy()
    out["total_runs"] = out["home_runs"] + out["away_runs"]
    out["run_line"] = out["home_runs"] - out["away_runs"]
    # Skellam-based P(home_runs > away_runs); ties split evenly as a tiebreak approximation
    hr, ar = out["home_runs"].clip(lower=0.05), out["away_runs"].clip(lower=0.05)
    p_home_more = 1.0 - sps.skellam.cdf(0, hr, ar)
    p_tie = sps.skellam.pmf(0, hr, ar)
    out["home_win_prob_runs"] = p_home_more + 0.5 * p_tie
    out["home_win_prob"] = 0.7 * out["home_win"] + 0.3 * out["home_win_prob_runs"]
    out["predicted_winner"] = np.where(out["home_win_prob"] >= 0.5, "home", "away")
    return out
