"""Shared LightGBM wrapper for multi-target count / probability prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS: dict[str, Any] = {
    "learning_rate": 0.06,
    "num_leaves": 31,
    "min_child_samples": 60,
    "feature_fraction": 0.5,
    "max_bin": 63,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": 4,
}


def _time_split(dates: pd.Series, valid_frac: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(dates.to_numpy(), kind="mergesort")
    n_valid = max(1, int(len(order) * valid_frac))
    return order[:-n_valid], order[-n_valid:]


class MultiTargetBooster:
    """One LightGBM booster per target sharing a feature list.

    ``objectives`` maps target -> LightGBM objective ("poisson", "regression", "binary").
    Early stopping uses the chronologically last ``valid_frac`` of rows.
    """

    def __init__(self, feature_cols: list[str], objectives: dict[str, str],
                 params: dict[str, Any] | None = None, max_rounds: int = 1500, valid_frac: float = 0.15):
        self.feature_cols = list(feature_cols)
        self.objectives = dict(objectives)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.max_rounds = max_rounds
        self.valid_frac = valid_frac
        self.boosters: dict[str, lgb.Booster] = {}
        self.best_iters: dict[str, int] = {}
        self.train_means: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, targets_prefix: str = "y_") -> "MultiTargetBooster":
        X = df[self.feature_cols].astype("float32")
        tr_idx, va_idx = _time_split(df["date"], self.valid_frac)
        for target, objective in self.objectives.items():
            y = df[f"{targets_prefix}{target}"].astype(float)
            self.train_means[target] = float(y.iloc[tr_idx].mean())
            dtrain = lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx], free_raw_data=False)
            dvalid = lgb.Dataset(X.iloc[va_idx], y.iloc[va_idx], reference=dtrain, free_raw_data=False)
            params = {**self.params, "objective": objective}
            if objective == "binary":
                params["metric"] = "binary_logloss"
            booster = lgb.train(
                params, dtrain, num_boost_round=self.max_rounds, valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(60, verbose=False)],
            )
            self.boosters[target] = booster
            self.best_iters[target] = int(booster.best_iteration or self.max_rounds)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # columns the model never saw are dropped; columns missing now (e.g. no Statcast yet) become NaN
        X = df.reindex(columns=self.feature_cols).astype("float32")
        out = {}
        for target, booster in self.boosters.items():
            out[target] = booster.predict(X, num_iteration=self.best_iters[target])
        return pd.DataFrame(out, index=df.index)

    def feature_importance(self, target: str, top: int = 20) -> pd.Series:
        booster = self.boosters[target]
        imp = pd.Series(booster.feature_importance("gain"), index=self.feature_cols)
        return imp.sort_values(ascending=False).head(top)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "MultiTargetBooster":
        return joblib.load(path)
