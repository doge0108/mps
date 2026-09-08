"""Bottom-up game features from the player models (stacking).

The game model gets, for each side, the lineup's summed expected hits / walks /
total bases / runs and the starter's expected earned runs, outs and strikeouts.
For training these come from *cross-fitted* player models (each fold predicted
by models that never saw it), so the game model learns how much to trust them
without leakage.  At prediction time the final player models produce the same
columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.build import feature_columns
from .base import MultiTargetBooster
from .player_model import BATTER_OBJECTIVES, PITCHER_OBJECTIVES

STACK_BAT = ["h", "hr", "bb", "tb", "r", "rbi", "so"]
STACK_PIT = ["so", "er", "outs", "h", "bb"]
STACK_ROUNDS = 400


def fold_labels(dates: pd.Series, seasons: pd.Series) -> pd.Series:
    """Fold = season when there are at least two seasons, otherwise two halves by date."""
    seasons = pd.Series(seasons).astype("Int64")
    if seasons.dropna().nunique() >= 2:
        return seasons.astype("float").fillna(-1).astype(int)
    cut = pd.to_datetime(dates).median()
    return (pd.to_datetime(dates) > cut).astype(int)


def cross_fit(feats: pd.DataFrame, objectives: dict[str, str], folds: pd.Series, max_rounds: int = STACK_ROUNDS,
              verbose: bool = False) -> pd.DataFrame:
    """Out-of-fold predictions for every row of ``feats`` (columns = objective targets)."""
    cols = feature_columns(feats)
    out = pd.DataFrame(index=feats.index, columns=list(objectives), dtype=float)
    for f in sorted(folds.unique()):
        train_idx = folds != f
        test_idx = folds == f
        if train_idx.sum() < 50 or test_idx.sum() == 0:
            continue
        if verbose:
            print(f"  cross-fit fold {f}: {int(train_idx.sum())} train rows -> {int(test_idx.sum())} rows")
        model = MultiTargetBooster(cols, objectives, max_rounds=max_rounds).fit(feats[train_idx])
        out.loc[test_idx, list(objectives)] = model.predict(feats[test_idx]).to_numpy()
    return out


def stack_from_predictions(bat_rows: pd.DataFrame, pit_rows: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player expectations into one row per game (``stk_`` columns).

    ``bat_rows``: game_pk, team_id, batting_order, bat_starter + STACK_BAT expectation columns.
    ``pit_rows``: game_pk, team_id + STACK_PIT expectation columns (starters only).
    ``games``: game_pk, home_team_id, away_team_id.
    """
    g = games[["game_pk", "home_team_id", "away_team_id"]].drop_duplicates("game_pk")
    starters = bat_rows[(bat_rows["bat_starter"] == 1) & bat_rows["batting_order"].between(1, 9)]
    lineup = starters.groupby(["game_pk", "team_id"], sort=False)[STACK_BAT].sum().reset_index()
    lineup["n"] = starters.groupby(["game_pk", "team_id"], sort=False).size().to_numpy()
    sp = pit_rows.groupby(["game_pk", "team_id"], sort=False)[STACK_PIT].first().reset_index()
    out = g.copy()
    for side in ("home", "away"):
        tid = f"{side}_team_id"
        lu = lineup.rename(columns={"team_id": tid, **{c: f"stk_{side}_{c}" for c in STACK_BAT}, "n": f"stk_{side}_n"})
        out = out.merge(lu, on=["game_pk", tid], how="left")
        ps = sp.rename(columns={"team_id": tid, **{c: f"stk_{side}_sp_{c}" for c in STACK_PIT}})
        out = out.merge(ps, on=["game_pk", tid], how="left")
    out["stk_r_gap"] = out["stk_home_r"] - out["stk_away_r"]
    out["stk_tb_gap"] = out["stk_home_tb"] - out["stk_away_tb"]
    out["stk_sp_er_gap"] = out["stk_home_sp_er"] - out["stk_away_sp_er"]
    out["stk_sp_so_gap"] = out["stk_home_sp_so"] - out["stk_away_sp_so"]
    # crude bottom-up run estimates: lineup runs vs. what the opposing starter allows over his outs
    for side, opp in (("home", "away"), ("away", "home")):
        sp_share = (out[f"stk_{opp}_sp_outs"] / 27.0).clip(0, 1)
        out[f"stk_{side}_runs_est"] = sp_share * out[f"stk_{opp}_sp_er"] + (1 - sp_share) * out[f"stk_{side}_r"] \
            + 0.5 * (out[f"stk_{side}_r"] - out[f"stk_{opp}_sp_er"] * sp_share)
    out["stk_total_est"] = out["stk_home_runs_est"] + out["stk_away_runs_est"]
    out["stk_est_gap"] = out["stk_home_runs_est"] - out["stk_away_runs_est"]
    return out.drop(columns=["home_team_id", "away_team_id"])


def oof_stack_features(bf: pd.DataFrame, pf: pd.DataFrame, games: pd.DataFrame, max_rounds: int = STACK_ROUNDS,
                       verbose: bool = False) -> pd.DataFrame:
    """Cross-fitted stack features for every game that appears in ``bf`` / ``pf``."""
    seasons = games.drop_duplicates("game_pk").set_index("game_pk")["season"]
    b_folds = fold_labels(bf["date"], bf["game_pk"].map(seasons))
    p_folds = fold_labels(pf["date"], pf["game_pk"].map(seasons))
    b_obj = {t: BATTER_OBJECTIVES[t] for t in STACK_BAT}
    p_obj = {t: PITCHER_OBJECTIVES[t] for t in STACK_PIT}
    b_pred = cross_fit(bf, b_obj, b_folds, max_rounds, verbose)
    p_pred = cross_fit(pf, p_obj, p_folds, max_rounds, verbose)
    bat_rows = pd.concat([bf[["game_pk", "team_id", "batting_order", "bat_starter"]], b_pred], axis=1)
    pit_rows = pd.concat([pf[["game_pk", "team_id"]], p_pred], axis=1)
    return stack_from_predictions(bat_rows, pit_rows, games)


def model_stack_features(batter: MultiTargetBooster, pitcher: MultiTargetBooster, bf: pd.DataFrame,
                         pf: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Stack features from already-fitted player models (for held-out or future games)."""
    b_pred = batter.predict(bf)[STACK_BAT]
    p_pred = pitcher.predict(pf)[STACK_PIT]
    bat_rows = pd.concat([bf[["game_pk", "team_id", "batting_order", "bat_starter"]].reset_index(drop=True),
                          b_pred.reset_index(drop=True)], axis=1)
    pit_rows = pd.concat([pf[["game_pk", "team_id"]].reset_index(drop=True), p_pred.reset_index(drop=True)], axis=1)
    return stack_from_predictions(bat_rows, pit_rows, games)


def attach_stack(game_feats: pd.DataFrame, stack: pd.DataFrame | None) -> pd.DataFrame:
    if stack is None or stack.empty:
        return game_feats
    return game_feats.merge(stack, on="game_pk", how="left")
