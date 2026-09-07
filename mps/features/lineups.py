"""Lineup-level features: strength and handedness mix of the nine starters."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .states import asof_join

LINEUP_STATE_COLS = ["b_avg_r30", "b_slg_r30", "b_hr_rate_r30", "b_k_rate_r30", "b_bb_rate_r30", "b_form_slg"]


def starters_from_lines(batting_lines: pd.DataFrame) -> pd.DataFrame:
    """The nine starters of every (game, team) as recorded in the box score."""
    df = batting_lines
    starters = df[(df["bat_starter"] == 1) & df["batting_order"].between(1, 9)]
    return starters[["game_pk", "date", "team_id", "player_id", "batting_order"]].reset_index(drop=True)


def latest_lineup(batting_lines: pd.DataFrame, team_id: int, before: pd.Timestamp) -> pd.DataFrame:
    """The most recent starting nine a team used before ``before`` (fallback when nothing is announced)."""
    df = batting_lines[(batting_lines["team_id"] == team_id) & (batting_lines["date"] < before)]
    if df.empty:
        return pd.DataFrame(columns=["team_id", "player_id", "batting_order"])
    last_pk = df.sort_values(["date", "game_pk"])["game_pk"].iloc[-1]
    last = df[(df["game_pk"] == last_pk) & (df["bat_starter"] == 1) & df["batting_order"].between(1, 9)]
    return last[["team_id", "player_id", "batting_order"]].sort_values("batting_order").reset_index(drop=True)


def lineup_features(lineup_rows: pd.DataFrame, batter_state: pd.DataFrame,
                    players: pd.DataFrame | None, sc_batter_state: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate pre-game batter states over each lineup.

    ``lineup_rows``: game_pk, date, team_id, player_id.  Returns one row per (game_pk, team_id)
    with ``lu_`` columns; the caller merges them onto game/pitcher rows.
    """
    if lineup_rows.empty:
        return pd.DataFrame(columns=["game_pk", "team_id"])
    rows = lineup_rows[["game_pk", "date", "team_id", "player_id"]].copy()
    rows["date"] = pd.to_datetime(rows["date"])
    joined = asof_join(rows, batter_state[["player_id", "date"] + LINEUP_STATE_COLS], "player_id", "player_id")
    if sc_batter_state is not None and not sc_batter_state.empty:
        joined = asof_join(joined, sc_batter_state[["player_id", "date", "sc_xwoba_r30", "sc_ev_r30"]], "player_id", "player_id")
    else:
        joined["sc_xwoba_r30"] = np.nan
        joined["sc_ev_r30"] = np.nan
    if players is not None and not players.empty and "bats" in players.columns:
        joined = joined.merge(players[["player_id", "bats"]], on="player_id", how="left")
    else:
        joined["bats"] = np.nan
    joined["lhb"] = (joined["bats"] == "L").astype(float)
    joined["shb"] = (joined["bats"] == "S").astype(float)
    joined["known"] = joined["b_avg_r30"].notna().astype(float)
    agg = joined.groupby(["game_pk", "team_id"]).agg(
        lu_avg_r30=("b_avg_r30", "mean"), lu_slg_r30=("b_slg_r30", "mean"),
        lu_hr_rate_r30=("b_hr_rate_r30", "mean"), lu_k_rate_r30=("b_k_rate_r30", "mean"),
        lu_bb_rate_r30=("b_bb_rate_r30", "mean"), lu_form_slg=("b_form_slg", "mean"),
        lu_lhb_share=("lhb", "mean"), lu_shb_share=("shb", "mean"), lu_known=("known", "sum"),
        lu_size=("player_id", "count"), lu_xwoba_r30=("sc_xwoba_r30", "mean"), lu_ev_r30=("sc_ev_r30", "mean"),
    ).reset_index()
    return agg
