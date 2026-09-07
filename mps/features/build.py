"""Assemble model-ready feature matrices for batters, pitchers and games."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import BATTING_TARGETS, PITCHING_TARGETS
from ..data.store import Dataset
from .states import asof_join, batter_state, pitcher_state, team_state

META_COLS = {"game_pk", "date", "player_id", "player_name", "team_id", "opp_team_id", "opp_sp_id",
             "season", "home_team_id", "away_team_id", "home_sp_id", "away_sp_id", "venue_id",
             "status", "game_type", "home_score", "away_score"}


@dataclass
class States:
    batters: pd.DataFrame
    pitchers: pd.DataFrame
    teams: pd.DataFrame

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "States":
        return cls(
            batters=batter_state(ds.batting_lines),
            pitchers=pitcher_state(ds.pitching_lines),
            teams=team_state(ds.games, ds.pitching_lines),
        )


def _drop_state_meta(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c.endswith("_last_date") or c.endswith("_name") or c.endswith("_team_id")
            and c not in ("team_id", "opp_team_id")]
    return df.drop(columns=drop)


def _finalise(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """Convert last-date columns to rest days and drop non-numeric helper columns."""
    for col in date_cols:
        if col in df.columns:
            new = col.replace("_last_date", "_rest_days")
            df[new] = (df["date"] - pd.to_datetime(df[col])).dt.days.astype("float")
    return _drop_state_meta(df)


# ---------------------------------------------------------------- batters
def assemble_batter_features(spec: pd.DataFrame, states: States) -> pd.DataFrame:
    """``spec`` rows: date, player_id, team_id, opp_team_id, opp_sp_id, is_home, batting_order."""
    df = spec.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = asof_join(df, states.batters, "player_id", "player_id")
    df = asof_join(df, states.pitchers, "opp_sp_id", "player_id", prefix="opp_")
    df = asof_join(df, states.teams, "team_id", "team_id", prefix="own_")
    df = asof_join(df, states.teams, "opp_team_id", "team_id", prefix="opp_")
    df["month"] = df["date"].dt.month
    df["is_home"] = df["is_home"].astype(float)
    df["batting_order"] = pd.to_numeric(df["batting_order"], errors="coerce").astype(float)
    df["park_factor"] = np.where(df["is_home"] == 1, df["own_tm_park_factor"], df["opp_tm_park_factor"])
    df["opp_sp_known"] = df["opp_p_apps"].notna().astype(float)
    return _finalise(df, ["b_last_date", "opp_p_last_date", "own_tm_last_date", "opp_tm_last_date"])


def build_batter_training(ds: Dataset, states: States | None = None) -> pd.DataFrame:
    states = states or States.from_dataset(ds)
    lines = ds.batting_lines
    spec = lines[["game_pk", "date", "player_id", "player_name", "team_id", "opp_team_id", "opp_sp_id",
                  "is_home", "batting_order"]].copy()
    feats = assemble_batter_features(spec, states)
    for t in BATTING_TARGETS:
        feats[f"y_{t}"] = lines[t].to_numpy()
    feats["y_pa"] = lines["pa"].to_numpy()
    return feats


# --------------------------------------------------------------- pitchers
def assemble_pitcher_features(spec: pd.DataFrame, states: States) -> pd.DataFrame:
    """``spec`` rows: date, player_id, team_id, opp_team_id, is_home."""
    df = spec.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = asof_join(df, states.pitchers, "player_id", "player_id")
    df = asof_join(df, states.teams, "team_id", "team_id", prefix="own_")
    df = asof_join(df, states.teams, "opp_team_id", "team_id", prefix="opp_")
    df["month"] = df["date"].dt.month
    df["is_home"] = df["is_home"].astype(float)
    df["park_factor"] = np.where(df["is_home"] == 1, df["own_tm_park_factor"], df["opp_tm_park_factor"])
    return _finalise(df, ["p_last_date", "own_tm_last_date", "opp_tm_last_date"])


def build_pitcher_training(ds: Dataset, states: States | None = None, starters_only: bool = True) -> pd.DataFrame:
    states = states or States.from_dataset(ds)
    lines = ds.pitching_lines
    if starters_only:
        lines = lines[lines["is_starter"] == 1]
    lines = lines.reset_index(drop=True)
    spec = lines[["game_pk", "date", "player_id", "player_name", "team_id", "opp_team_id", "is_home"]].copy()
    feats = assemble_pitcher_features(spec, states)
    for t in PITCHING_TARGETS:
        feats[f"y_{t}"] = lines[t].to_numpy()
    return feats


# ------------------------------------------------------------------ games
def assemble_game_features(spec: pd.DataFrame, states: States) -> pd.DataFrame:
    """``spec`` rows: date, home_team_id, away_team_id, home_sp_id, away_sp_id."""
    df = spec.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = asof_join(df, states.teams, "home_team_id", "team_id", prefix="home_")
    df = asof_join(df, states.teams, "away_team_id", "team_id", prefix="away_")
    df = asof_join(df, states.pitchers, "home_sp_id", "player_id", prefix="hsp_")
    df = asof_join(df, states.pitchers, "away_sp_id", "player_id", prefix="asp_")
    df["month"] = df["date"].dt.month
    df["elo_diff"] = df["home_tm_elo"] - df["away_tm_elo"]
    df["elo_home_prob"] = 1.0 / (1.0 + 10 ** ((df["away_tm_elo"] - (df["home_tm_elo"] + 24.0)) / 400.0))
    for w in (10, 30, 162):
        df[f"run_diff_gap_r{w}"] = df[f"home_tm_run_diff_r{w}"] - df[f"away_tm_run_diff_r{w}"]
        df[f"win_gap_r{w}"] = df[f"home_tm_win_r{w}"] - df[f"away_tm_win_r{w}"]
    for w in (10, 30):
        df[f"sp_era_gap_r{w}"] = df[f"hsp_p_era_r{w}"] - df[f"asp_p_era_r{w}"]
        df[f"sp_k_gap_r{w}"] = df[f"hsp_p_k_rate_r{w}"] - df[f"asp_p_k_rate_r{w}"]
    df["home_sp_known"] = df["hsp_p_apps"].notna().astype(float)
    df["away_sp_known"] = df["asp_p_apps"].notna().astype(float)
    return _finalise(df, ["home_tm_last_date", "away_tm_last_date", "hsp_p_last_date", "asp_p_last_date"])


def build_game_training(ds: Dataset, states: States | None = None) -> pd.DataFrame:
    states = states or States.from_dataset(ds)
    games = ds.games[ds.games["home_score"].notna() & ds.games["away_score"].notna()].reset_index(drop=True)
    spec = games[["game_pk", "date", "season", "home_team_id", "away_team_id", "home_sp_id", "away_sp_id"]].copy()
    feats = assemble_game_features(spec, states)
    feats["y_home_win"] = (games["home_score"] > games["away_score"]).astype(int).to_numpy()
    feats["y_home_runs"] = games["home_score"].astype(float).to_numpy()
    feats["y_away_runs"] = games["away_score"].astype(float).to_numpy()
    feats["y_total_runs"] = feats["y_home_runs"] + feats["y_away_runs"]
    return feats


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric non-target, non-metadata columns."""
    cols = []
    for c in df.columns:
        if c in META_COLS or c.startswith("y_") or c.startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols
