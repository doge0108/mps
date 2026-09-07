"""Assemble model-ready feature matrices for batters, pitchers and games."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BATTING_TARGETS, PITCHING_TARGETS
from ..data.store import Dataset
from .lineups import lineup_features, starters_from_lines
from .states import (asof_join, batter_split_state, batter_state, hand_code, pitcher_state, split_key,
                     team_state)

META_COLS = {"game_pk", "date", "player_id", "player_name", "team_id", "opp_team_id", "opp_sp_id",
             "season", "home_team_id", "away_team_id", "home_sp_id", "away_sp_id", "venue_id",
             "status", "game_type", "home_score", "away_score", "day_night", "wind_dir", "condition",
             "bats", "opp_hand", "throws"}

LINEUP_COLS = ["lu_avg_r30", "lu_slg_r30", "lu_hr_rate_r30", "lu_k_rate_r30", "lu_bb_rate_r30",
               "lu_form_slg", "lu_lhb_share", "lu_shb_share", "lu_known", "lu_size"]


@dataclass
class States:
    batters: pd.DataFrame
    pitchers: pd.DataFrame
    teams: pd.DataFrame
    splits: pd.DataFrame
    players: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "States":
        return cls(
            batters=batter_state(ds.batting_lines),
            pitchers=pitcher_state(ds.pitching_lines),
            teams=team_state(ds.games, ds.pitching_lines),
            splits=batter_split_state(ds.batting_lines, ds.players),
            players=ds.players,
        )

    def hand(self, player_ids: pd.Series, col: str) -> pd.Series:
        """Look up bats/throws for a series of player ids (NaN when unknown)."""
        if self.players is None or self.players.empty or col not in self.players.columns:
            return pd.Series(np.nan, index=player_ids.index, dtype="object")
        lookup = self.players.set_index("player_id")[col]
        return pd.to_numeric(player_ids, errors="coerce").map(lookup)


def _drop_state_meta(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c.endswith("_last_date") or c.endswith("_name")
            or (c.endswith("_team_id") and c not in ("team_id", "opp_team_id", "home_team_id", "away_team_id"))]
    return df.drop(columns=drop)


def _finalise(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """Convert last-date columns to rest days and drop non-numeric helper columns."""
    for col in date_cols:
        if col in df.columns:
            new = col.replace("_last_date", "_rest_days")
            df[new] = (df["date"] - pd.to_datetime(df[col])).dt.days.astype("float")
    return _drop_state_meta(df)


def weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric weather encoding from the games table columns (NaN-safe)."""
    out = pd.DataFrame(index=df.index)
    out["wx_temp_f"] = pd.to_numeric(df.get("temp_f"), errors="coerce").astype(float)
    out["wx_wind_mph"] = pd.to_numeric(df.get("wind_mph"), errors="coerce").astype(float)
    wind_dir = df.get("wind_dir", pd.Series(np.nan, index=df.index)).astype("string").str.lower()
    cond = df.get("condition", pd.Series(np.nan, index=df.index)).astype("string").str.lower()
    day_night = df.get("day_night", pd.Series(np.nan, index=df.index)).astype("string").str.lower()
    known = wind_dir.notna()
    out["wx_wind_out"] = np.where(known, wind_dir.str.startswith("out").fillna(False).astype(float), np.nan)
    out["wx_wind_in"] = np.where(known, wind_dir.str.startswith("in").fillna(False).astype(float), np.nan)
    out["wx_wind_out_mph"] = out["wx_wind_out"] * out["wx_wind_mph"]
    out["wx_wind_in_mph"] = out["wx_wind_in"] * out["wx_wind_mph"]
    dome = cond.str.contains("dome|roof closed", regex=True).fillna(False)
    out["wx_dome"] = np.where(cond.notna(), dome.astype(float), np.nan)
    wet = cond.str.contains("rain|drizzle|shower", regex=True).fillna(False)
    out["wx_precip"] = np.where(cond.notna(), wet.astype(float), np.nan)
    out["wx_night"] = np.where(day_night.notna(), (day_night == "night").astype(float), np.nan)
    return out


def _attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, weather_features(df)], axis=1)


# ---------------------------------------------------------------- batters
def assemble_batter_features(spec: pd.DataFrame, states: States) -> pd.DataFrame:
    """``spec`` rows: date, player_id, team_id, opp_team_id, opp_sp_id, is_home, batting_order,
    optional weather columns (temp_f, wind_mph, wind_dir, condition, day_night)."""
    df = spec.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = asof_join(df, states.batters, "player_id", "player_id")
    df = asof_join(df, states.pitchers, "opp_sp_id", "player_id", prefix="opp_")
    df = asof_join(df, states.teams, "team_id", "team_id", prefix="own_")
    df = asof_join(df, states.teams, "opp_team_id", "team_id", prefix="opp_")
    # platoon: batter side vs. starter hand, and the batter's own rolling split vs. that hand
    df["bats"] = states.hand(df["player_id"], "bats")
    df["opp_hand"] = states.hand(df["opp_sp_id"], "throws")
    df["bats_code"] = hand_code(df["bats"])
    df["opp_hand_code"] = hand_code(df["opp_hand"])
    edge = np.where(df["bats"] == "S", 1.0, np.where(df["bats"] != df["opp_hand"], 1.0, -1.0))
    df["platoon_edge"] = np.where(df["bats"].isna() | df["opp_hand"].isna(), np.nan, edge)
    df["_split_key"] = split_key(df["player_id"], df["opp_hand"])
    df = asof_join(df, states.splits, "_split_key", "split_key")
    df = df.drop(columns=["_split_key"])
    df["split_avg_delta"] = df["sp_avg"] - df["b_avg_r30"]
    df["split_slg_delta"] = df["sp_slg"] - df["b_slg_r30"]
    df["month"] = df["date"].dt.month
    df["is_home"] = df["is_home"].astype(float)
    df["batting_order"] = pd.to_numeric(df["batting_order"], errors="coerce").astype(float)
    df["park_factor"] = np.where(df["is_home"] == 1, df["own_tm_park_factor"], df["opp_tm_park_factor"])
    df["opp_sp_known"] = df["opp_p_apps"].notna().astype(float)
    df = _attach_weather(df)
    return _finalise(df, ["b_last_date", "opp_p_last_date", "own_tm_last_date", "opp_tm_last_date"])


def _spec_with_weather(lines: pd.DataFrame, games: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    wx = games[["game_pk", "temp_f", "wind_mph", "wind_dir", "condition", "day_night"]]
    return lines[cols].merge(wx, on="game_pk", how="left")


def build_batter_training(ds: Dataset, states: States | None = None) -> pd.DataFrame:
    states = states or States.from_dataset(ds)
    lines = ds.batting_lines
    spec = _spec_with_weather(lines, ds.games, ["game_pk", "date", "player_id", "player_name", "team_id",
                                                "opp_team_id", "opp_sp_id", "is_home", "batting_order"])
    feats = assemble_batter_features(spec, states)
    for t in BATTING_TARGETS:
        feats[f"y_{t}"] = lines[t].to_numpy()
    feats["y_pa"] = lines["pa"].to_numpy()
    return feats


# --------------------------------------------------------------- pitchers
def assemble_pitcher_features(spec: pd.DataFrame, states: States, lineups: pd.DataFrame | None = None) -> pd.DataFrame:
    """``spec`` rows: game_pk, date, player_id, team_id, opp_team_id, is_home (+ optional weather).
    ``lineups``: opposing starters as (game_pk, date, team_id, player_id) rows."""
    df = spec.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = asof_join(df, states.pitchers, "player_id", "player_id")
    df = asof_join(df, states.teams, "team_id", "team_id", prefix="own_")
    df = asof_join(df, states.teams, "opp_team_id", "team_id", prefix="opp_")
    df["throws_code"] = hand_code(states.hand(df["player_id"], "throws"))
    df = _merge_lineup(df, lineups, states, "opp_team_id", "opplu_")
    df["month"] = df["date"].dt.month
    df["is_home"] = df["is_home"].astype(float)
    df["park_factor"] = np.where(df["is_home"] == 1, df["own_tm_park_factor"], df["opp_tm_park_factor"])
    df = _attach_weather(df)
    return _finalise(df, ["p_last_date", "own_tm_last_date", "opp_tm_last_date"])


def _merge_lineup(df: pd.DataFrame, lineups: pd.DataFrame | None, states: States, team_col: str,
                  prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}{c[3:]}" for c in LINEUP_COLS]
    if lineups is None or lineups.empty or "game_pk" not in df.columns:
        for c in cols:
            df[c] = np.nan
        return df
    lf = lineup_features(lineups, states.batters, states.players)
    lf = lf.rename(columns={c: f"{prefix}{c[3:]}" for c in LINEUP_COLS})
    lf = lf.rename(columns={"team_id": team_col})
    out = df.merge(lf, on=["game_pk", team_col], how="left")
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def build_pitcher_training(ds: Dataset, states: States | None = None, starters_only: bool = True) -> pd.DataFrame:
    states = states or States.from_dataset(ds)
    lines = ds.pitching_lines
    if starters_only:
        lines = lines[lines["is_starter"] == 1]
    lines = lines.reset_index(drop=True)
    spec = _spec_with_weather(lines, ds.games, ["game_pk", "date", "player_id", "player_name", "team_id",
                                                "opp_team_id", "is_home"])
    feats = assemble_pitcher_features(spec, states, starters_from_lines(ds.batting_lines))
    for t in PITCHING_TARGETS:
        feats[f"y_{t}"] = lines[t].to_numpy()
    return feats


# ------------------------------------------------------------------ games
def assemble_game_features(spec: pd.DataFrame, states: States, lineups: pd.DataFrame | None = None) -> pd.DataFrame:
    """``spec`` rows: game_pk, date, home_team_id, away_team_id, home_sp_id, away_sp_id (+ weather)."""
    df = spec.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = asof_join(df, states.teams, "home_team_id", "team_id", prefix="home_")
    df = asof_join(df, states.teams, "away_team_id", "team_id", prefix="away_")
    df = asof_join(df, states.pitchers, "home_sp_id", "player_id", prefix="hsp_")
    df = asof_join(df, states.pitchers, "away_sp_id", "player_id", prefix="asp_")
    df["hsp_throws_code"] = hand_code(states.hand(df["home_sp_id"], "throws"))
    df["asp_throws_code"] = hand_code(states.hand(df["away_sp_id"], "throws"))
    df = _merge_lineup(df, lineups, states, "home_team_id", "hlu_")
    df = _merge_lineup(df, lineups, states, "away_team_id", "alu_")
    df["lu_slg_gap"] = df["hlu_slg_r30"] - df["alu_slg_r30"]
    df["lu_form_gap"] = df["hlu_form_slg"] - df["alu_form_slg"]
    df["month"] = df["date"].dt.month
    df["elo_diff"] = df["home_tm_elo"] - df["away_tm_elo"]
    df["elo_home_prob"] = 1.0 / (1.0 + 10 ** ((df["away_tm_elo"] - (df["home_tm_elo"] + 24.0)) / 400.0))
    for w in (5, 10, 30, 162):
        df[f"run_diff_gap_r{w}"] = df[f"home_tm_run_diff_r{w}"] - df[f"away_tm_run_diff_r{w}"]
        df[f"win_gap_r{w}"] = df[f"home_tm_win_r{w}"] - df[f"away_tm_win_r{w}"]
    for w in (3, 10, 30):
        df[f"sp_era_gap_r{w}"] = df[f"hsp_p_era_r{w}"] - df[f"asp_p_era_r{w}"]
        df[f"sp_k_gap_r{w}"] = df[f"hsp_p_k_rate_r{w}"] - df[f"asp_p_k_rate_r{w}"]
    df["home_sp_known"] = df["hsp_p_apps"].notna().astype(float)
    df["away_sp_known"] = df["asp_p_apps"].notna().astype(float)
    df = _attach_weather(df)
    return _finalise(df, ["home_tm_last_date", "away_tm_last_date", "hsp_p_last_date", "asp_p_last_date"])


def build_game_training(ds: Dataset, states: States | None = None) -> pd.DataFrame:
    states = states or States.from_dataset(ds)
    games = ds.played_games().reset_index(drop=True)
    spec = games[["game_pk", "date", "season", "home_team_id", "away_team_id", "home_sp_id", "away_sp_id",
                  "temp_f", "wind_mph", "wind_dir", "condition", "day_night"]].copy()
    feats = assemble_game_features(spec, states, starters_from_lines(ds.batting_lines))
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
