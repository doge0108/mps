"""Post-game "state" tables and the as-of join that makes features leak-free.

A state row summarises everything known about a player/team *after* the games
on a given date.  Features for a game on date D are obtained by joining the
latest state row with date strictly before D (``asof_join``).  Training and
live prediction therefore share exactly the same code path: to predict a
future game we simply look up the latest available states.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BATTING_STATS, PITCHING_STATS
from .common import safe_div
from .team_features import ELO_HOME, ELO_K, ELO_SEASON_REGRESS, ELO_START, team_game_log

BATTER_ROLL_COLS = ["pa", "ab", "h", "hr", "rbi", "r", "bb", "so", "sb", "tb", "d"]
BATTER_WINDOWS = (7, 15, 30, 60)
PITCHER_ROLL_COLS = ["outs", "h", "er", "bb", "so", "hr", "bf", "pitches"]
PITCHER_WINDOWS = (5, 10, 30)
TEAM_WINDOWS = (10, 30, 162)


def _rolling_inclusive(df: pd.DataFrame, by: str, cols: list[str], window: int, prefix: str) -> pd.DataFrame:
    grp = df.groupby(by, sort=False)[cols]
    out = grp.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
    out.columns = [f"{prefix}{c}_r{window}" for c in cols]
    return out


def _collapse_same_day(state: pd.DataFrame, key: str) -> pd.DataFrame:
    """Keep the last row per (key, date) so doubleheaders yield one state per day."""
    return state.drop_duplicates(subset=[key, "date"], keep="last").reset_index(drop=True)


# ---------------------------------------------------------------- batters
def batter_state(batting_lines: pd.DataFrame) -> pd.DataFrame:
    df = batting_lines.sort_values(["player_id", "date", "game_pk"]).reset_index(drop=True)
    parts = [df[["player_id", "date", "team_id", "player_name", "batting_order"]].rename(
        columns={"team_id": "b_team_id", "player_name": "b_name", "batting_order": "b_last_order"})]
    for w in BATTER_WINDOWS:
        parts.append(_rolling_inclusive(df, "player_id", BATTER_ROLL_COLS, w, "b_"))
    grp = df.groupby("player_id", sort=False)
    cum = grp[BATTER_ROLL_COLS].cumsum()
    cum.columns = [f"b_{c}_cum" for c in BATTER_ROLL_COLS]
    parts.append(cum)
    st = pd.concat(parts, axis=1)
    st["b_games"] = grp.cumcount() + 1
    st["b_last_date"] = df["date"]
    # derived rates from the 30 / 60 game windows and cumulative totals
    for w in (30, 60):
        st[f"b_avg_r{w}"] = safe_div(st[f"b_h_r{w}"], st[f"b_ab_r{w}"])
        st[f"b_hr_rate_r{w}"] = safe_div(st[f"b_hr_r{w}"], st[f"b_pa_r{w}"])
        st[f"b_bb_rate_r{w}"] = safe_div(st[f"b_bb_r{w}"], st[f"b_pa_r{w}"])
        st[f"b_k_rate_r{w}"] = safe_div(st[f"b_so_r{w}"], st[f"b_pa_r{w}"])
        st[f"b_slg_r{w}"] = safe_div(st[f"b_tb_r{w}"], st[f"b_ab_r{w}"])
        st[f"b_sb_rate_r{w}"] = safe_div(st[f"b_sb_r{w}"], st[f"b_pa_r{w}"])
    st["b_avg_cum"] = safe_div(st["b_h_cum"], st["b_ab_cum"])
    st["b_hr_rate_cum"] = safe_div(st["b_hr_cum"], st["b_pa_cum"])
    st["b_k_rate_cum"] = safe_div(st["b_so_cum"], st["b_pa_cum"])
    st["b_bb_rate_cum"] = safe_div(st["b_bb_cum"], st["b_pa_cum"])
    st["b_slg_cum"] = safe_div(st["b_tb_cum"], st["b_ab_cum"])
    # raw cumulative counts are non-stationary (they only grow with the size of the
    # dataset) so keep the rates and a capped experience count only
    st["b_games"] = st["b_games"].clip(upper=150)
    st = st.drop(columns=[f"b_{c}_cum" for c in BATTER_ROLL_COLS])
    st = st.sort_values(["date", "player_id"]).reset_index(drop=True)
    return _collapse_same_day(st, "player_id")


# --------------------------------------------------------------- pitchers
def pitcher_state(pitching_lines: pd.DataFrame) -> pd.DataFrame:
    df = pitching_lines.sort_values(["player_id", "date", "game_pk"]).reset_index(drop=True)
    parts = [df[["player_id", "date", "team_id", "player_name"]].rename(
        columns={"team_id": "p_team_id", "player_name": "p_name"})]
    for w in PITCHER_WINDOWS:
        parts.append(_rolling_inclusive(df, "player_id", PITCHER_ROLL_COLS, w, "p_"))
    grp = df.groupby("player_id", sort=False)
    cum = grp[PITCHER_ROLL_COLS].cumsum()
    cum.columns = [f"p_{c}_cum" for c in PITCHER_ROLL_COLS]
    parts.append(cum)
    st = pd.concat(parts, axis=1)
    st["p_apps"] = grp.cumcount() + 1
    st["p_starts"] = grp["is_starter"].cumsum()
    st["p_start_share_r10"] = _rolling_inclusive(df, "player_id", ["is_starter"], 10, "x_")["x_is_starter_r10"]
    st["p_last_date"] = df["date"]
    for w in (10, 30):
        st[f"p_k_rate_r{w}"] = safe_div(st[f"p_so_r{w}"], st[f"p_bf_r{w}"])
        st[f"p_bb_rate_r{w}"] = safe_div(st[f"p_bb_r{w}"], st[f"p_bf_r{w}"])
        st[f"p_hr_rate_r{w}"] = safe_div(st[f"p_hr_r{w}"], st[f"p_bf_r{w}"])
        st[f"p_h_rate_r{w}"] = safe_div(st[f"p_h_r{w}"], st[f"p_bf_r{w}"])
        st[f"p_era_r{w}"] = safe_div(st[f"p_er_r{w}"] * 27.0, st[f"p_outs_r{w}"])
    st["p_k_rate_cum"] = safe_div(st["p_so_cum"], st["p_bf_cum"])
    st["p_bb_rate_cum"] = safe_div(st["p_bb_cum"], st["p_bf_cum"])
    st["p_hr_rate_cum"] = safe_div(st["p_hr_cum"], st["p_bf_cum"])
    st["p_era_cum"] = safe_div(st["p_er_cum"] * 27.0, st["p_outs_cum"])
    st["p_apps"] = st["p_apps"].clip(upper=60)
    st["p_starts"] = st["p_starts"].clip(upper=40)
    st = st.drop(columns=[f"p_{c}_cum" for c in PITCHER_ROLL_COLS])
    st = st.sort_values(["date", "player_id"]).reset_index(drop=True)
    return _collapse_same_day(st, "player_id")


# ------------------------------------------------------------------ teams
def _post_game_elo(games: pd.DataFrame) -> pd.DataFrame:
    g = games.sort_values(["date", "game_pk"]).reset_index(drop=True)
    ratings: dict[int, float] = {}
    season_seen = None
    rows = []
    for row in g.itertuples(index=False):
        season = None if pd.isna(row.season) else int(row.season)
        if season is not None and season_seen is not None and season != season_seen:
            for tid in ratings:
                ratings[tid] += ELO_SEASON_REGRESS * (ELO_START - ratings[tid])
        if season is not None:
            season_seen = season
        h, a = int(row.home_team_id), int(row.away_team_id)
        rh, ra = ratings.get(h, ELO_START), ratings.get(a, ELO_START)
        if not (pd.isna(row.home_score) or pd.isna(row.away_score)):
            exp_home = 1.0 / (1.0 + 10 ** ((ra - (rh + ELO_HOME)) / 400.0))
            hs, as_ = float(row.home_score), float(row.away_score)
            result = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
            margin = abs(hs - as_)
            delta = ELO_K * (np.log1p(margin) if margin > 0 else 1.0) * (result - exp_home)
            rh, ra = rh + delta, ra - delta
            ratings[h], ratings[a] = rh, ra
        rows.append((row.game_pk, h, rh))
        rows.append((row.game_pk, a, ra))
    return pd.DataFrame(rows, columns=["game_pk", "team_id", "tm_elo"])


def team_state(games: pd.DataFrame, pitching_lines: pd.DataFrame | None = None) -> pd.DataFrame:
    """Post-game team state: form, Elo, park factor, bullpen quality, last game date."""
    played = games[games["home_score"].notna() & games["away_score"].notna()]
    log = team_game_log(played)
    log = log.sort_values(["team_id", "date", "game_pk"]).reset_index(drop=True)
    parts = [log[["team_id", "date", "game_pk"]]]
    for w in TEAM_WINDOWS:
        parts.append(_rolling_inclusive(log, "team_id", ["rs", "ra", "win", "run_diff"], w, "tm_"))
    st = pd.concat(parts, axis=1)
    st["tm_games"] = (log.groupby("team_id", sort=False).cumcount() + 1).clip(upper=162)
    st["tm_last_date"] = log["date"]
    # park factor: total runs in the team's home games relative to league average (last 81 home games)
    league_avg = log["total_runs"].mean()
    is_home = log["is_home"] == 1
    home_tr = log["total_runs"].where(is_home)
    home_roll = home_tr.groupby(log["team_id"]).transform(lambda s: s.rolling(81, min_periods=5).mean())
    home_roll = home_roll.groupby(log["team_id"]).ffill()
    st["tm_park_factor"] = (home_roll / league_avg).to_numpy()
    st = st.merge(_post_game_elo(played), on=["game_pk", "team_id"], how="left")
    if pitching_lines is not None and len(pitching_lines):
        # bullpen ERA / K-rate: relievers only, last 30 team games
        rp = pitching_lines[pitching_lines["is_starter"] == 0]
        agg = rp.groupby(["team_id", "game_pk"], sort=False)[["outs", "er", "so", "bf"]].sum().reset_index()
        agg = agg.merge(log[["team_id", "game_pk", "date"]], on=["team_id", "game_pk"], how="inner")
        agg = agg.sort_values(["team_id", "date", "game_pk"]).reset_index(drop=True)
        roll = _rolling_inclusive(agg, "team_id", ["outs", "er", "so", "bf"], 30, "bp_")
        agg = pd.concat([agg[["team_id", "game_pk"]], roll], axis=1)
        agg["tm_bullpen_era_r30"] = safe_div(agg["bp_er_r30"] * 27.0, agg["bp_outs_r30"])
        agg["tm_bullpen_k_rate_r30"] = safe_div(agg["bp_so_r30"], agg["bp_bf_r30"])
        st = st.merge(agg[["team_id", "game_pk", "tm_bullpen_era_r30", "tm_bullpen_k_rate_r30"]],
                      on=["team_id", "game_pk"], how="left")
        st = st.sort_values(["team_id", "date", "game_pk"]).reset_index(drop=True)
        for col in ("tm_bullpen_era_r30", "tm_bullpen_k_rate_r30"):
            st[col] = st.groupby("team_id", sort=False)[col].ffill()
    st = st.drop(columns=["game_pk"]).sort_values(["date", "team_id"]).reset_index(drop=True)
    return _collapse_same_day(st, "team_id")


# ---------------------------------------------------------------- as-of
def asof_join(left: pd.DataFrame, state: pd.DataFrame, left_key: str, state_key: str,
              prefix: str = "") -> pd.DataFrame:
    """Attach the latest state row with ``state.date < left.date`` for each left row.

    ``left`` may contain NaN keys (unknown starting pitcher); those rows get NaN features.
    Row order of ``left`` is preserved.
    """
    left = left.copy()
    left["_order"] = np.arange(len(left))
    left["_key"] = pd.to_numeric(left[left_key], errors="coerce").fillna(-1).astype("int64")
    st = state.copy()
    st["_key"] = st[state_key].astype("int64")
    feat_cols = [c for c in st.columns if c not in (state_key, "date", "_key")]
    if prefix:
        st = st.rename(columns={c: f"{prefix}{c}" for c in feat_cols})
        feat_cols = [f"{prefix}{c}" for c in feat_cols]
    left_sorted = left.sort_values("date", kind="mergesort")
    st_sorted = st.sort_values("date", kind="mergesort")
    merged = pd.merge_asof(
        left_sorted, st_sorted[["_key", "date"] + feat_cols], on="date", by="_key",
        allow_exact_matches=False, direction="backward",
    )
    merged = merged.sort_values("_order").drop(columns=["_order", "_key"]).reset_index(drop=True)
    return merged
