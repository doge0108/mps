"""Marcel-style preseason projections and player age.

Marcel (Tom Tango's "monkey" projection) weights the last three seasons
5/4/3 (batters) or 3/2/1 (pitchers), regresses toward the league rate with a
fixed number of phantom plate appearances / outs, and applies a simple aging
curve.  The result is a prior for the coming season that is far better than a
league average for players with little current-season data.  Each row is dated
January 1st of the season it projects, so the usual as-of join picks it up for
every game of that season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAT_WEIGHTS = (5.0, 4.0, 3.0)
BAT_REGRESS_PA = 1200.0
PIT_WEIGHTS = (3.0, 2.0, 1.0)
PIT_REGRESS_OUTS = 400.0  # ~134 innings
BAT_RATES = {"avg": ("h", "ab"), "hr_rate": ("hr", "pa"), "bb_rate": ("bb", "pa"), "k_rate": ("so", "pa"),
             "slg": ("tb", "ab"), "sb_rate": ("sb", "pa")}
PIT_RATES = {"era": ("er", "outs"), "k_rate": ("so", "bf"), "bb_rate": ("bb", "bf"), "hr_rate": ("hr", "bf"),
             "h_rate": ("h", "bf")}


def age_on(date: pd.Series, birth_date: pd.Series) -> pd.Series:
    """Age in years (NaN when the birth date is unknown)."""
    bd = pd.to_datetime(birth_date, errors="coerce")
    return ((pd.to_datetime(date) - bd).dt.days / 365.25).astype(float)


def _age_factor(age: float | np.ndarray, batter: bool) -> np.ndarray:
    """Marcel aging: improve until 29, decline after (rates of good things)."""
    age = np.asarray(age, dtype=float)
    up, down = (0.006, 0.003) if batter else (0.004, 0.002)
    f = np.where(age < 29, 1 + (29 - age) * up, 1 - (age - 29) * down)
    return np.where(np.isnan(age), 1.0, f)


def _season_totals(lines: pd.DataFrame, games: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    seasons = games.drop_duplicates("game_pk").set_index("game_pk")["season"]
    df = lines.assign(season=lines["game_pk"].map(seasons).astype("Int64")).dropna(subset=["season"])
    df["season"] = df["season"].astype(int)
    return df.groupby(["player_id", "season"], sort=False)[cols].sum().reset_index()


def _project(totals: pd.DataFrame, rates: dict, weights: tuple, regress: float, denom_col: str,
             players: pd.DataFrame | None, batter: bool, prefix: str) -> pd.DataFrame:
    """Weighted prior-seasons projection for every (player, season) with at least one prior season."""
    if totals.empty:
        return pd.DataFrame(columns=["player_id", "date"])
    league = totals.groupby("season")[sorted({c for pair in rates.values() for c in pair})].sum()
    seasons = sorted(totals["season"].unique())
    target_seasons = seasons[1:] + [seasons[-1] + 1]
    by_player = {pid: g.set_index("season") for pid, g in totals.groupby("player_id")}
    birth = None
    if players is not None and not players.empty and "birth_date" in players.columns:
        birth = players.set_index("player_id")["birth_date"]
    rows = []
    for pid, hist in by_player.items():
        for season in target_seasons:
            prior = [(season - k, w) for k, w in zip((1, 2, 3), weights) if (season - k) in hist.index]
            if not prior:
                continue
            row = {"player_id": pid, "date": pd.Timestamp(year=season, month=1, day=1)}
            denom_w = sum(hist.loc[s, denom_col] * w for s, w in prior)
            row[f"{prefix}reliability"] = denom_w / (denom_w + regress)
            # league rate is a weighted average over the same seasons
            for name, (num, den) in rates.items():
                num_w = sum(hist.loc[s, num] * w for s, w in prior)
                den_w = sum(hist.loc[s, den] * w for s, w in prior)
                lg_num = sum(league.loc[s, num] * w for s, w in prior)
                lg_den = sum(league.loc[s, den] * w for s, w in prior)
                lg_rate = lg_num / lg_den if lg_den else 0.0
                phantom = regress * (den_w / denom_w if denom_w else 1.0)
                rate = (num_w + lg_rate * phantom) / (den_w + phantom) if (den_w + phantom) else lg_rate
                row[f"{prefix}{name}"] = rate
            age = np.nan
            if birth is not None and pid in birth.index and pd.notna(birth.loc[pid]):
                age = (row["date"] - pd.Timestamp(birth.loc[pid])).days / 365.25 + 0.5
            f = float(_age_factor(age, batter))
            good = ("avg", "hr_rate", "bb_rate", "slg", "sb_rate") if batter else ("k_rate",)
            bad = ("k_rate",) if batter else ("era", "bb_rate", "hr_rate", "h_rate")
            for name in good:
                row[f"{prefix}{name}"] *= f
            for name in bad:
                row[f"{prefix}{name}"] /= f
            if f"{prefix}era" in row:  # ERA is per 27 outs
                row[f"{prefix}era"] = row[f"{prefix}era"] * 27.0
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["date", "player_id"]).reset_index(drop=True)


def batter_priors(batting_lines: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame | None) -> pd.DataFrame:
    totals = _season_totals(batting_lines, games, ["pa", "ab", "h", "hr", "bb", "so", "tb", "sb"])
    return _project(totals, BAT_RATES, BAT_WEIGHTS, BAT_REGRESS_PA, "pa", players, True, "mc_")


def pitcher_priors(pitching_lines: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame | None) -> pd.DataFrame:
    totals = _season_totals(pitching_lines, games, ["outs", "bf", "er", "so", "bb", "hr", "h"])
    return _project(totals, PIT_RATES, PIT_WEIGHTS, PIT_REGRESS_OUTS, "outs", players, False, "mcp_")
