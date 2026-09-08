"""Statcast (Baseball Savant) pitch-level data, aggregated to per player-game rows.

Baseball Savant's search endpoint returns CSV with one row per pitch and no API
key.  It caps each response at a few tens of thousands of rows, so seasons are
fetched in short date windows.  Only per-game aggregates are kept on disk:
quality of contact (exit velocity, hard-hit, barrels, expected stats) for
batters, and velocity / whiff / contact-allowed for pitchers.
"""
from __future__ import annotations

import io
import logging
import time
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import STATCAST_BATTING_COLUMNS, STATCAST_PITCHING_COLUMNS

log = logging.getLogger(__name__)

SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
WINDOW_DAYS = 3  # ~15 games/day * ~300 pitches keeps each window under the row cap

USE_COLS = ["game_pk", "game_date", "batter", "pitcher", "events", "description", "zone", "pitch_type",
            "release_speed", "release_spin_rate", "launch_speed", "launch_angle", "type",
            "estimated_ba_using_speedangle", "estimated_woba_using_speedangle"]
SWING_DESC = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
              "hit_into_play_no_out", "hit_into_play_score", "foul_bunt", "missed_bunt", "bunt_foul_tip"}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
FASTBALLS = {"FF", "SI", "FC", "FA", "FT"}


def is_barrel(ev: pd.Series, la: pd.Series) -> pd.Series:
    """Statcast 'barrel' classification (the standard speed/angle rule)."""
    ev = pd.to_numeric(ev, errors="coerce")
    la = pd.to_numeric(la, errors="coerce")
    return ((ev * 1.5 - la >= 117) & (ev + la >= 124) & (ev >= 98) & (la >= 4) & (la <= 50)).fillna(False)


def savant_params(start: str, end: str, player_type: str = "batter") -> dict:
    return {
        "all": "true", "hfPT": "", "hfAB": "", "hfBBT": "", "hfPR": "", "hfZ": "", "stadium": "", "hfBBL": "",
        "hfNewZones": "", "hfGT": "R|", "hfSea": "", "hfSit": "", "player_type": player_type, "hfOuts": "",
        "opponent": "", "pitcher_throws": "", "batter_stands": "", "hfSA": "",
        "game_date_gt": start, "game_date_lt": end, "team": "", "position": "", "hfRO": "", "home_road": "",
        "hfFlag": "", "metric_1": "", "hfInn": "", "min_pitches": 0, "min_results": 0, "group_by": "name",
        "sort_col": "pitches", "player_event_sort": "api_p_release_speed", "sort_order": "desc", "min_abs": 0,
        "type": "details",
    }


def fetch_pitches(start: str, end: str, session: requests.Session | None = None, timeout: float = 120.0,
                  cache_dir: Path | None = None, today: _date | None = None) -> pd.DataFrame:
    """Pitch-level rows for [start, end] (inclusive); cached on disk when the window is in the past."""
    today = today or _date.today()
    cpath = None
    if cache_dir is not None and _date.fromisoformat(end) < today:
        cpath = Path(cache_dir) / f"statcast_{start}_{end}.csv.gz"
        if cpath.exists():
            return pd.read_csv(cpath, low_memory=False)
    session = session or requests.Session()
    resp = session.get(SAVANT_URL, params=savant_params(start, end), timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    if cpath is not None:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cpath, index=False, compression="gzip")
    return df


def aggregate_pitches(pitches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce pitch rows to (statcast_batting, statcast_pitching) per player-game aggregates."""
    if pitches is None or pitches.empty:
        return (pd.DataFrame(columns=STATCAST_BATTING_COLUMNS), pd.DataFrame(columns=STATCAST_PITCHING_COLUMNS))
    df = pitches.copy()
    for col in USE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[USE_COLS]
    df["date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    desc = df["description"].astype("string").fillna("")
    df["swing"] = desc.isin(SWING_DESC).astype(float)
    df["whiff"] = desc.isin(WHIFF_DESC).astype(float)
    df["called_strike"] = (desc == "called_strike").astype(float)
    zone = pd.to_numeric(df["zone"], errors="coerce")
    df["out_zone"] = (zone >= 11).astype(float).where(zone.notna(), np.nan)
    df["chase"] = df["swing"] * df["out_zone"].fillna(0)
    df["bip"] = (df["type"].astype("string") == "X").astype(float)
    ev = pd.to_numeric(df["launch_speed"], errors="coerce").where(df["bip"] == 1)
    la = pd.to_numeric(df["launch_angle"], errors="coerce").where(df["bip"] == 1)
    df["ev"] = ev
    df["la"] = la
    df["hard_hit"] = (ev >= 95).astype(float)
    df["barrel"] = is_barrel(ev, la).astype(float)
    df["xba"] = pd.to_numeric(df["estimated_ba_using_speedangle"], errors="coerce").where(df["bip"] == 1)
    df["xwoba"] = pd.to_numeric(df["estimated_woba_using_speedangle"], errors="coerce").where(df["bip"] == 1)
    fast = df["pitch_type"].astype("string").isin(FASTBALLS)
    df["fastball"] = fast.astype(float)
    velo = pd.to_numeric(df["release_speed"], errors="coerce")
    df["fb_velo"] = velo.where(fast)
    df["spin"] = pd.to_numeric(df["release_spin_rate"], errors="coerce")
    df["ev_for_max"] = ev

    bat = df.groupby(["game_pk", "date", "batter"], sort=False).agg(
        pitches=("swing", "size"), swings=("swing", "sum"), whiffs=("whiff", "sum"), chases=("chase", "sum"),
        out_zone_pitches=("out_zone", "sum"), bip=("bip", "sum"), ev_sum=("ev", "sum"), ev_max=("ev_for_max", "max"),
        la_sum=("la", "sum"), hard_hit=("hard_hit", "sum"), barrels=("barrel", "sum"),
        xba_sum=("xba", "sum"), xwoba_sum=("xwoba", "sum"),
    ).reset_index().rename(columns={"batter": "player_id"})
    pit = df.groupby(["game_pk", "date", "pitcher"], sort=False).agg(
        pitches=("swing", "size"), fastballs=("fastball", "sum"), fb_velo_sum=("fb_velo", "sum"),
        fb_velo_max=("fb_velo", "max"), spin_sum=("spin", "sum"), swings=("swing", "sum"), whiffs=("whiff", "sum"),
        called_strikes=("called_strike", "sum"), chases=("chase", "sum"), out_zone_pitches=("out_zone", "sum"),
        bip=("bip", "sum"), ev_sum=("ev", "sum"), hard_hit=("hard_hit", "sum"), barrels=("barrel", "sum"),
        xwoba_sum=("xwoba", "sum"),
    ).reset_index().rename(columns={"pitcher": "player_id"})
    return bat[STATCAST_BATTING_COLUMNS], pit[STATCAST_PITCHING_COLUMNS]


def fetch_statcast_range(start: _date, end: _date, cache_dir: Path | None = None, today: _date | None = None,
                         session: requests.Session | None = None, pause: float = 1.0,
                         progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All Statcast aggregates between two dates, fetched in short windows."""
    bats, pits = [], []
    cur = start
    n = 0
    while cur <= end:
        stop = min(cur + timedelta(days=WINDOW_DAYS - 1), end)
        try:
            pitches = fetch_pitches(cur.isoformat(), stop.isoformat(), session, cache_dir=cache_dir, today=today)
        except Exception as exc:  # pragma: no cover - network
            log.warning("statcast window %s..%s failed: %s", cur, stop, exc)
            pitches = pd.DataFrame()
        b, p = aggregate_pitches(pitches)
        bats.append(b)
        pits.append(p)
        n += 1
        if progress and n % 10 == 0:
            print(f"  statcast: through {stop}")
        cur = stop + timedelta(days=1)
        if pause:
            time.sleep(pause)
    bat = pd.concat(bats, ignore_index=True) if bats else pd.DataFrame(columns=STATCAST_BATTING_COLUMNS)
    pit = pd.concat(pits, ignore_index=True) if pits else pd.DataFrame(columns=STATCAST_PITCHING_COLUMNS)
    return bat, pit
