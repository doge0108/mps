"""Download real MLB data: full seasons, incremental updates, and upcoming schedules."""
from __future__ import annotations

import logging
from datetime import date as _date, timedelta
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_DATA_DIR
from .mlb_api import (MLBStatsClient, parse_boxscore, parse_boxscore_lineups, parse_schedule, parse_schedule_lineups,
                      parse_umpire)
from .statcast import fetch_statcast_range
from .store import Dataset, empty_lineups, empty_players, normalise

log = logging.getLogger(__name__)

SEASON_START = "03-15"
SEASON_END = "11-10"
UPCOMING_TYPES = ("R", "P", "F", "D", "L", "W")  # regular season + every postseason round


def _fetch_finals(client: MLBStatsClient, sched_rows: list[dict], weather: bool, label: str,
                  progress: bool) -> tuple[list[dict], list[dict], list[dict]]:
    games, batting, pitching = [], [], []
    finals = [g for g in sched_rows if g["status"] == "Final"]
    for i, g in enumerate(finals, start=1):
        if progress and i % 100 == 0:
            print(f"  {label}: {i}/{len(finals)} boxscores")
        try:
            box = client.boxscore(g["game_pk"])
        except Exception as exc:  # pragma: no cover - network
            log.warning("skipping game %s: %s", g["game_pk"], exc)
            continue
        b, p, starters = parse_boxscore(box, g["game_pk"], g["date"])
        row = dict(g)
        row["home_sp_id"] = starters.get("home_sp_id") or row.get("home_sp_id")
        row["away_sp_id"] = starters.get("away_sp_id") or row.get("away_sp_id")
        row.update({k: v for k, v in parse_umpire(box).items() if v is not None})  # boxscore carries officials
        if weather:
            try:
                row.update({k: v for k, v in client.weather(g["game_pk"]).items() if v is not None})
            except Exception as exc:  # pragma: no cover - network
                log.warning("no weather for game %s: %s", g["game_pk"], exc)
        games.append(row)
        batting.extend(b)
        pitching.extend(p)
    return games, batting, pitching


def _to_dataset(games, batting, pitching, players=None, lineups=None, statcast=None) -> Dataset:
    frames = {
        "games": pd.DataFrame(games),
        "batting_lines": pd.DataFrame(batting),
        "pitching_lines": pd.DataFrame(pitching),
        "players": pd.DataFrame(players) if players else empty_players(),
        "lineups": pd.DataFrame(lineups) if lineups else empty_lineups(),
    }
    if statcast is not None:
        frames["statcast_batting"], frames["statcast_pitching"] = statcast
    return Dataset(**normalise(frames))


def fetch_statcast(start: _date, end: _date, data_dir: Path, today: _date | None = None,
                   progress: bool = True, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Statcast aggregates between two dates (cached per window under data/raw)."""
    print(f"Fetching Statcast {start} .. {end} ...")
    bat, pit = fetch_statcast_range(start, end, cache_dir=Path(data_dir) / "raw", today=today, progress=progress,
                                    verbose=verbose)
    print(f"  {len(bat)} batter-games, {len(pit)} pitcher-games with Statcast data")
    return bat, pit


def fetch_range(client: MLBStatsClient, start: str, end: str, game_types=("R",), weather: bool = True,
                progress: bool = True, label: str | None = None) -> Dataset:
    """All completed games between two ISO dates (inclusive)."""
    sched = parse_schedule(client.schedule(start, end, game_types))
    games, batting, pitching = _fetch_finals(client, sched, weather, label or f"{start}..{end}", progress)
    return _to_dataset(games, batting, pitching)


def fetch_season(client: MLBStatsClient, season: int, game_types=("R",), weather: bool = True,
                 progress: bool = True) -> Dataset:
    """A full season, or everything played so far if the season is still in progress."""
    end = _date.fromisoformat(f"{season}-{SEASON_END}")
    if end >= client.today:
        end = client.today
    ds = fetch_range(client, f"{season}-{SEASON_START}", end.isoformat(), game_types, weather, progress, str(season))
    try:
        ds.players = pd.DataFrame(client.players(season))
    except Exception as exc:  # pragma: no cover - network
        log.warning("could not fetch player list for %s: %s", season, exc)
    return Dataset(**normalise(ds.__dict__))


def fetch_seasons(seasons: list[int], data_dir: Path = DEFAULT_DATA_DIR, client: MLBStatsClient | None = None,
                  game_types=("R",), weather: bool = True, statcast: bool = True) -> Dataset:
    data_dir = Path(data_dir)
    client = client or MLBStatsClient(cache_dir=data_dir / "raw")
    combined: Dataset | None = None
    for season in seasons:
        print(f"Fetching {season} ...")
        ds = fetch_season(client, season, game_types, weather)
        print(f"  {len(ds.games)} games, {len(ds.batting_lines)} batting lines, {len(ds.pitching_lines)} pitching lines")
        if statcast and len(ds.games):
            start, end = ds.games["date"].min().date(), ds.games["date"].max().date()
            ds.statcast_batting, ds.statcast_pitching = fetch_statcast(start, end, data_dir, client.today)
            ds = Dataset(**normalise(ds.__dict__))
        combined = ds if combined is None else combined.concat(ds)
    assert combined is not None
    if (data_dir / "games.csv").exists():
        combined = Dataset.load(data_dir).concat(combined)
    combined.save(data_dir)
    return combined


def fetch_upcoming(client: MLBStatsClient, start: str, end: str, weather: bool = True) -> Dataset:
    """Scheduled (not yet final) games with probable pitchers, announced lineups and forecast weather."""
    payload = client.schedule(start, end, UPCOMING_TYPES, with_lineups=True)
    rows = [g for g in parse_schedule(payload) if g["status"] != "Final"]
    lineups = parse_schedule_lineups(payload)
    have = {(r["game_pk"], r["team_id"]) for r in lineups}
    for g in rows:
        if (g["game_pk"], g["home_team_id"]) in have and (g["game_pk"], g["away_team_id"]) in have:
            continue
        # lineups not in the schedule hydration yet: the pre-game boxscore carries them once posted
        try:
            box = client.boxscore(g["game_pk"], final=False)
            lineups.extend(parse_boxscore_lineups(box, g["game_pk"], g["date"]))
            g.update({k: v for k, v in parse_umpire(box).items() if v is not None})
        except Exception as exc:  # pragma: no cover - network
            log.debug("no pre-game boxscore for %s: %s", g["game_pk"], exc)
        if weather:
            try:
                g.update({k: v for k, v in client.weather(g["game_pk"], final=False).items() if v is not None})
            except Exception as exc:  # pragma: no cover - network
                log.debug("no forecast for %s: %s", g["game_pk"], exc)
    for g in rows:
        g["home_score"] = None
        g["away_score"] = None
    return _to_dataset(rows, [], [], lineups=lineups)


def update(data_dir: Path = DEFAULT_DATA_DIR, days_ahead: int = 7, client: MLBStatsClient | None = None,
           weather: bool = True, game_types=("R",), statcast: bool = True) -> Dataset:
    """Incremental refresh: new final games since the last stored game, plus the upcoming schedule.

    Safe to run every day during the season; upcoming rows are replaced by their box scores
    once the games finish.
    """
    data_dir = Path(data_dir)
    client = client or MLBStatsClient(cache_dir=data_dir / "raw")
    existing = Dataset.load(data_dir)
    today = client.today
    played = existing.played_games()
    if played.empty:
        start = _date(today.year, 3, 15)
    else:
        start = played["date"].max().date() - timedelta(days=1)  # re-check the last day (suspended games)
    print(f"Updating finals from {start} to {today} ...")
    new = fetch_range(client, start.isoformat(), today.isoformat(), game_types, weather, label="update")
    print(f"  {len(new.games)} completed games in window")
    try:
        new.players = pd.DataFrame(client.players(today.year))
    except Exception as exc:  # pragma: no cover - network
        log.warning("could not refresh player list: %s", exc)
    if statcast and len(new.games):
        sc_start = start
        if len(existing.statcast_batting):
            sc_start = max(start, existing.statcast_batting["date"].max().date() - timedelta(days=1))
        new.statcast_batting, new.statcast_pitching = fetch_statcast(sc_start, today, data_dir, today)
    new = Dataset(**normalise(new.__dict__))
    end = today + timedelta(days=days_ahead)
    print(f"Fetching upcoming schedule {today} .. {end} ...")
    upcoming = fetch_upcoming(client, today.isoformat(), end.isoformat(), weather)
    print(f"  {len(upcoming.games)} upcoming games, {len(upcoming.lineups)} announced lineup slots")
    # drop stale upcoming rows before merging so postponed games disappear
    existing.games = existing.played_games().reset_index(drop=True)
    existing.lineups = existing.lineups[existing.lineups["game_pk"].isin(existing.games["game_pk"])].reset_index(drop=True)
    combined = existing.concat(new).concat(upcoming)
    combined.save(data_dir)
    return combined


def fetch_schedule(date: str, client: MLBStatsClient | None = None) -> pd.DataFrame:
    """Games for a single date (any state) with probable pitchers; used by ``--live`` predictions."""
    client = client or MLBStatsClient(cache_dir=None)
    rows = parse_schedule(client.schedule(date, date, UPCOMING_TYPES))
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df
