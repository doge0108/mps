"""Download real MLB seasons into the three core tables."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_DATA_DIR
from .mlb_api import MLBStatsClient, parse_boxscore
from .store import Dataset, normalise

log = logging.getLogger(__name__)

SEASON_WINDOWS = {  # generous bounds; the schedule endpoint only returns real games
    "start": "03-15",
    "end": "11-10",
}


def fetch_season(client: MLBStatsClient, season: int, game_types: tuple[str, ...] = ("R",),
                 progress: bool = True) -> Dataset:
    sched = client.schedule(f"{season}-{SEASON_WINDOWS['start']}", f"{season}-{SEASON_WINDOWS['end']}", game_types)
    games, batting, pitching = [], [], []
    finals = [g for g in sched if g["status"] == "Final"]
    for i, g in enumerate(finals, start=1):
        if progress and i % 100 == 0:
            print(f"  {season}: {i}/{len(finals)} boxscores")
        try:
            box = client.boxscore(g["game_pk"])
        except Exception as exc:  # pragma: no cover - network
            log.warning("skipping game %s: %s", g["game_pk"], exc)
            continue
        b, p, starters = parse_boxscore(box, g["game_pk"], g["date"])
        row = dict(g)
        row["home_sp_id"] = starters.get("home_sp_id") or row.get("home_sp_id")
        row["away_sp_id"] = starters.get("away_sp_id") or row.get("away_sp_id")
        games.append(row)
        batting.extend(b)
        pitching.extend(p)
    frames = normalise({
        "games": pd.DataFrame(games),
        "batting_lines": pd.DataFrame(batting),
        "pitching_lines": pd.DataFrame(pitching),
    })
    return Dataset(**frames)


def fetch_seasons(seasons: list[int], data_dir: Path = DEFAULT_DATA_DIR, client: MLBStatsClient | None = None,
                  game_types: tuple[str, ...] = ("R",)) -> Dataset:
    data_dir = Path(data_dir)
    client = client or MLBStatsClient(cache_dir=data_dir / "raw")
    combined: Dataset | None = None
    for season in seasons:
        print(f"Fetching {season} ...")
        ds = fetch_season(client, season, game_types)
        print(f"  {len(ds.games)} games, {len(ds.batting_lines)} batting lines, {len(ds.pitching_lines)} pitching lines")
        combined = ds if combined is None else combined.concat(ds)
    if (data_dir / "games.csv").exists():
        existing = Dataset.load(data_dir)
        combined = existing.concat(combined) if combined is not None else existing
    assert combined is not None
    combined.save(data_dir)
    return combined


def fetch_schedule(date: str, client: MLBStatsClient | None = None, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Upcoming (or completed) games for a single date with probable pitchers."""
    client = client or MLBStatsClient(cache_dir=None)
    rows = client.schedule(date, date, ("R", "P", "F", "D", "L", "W"))
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df
