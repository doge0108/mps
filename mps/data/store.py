"""Load/save the three core tables (games, batting_lines, pitching_lines)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import BATTING_STATS, GAMES_COLUMNS, PITCHING_STATS

TABLES = ("games", "batting_lines", "pitching_lines")


@dataclass
class Dataset:
    games: pd.DataFrame
    batting_lines: pd.DataFrame
    pitching_lines: pd.DataFrame

    def seasons(self) -> list[int]:
        return sorted(int(s) for s in self.games["season"].dropna().unique())

    def save(self, data_dir: Path) -> None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        for name in TABLES:
            getattr(self, name).to_csv(data_dir / f"{name}.csv", index=False)

    @classmethod
    def load(cls, data_dir: Path) -> "Dataset":
        data_dir = Path(data_dir)
        frames = {}
        for name in TABLES:
            path = data_dir / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run `mps fetch` (real data) or `mps simulate` (offline data) first."
                )
            frames[name] = pd.read_csv(path)
        return cls(**normalise(frames))

    def concat(self, other: "Dataset") -> "Dataset":
        frames = {}
        for name in TABLES:
            merged = pd.concat([getattr(self, name), getattr(other, name)], ignore_index=True)
            key = ["game_pk"] if name == "games" else ["game_pk", "player_id"]
            frames[name] = merged.drop_duplicates(subset=key, keep="last")
        return Dataset(**normalise(frames))


def normalise(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Coerce dtypes and sort tables chronologically."""
    games = frames["games"].copy()
    for col in GAMES_COLUMNS:
        if col not in games.columns:
            games[col] = pd.NA
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()
    games["season"] = games["season"].astype("Int64")
    for col in ("home_sp_id", "away_sp_id", "venue_id", "home_score", "away_score"):
        games[col] = pd.to_numeric(games[col], errors="coerce").astype("Int64")
    games = games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    bat = frames["batting_lines"].copy()
    bat["date"] = pd.to_datetime(bat["date"]).dt.normalize()
    bat["opp_sp_id"] = pd.to_numeric(bat.get("opp_sp_id"), errors="coerce").astype("Int64")
    for col in BATTING_STATS:
        if col in bat.columns:
            bat[col] = pd.to_numeric(bat[col], errors="coerce").astype("float64")
    bat = bat.sort_values(["date", "game_pk", "player_id"]).reset_index(drop=True)

    pit = frames["pitching_lines"].copy()
    pit["date"] = pd.to_datetime(pit["date"]).dt.normalize()
    for col in PITCHING_STATS:
        if col in pit.columns:
            pit[col] = pd.to_numeric(pit[col], errors="coerce").astype("float64")
    pit = pit.sort_values(["date", "game_pk", "player_id"]).reset_index(drop=True)
    return {"games": games, "batting_lines": bat, "pitching_lines": pit}
