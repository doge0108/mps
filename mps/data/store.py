"""Load/save the core tables (games, batting_lines, pitching_lines, players, lineups)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import BATTING_STATS, GAMES_COLUMNS, LINEUPS_COLUMNS, PITCHING_STATS, PLAYERS_COLUMNS

TABLES = ("games", "batting_lines", "pitching_lines", "players", "lineups")
REQUIRED = ("games", "batting_lines", "pitching_lines")
KEYS = {
    "games": ["game_pk"], "batting_lines": ["game_pk", "player_id"], "pitching_lines": ["game_pk", "player_id"],
    "players": ["player_id"], "lineups": ["game_pk", "player_id"],
}


def empty_players() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in PLAYERS_COLUMNS})


def empty_lineups() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in LINEUPS_COLUMNS})


@dataclass
class Dataset:
    games: pd.DataFrame
    batting_lines: pd.DataFrame
    pitching_lines: pd.DataFrame
    players: pd.DataFrame = field(default_factory=empty_players)
    lineups: pd.DataFrame = field(default_factory=empty_lineups)

    def seasons(self) -> list[int]:
        played = self.games[self.games["home_score"].notna()]
        return sorted(int(s) for s in played["season"].dropna().unique())

    def played_games(self) -> pd.DataFrame:
        return self.games[self.games["home_score"].notna() & self.games["away_score"].notna()]

    def upcoming_games(self) -> pd.DataFrame:
        return self.games[self.games["home_score"].isna() | self.games["away_score"].isna()]

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
            if path.exists():
                frames[name] = pd.read_csv(path)
            elif name in REQUIRED:
                raise FileNotFoundError(
                    f"{path} not found. Run `mps fetch` (real data) or `mps simulate` (offline data) first."
                )
        return cls(**normalise(frames))

    def concat(self, other: "Dataset") -> "Dataset":
        frames = {}
        for name in TABLES:
            merged = pd.concat([getattr(self, name), getattr(other, name)], ignore_index=True)
            frames[name] = merged.drop_duplicates(subset=KEYS[name], keep="last")
        return Dataset(**normalise(frames))


def normalise(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Coerce dtypes and sort tables chronologically."""
    games = frames["games"].copy()
    for col in GAMES_COLUMNS:
        if col not in games.columns:
            games[col] = pd.NA
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()
    games["season"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    for col in ("home_sp_id", "away_sp_id", "venue_id", "home_score", "away_score"):
        games[col] = pd.to_numeric(games[col], errors="coerce").astype("Int64")
    for col in ("temp_f", "wind_mph"):
        games[col] = pd.to_numeric(games[col], errors="coerce").astype("float64")
    games = games.sort_values(["date", "game_pk"]).reset_index(drop=True)

    bat = frames["batting_lines"].copy()
    for col in ("game_pk", "date", "player_id", "player_name", "team_id", "opp_team_id", "is_home",
                "batting_order", "opp_sp_id", *BATTING_STATS):
        if col not in bat.columns:
            bat[col] = pd.Series(dtype="float64" if col != "date" else "datetime64[ns]")
    bat["date"] = pd.to_datetime(bat["date"]).dt.normalize()
    bat["opp_sp_id"] = pd.to_numeric(bat.get("opp_sp_id"), errors="coerce").astype("Int64")
    if "bat_starter" not in bat.columns:
        bat["bat_starter"] = (bat["batting_order"].fillna(0) > 0).astype(int)
    for col in BATTING_STATS:
        if col in bat.columns:
            bat[col] = pd.to_numeric(bat[col], errors="coerce").astype("float64")
    bat = bat.sort_values(["date", "game_pk", "player_id"]).reset_index(drop=True)

    pit = frames["pitching_lines"].copy()
    for col in ("game_pk", "date", "player_id", "player_name", "team_id", "opp_team_id", "is_home",
                "is_starter", *PITCHING_STATS):
        if col not in pit.columns:
            pit[col] = pd.Series(dtype="float64" if col != "date" else "datetime64[ns]")
    pit["date"] = pd.to_datetime(pit["date"]).dt.normalize()
    for col in PITCHING_STATS:
        if col in pit.columns:
            pit[col] = pd.to_numeric(pit[col], errors="coerce").astype("float64")
    pit = pit.sort_values(["date", "game_pk", "player_id"]).reset_index(drop=True)

    players = frames.get("players")
    players = empty_players() if players is None or players.empty else players.copy()
    for col in PLAYERS_COLUMNS:
        if col not in players.columns:
            players[col] = pd.NA
    if len(players):
        players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce").astype("int64")
        players = players.drop_duplicates("player_id", keep="last").reset_index(drop=True)

    lineups = frames.get("lineups")
    lineups = empty_lineups() if lineups is None or lineups.empty else lineups.copy()
    for col in LINEUPS_COLUMNS:
        if col not in lineups.columns:
            lineups[col] = pd.NA
    if len(lineups):
        lineups["date"] = pd.to_datetime(lineups["date"]).dt.normalize()
        for col in ("game_pk", "team_id", "player_id", "batting_order"):
            lineups[col] = pd.to_numeric(lineups[col], errors="coerce").astype("int64")
        lineups = lineups.sort_values(["date", "game_pk", "team_id", "batting_order"]).reset_index(drop=True)
    return {"games": games, "batting_lines": bat, "pitching_lines": pit, "players": players, "lineups": lineups}
