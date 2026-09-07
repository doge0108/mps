"""Shared constants: paths, stat columns, team reference data."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = Path(os.environ.get("MPS_DATA_DIR", "data"))
DEFAULT_MODELS_DIR = Path(os.environ.get("MPS_MODELS_DIR", "models"))

# Box-score columns stored for every batter appearance.
BATTING_STATS = ["pa", "ab", "r", "h", "d", "t", "hr", "rbi", "bb", "so", "sb", "hbp", "tb"]
# Box-score columns stored for every pitcher appearance ("outs" = innings pitched * 3).
PITCHING_STATS = ["outs", "h", "r", "er", "bb", "so", "hr", "bf", "pitches"]

# Stats the player model predicts.
BATTING_TARGETS = ["h", "hr", "rbi", "r", "bb", "so", "sb", "tb", "ab"]
PITCHING_TARGETS = ["so", "er", "outs", "h", "bb", "hr"]

GAMES_COLUMNS = [
    "game_pk", "date", "season", "game_type",
    "home_team_id", "away_team_id", "home_score", "away_score",
    "home_sp_id", "away_sp_id", "venue_id", "status",
    # weather / time of day (NaN when unknown)
    "day_night", "temp_f", "wind_mph", "wind_dir", "condition",
]
PLAYERS_COLUMNS = ["player_id", "player_name", "bats", "throws", "position", "team_id"]
LINEUPS_COLUMNS = ["game_pk", "date", "team_id", "player_id", "batting_order"]

HAND_CODES = {"L": 1, "R": 2, "S": 3}

# MLB Stats API team ids -> (abbreviation, name).  Used for both real and simulated data.
TEAMS: dict[int, tuple[str, str]] = {
    108: ("LAA", "Los Angeles Angels"),
    109: ("ARI", "Arizona Diamondbacks"),
    110: ("BAL", "Baltimore Orioles"),
    111: ("BOS", "Boston Red Sox"),
    112: ("CHC", "Chicago Cubs"),
    113: ("CIN", "Cincinnati Reds"),
    114: ("CLE", "Cleveland Guardians"),
    115: ("COL", "Colorado Rockies"),
    116: ("DET", "Detroit Tigers"),
    117: ("HOU", "Houston Astros"),
    118: ("KC", "Kansas City Royals"),
    119: ("LAD", "Los Angeles Dodgers"),
    120: ("WSH", "Washington Nationals"),
    121: ("NYM", "New York Mets"),
    133: ("ATH", "Athletics"),
    134: ("PIT", "Pittsburgh Pirates"),
    135: ("SD", "San Diego Padres"),
    136: ("SEA", "Seattle Mariners"),
    137: ("SF", "San Francisco Giants"),
    138: ("STL", "St. Louis Cardinals"),
    139: ("TB", "Tampa Bay Rays"),
    140: ("TEX", "Texas Rangers"),
    141: ("TOR", "Toronto Blue Jays"),
    142: ("MIN", "Minnesota Twins"),
    143: ("PHI", "Philadelphia Phillies"),
    144: ("ATL", "Atlanta Braves"),
    145: ("CWS", "Chicago White Sox"),
    146: ("MIA", "Miami Marlins"),
    147: ("NYY", "New York Yankees"),
    158: ("MIL", "Milwaukee Brewers"),
}
TEAM_ABBR_TO_ID = {abbr: tid for tid, (abbr, _) in TEAMS.items()}
TEAM_NAME_TO_ID = {name.lower(): tid for tid, (_, name) in TEAMS.items()}


def team_label(team_id: int) -> str:
    """Return the abbreviation for a team id, or the id itself if unknown."""
    return TEAMS.get(int(team_id), (str(team_id), ""))[0]


def resolve_team(text: str | int) -> int:
    """Resolve an abbreviation, full name, nickname or numeric id to a team id."""
    if isinstance(text, int) or str(text).isdigit():
        return int(text)
    key = str(text).strip()
    if key.upper() in TEAM_ABBR_TO_ID:
        return TEAM_ABBR_TO_ID[key.upper()]
    low = key.lower()
    if low in TEAM_NAME_TO_ID:
        return TEAM_NAME_TO_ID[low]
    for tid, (_, name) in TEAMS.items():
        if low in name.lower():
            return tid
    raise ValueError(f"Unknown team: {text!r}")
