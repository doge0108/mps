"""Team game log and Elo constants shared by the state builders."""
from __future__ import annotations

import numpy as np
import pandas as pd

ELO_K = 4.0
ELO_HOME = 24.0
ELO_START = 1500.0
ELO_SEASON_REGRESS = 0.33  # regress a third of the way to the mean between seasons


def team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game) with runs scored/allowed from the team's perspective."""
    g = games.sort_values(["date", "game_pk"]).reset_index(drop=True)
    home = pd.DataFrame({
        "game_pk": g["game_pk"], "date": g["date"], "season": g["season"],
        "team_id": g["home_team_id"], "opp_team_id": g["away_team_id"], "is_home": 1,
        "rs": g["home_score"].astype("float"), "ra": g["away_score"].astype("float"),
        "sp_id": g["home_sp_id"],
    })
    away = pd.DataFrame({
        "game_pk": g["game_pk"], "date": g["date"], "season": g["season"],
        "team_id": g["away_team_id"], "opp_team_id": g["home_team_id"], "is_home": 0,
        "rs": g["away_score"].astype("float"), "ra": g["home_score"].astype("float"),
        "sp_id": g["away_sp_id"],
    })
    log = pd.concat([home, away], ignore_index=True)
    log["win"] = np.where(log["rs"].isna(), np.nan, (log["rs"] > log["ra"]).astype(float))
    log["run_diff"] = log["rs"] - log["ra"]
    log["total_runs"] = log["rs"] + log["ra"]
    return log.sort_values(["date", "game_pk", "is_home"]).reset_index(drop=True)
