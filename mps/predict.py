"""High-level prediction API: player stat lines and game outcomes for a given date."""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BATTING_TARGETS, PITCHING_TARGETS, DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, resolve_team, team_label
from .data.store import Dataset
from .features.build import States, assemble_batter_features, assemble_game_features, assemble_pitcher_features
from .models.game_model import summarise_game_predictions
from .models.player_model import add_probabilities
from .models.registry import ModelBundle


@dataclass
class PlayerMatch:
    player_id: int
    name: str
    team_id: int
    is_batter: bool
    is_pitcher: bool
    last_seen: pd.Timestamp


@dataclass
class Predictor:
    """Loads data + models once, then answers prediction questions."""

    ds: Dataset
    models: ModelBundle
    states: States = field(init=False)
    _index: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.states = States.from_dataset(self.ds)
        self._index = self._build_player_index()

    @classmethod
    def load(cls, data_dir: Path = DEFAULT_DATA_DIR, models_dir: Path = DEFAULT_MODELS_DIR) -> "Predictor":
        return cls(ds=Dataset.load(data_dir), models=ModelBundle.load(models_dir))

    # ------------------------------------------------------------ lookups
    def _build_player_index(self) -> pd.DataFrame:
        bat = self.ds.batting_lines.groupby("player_id").agg(
            name=("player_name", "last"), team_id=("team_id", "last"), last_seen=("date", "max"),
            games=("game_pk", "count"))
        bat["is_batter"] = True
        pit = self.ds.pitching_lines
        pit = pit[pit["is_starter"] == 1].groupby("player_id").agg(
            name=("player_name", "last"), team_id=("team_id", "last"), last_seen=("date", "max"),
            games=("game_pk", "count"))
        pit["is_pitcher"] = True
        idx = bat.join(pit, how="outer", lsuffix="_b", rsuffix="_p")
        idx["name"] = idx["name_b"].fillna(idx["name_p"])
        idx["last_seen"] = idx[["last_seen_b", "last_seen_p"]].max(axis=1)
        # current team = team of the most recent appearance of either kind
        idx["team_id"] = np.where(
            idx["last_seen_p"].notna() & (idx["last_seen_p"].fillna(pd.Timestamp.min) >= idx["last_seen_b"].fillna(pd.Timestamp.min)),
            idx["team_id_p"], idx["team_id_b"])
        idx["is_batter"] = idx["is_batter"].fillna(False).astype(bool)
        idx["is_pitcher"] = idx["is_pitcher"].fillna(False).astype(bool)
        return idx[["name", "team_id", "last_seen", "is_batter", "is_pitcher"]]

    def find_player(self, query: str) -> PlayerMatch:
        idx = self._index
        if str(query).isdigit() and int(query) in idx.index:
            pid = int(query)
        else:
            names = idx["name"].astype(str)
            exact = idx[names.str.lower() == str(query).lower()]
            if len(exact):
                pid = int(exact.sort_values("last_seen").index[-1])
            else:
                lowered = {n.lower(): i for i, n in names.items()}
                close = difflib.get_close_matches(str(query).lower(), list(lowered), n=1, cutoff=0.6)
                if not close:
                    raise KeyError(f"No player matching {query!r} in the dataset")
                pid = int(lowered[close[0]])
        row = idx.loc[pid]
        return PlayerMatch(pid, str(row["name"]), int(row["team_id"]), bool(row["is_batter"]),
                           bool(row["is_pitcher"]), pd.Timestamp(row["last_seen"]))

    def search_players(self, query: str, n: int = 5) -> pd.DataFrame:
        names = self._index["name"].astype(str)
        mask = names.str.lower().str.contains(str(query).lower(), regex=False)
        return self._index[mask].sort_values("last_seen", ascending=False).head(n)

    def games_on(self, date: str | pd.Timestamp, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
        """Games scheduled on a date: from an explicit schedule frame or the stored games table."""
        date = pd.Timestamp(date).normalize()
        src = schedule if schedule is not None else self.ds.games
        return src[pd.to_datetime(src["date"]).dt.normalize() == date].reset_index(drop=True)

    def _probable_starter(self, team_id: int, date: pd.Timestamp) -> int | None:
        """Guess the next starter from the rotation order (next after the most recent 5 starters)."""
        pit = self.ds.pitching_lines
        starts = pit[(pit["team_id"] == team_id) & (pit["is_starter"] == 1) & (pit["date"] < date)]
        starts = starts.sort_values(["date", "game_pk"])
        if starts.empty:
            return None
        recent = starts["player_id"].tail(5).tolist()
        if len(recent) < 5:
            return int(recent[0])
        return int(recent[0])  # five-man rotation: the pitcher who started 5 games ago

    # ------------------------------------------------------- game context
    def resolve_game_context(self, team_id: int, date: pd.Timestamp, opponent: str | int | None = None,
                             is_home: int | None = None, schedule: pd.DataFrame | None = None) -> dict:
        games = self.games_on(date, schedule)
        game = games[(games["home_team_id"] == team_id) | (games["away_team_id"] == team_id)]
        if len(game):
            g = game.iloc[0]
            home = int(g["home_team_id"]) == team_id
            opp = int(g["away_team_id"] if home else g["home_team_id"])
            sp_col = "away_sp_id" if home else "home_sp_id"
            opp_sp = g[sp_col]
            opp_sp = None if pd.isna(opp_sp) else int(opp_sp)
            own_sp = g["home_sp_id" if home else "away_sp_id"]
            own_sp = None if pd.isna(own_sp) else int(own_sp)
            return {"source": "schedule", "game_pk": int(g["game_pk"]), "is_home": int(home), "opp_team_id": opp,
                    "opp_sp_id": opp_sp if opp_sp is not None else self._probable_starter(opp, date),
                    "own_sp_id": own_sp if own_sp is not None else self._probable_starter(team_id, date)}
        if opponent is not None:
            opp = resolve_team(opponent)
            home = 1 if is_home is None else int(is_home)
            return {"source": "manual", "game_pk": None, "is_home": home, "opp_team_id": opp,
                    "opp_sp_id": self._probable_starter(opp, date),
                    "own_sp_id": self._probable_starter(team_id, date)}
        return {"source": "unknown", "game_pk": None, "is_home": 1 if is_home is None else int(is_home),
                "opp_team_id": None, "opp_sp_id": None, "own_sp_id": None}

    # ------------------------------------------------------------ players
    def predict_player(self, player: str, date: str | pd.Timestamp, opponent: str | int | None = None,
                       is_home: int | None = None, batting_order: int | None = None,
                       schedule: pd.DataFrame | None = None) -> dict:
        date = pd.Timestamp(date).normalize()
        match = self.find_player(player)
        ctx = self.resolve_game_context(match.team_id, date, opponent, is_home, schedule)
        result: dict = {
            "player": match.name, "player_id": match.player_id, "team": team_label(match.team_id),
            "date": date.strftime("%Y-%m-%d"),
            "opponent": team_label(ctx["opp_team_id"]) if ctx["opp_team_id"] is not None else None,
            "is_home": bool(ctx["is_home"]), "context_source": ctx["source"],
        }
        if match.is_batter:
            last_order = self.states.batters.loc[self.states.batters["player_id"] == match.player_id, "b_last_order"]
            order = batting_order if batting_order is not None else (int(last_order.iloc[-1]) if len(last_order) else 5)
            spec = pd.DataFrame([{
                "date": date, "player_id": match.player_id, "team_id": match.team_id,
                "opp_team_id": ctx["opp_team_id"] if ctx["opp_team_id"] is not None else -1,
                "opp_sp_id": ctx["opp_sp_id"], "is_home": ctx["is_home"], "batting_order": order,
            }])
            feats = assemble_batter_features(spec, self.states)
            pred = add_probabilities(self.models.batter.predict(feats))
            row = pred.iloc[0]
            result["batting"] = {
                "opposing_starter": self._pitcher_name(ctx["opp_sp_id"]),
                "expected": {t: round(float(row[t]), 3) for t in BATTING_TARGETS},
                "probabilities": {
                    "hit": round(float(row["p_h_ge1"]), 3), "multi_hit": round(float(row["p_h_ge2"]), 3),
                    "home_run": round(float(row["p_hr_ge1"]), 3), "rbi": round(float(row["p_rbi_ge1"]), 3),
                    "run": round(float(row["p_r_ge1"]), 3), "walk": round(float(row["p_bb_ge1"]), 3),
                    "stolen_base": round(float(row["p_sb_ge1"]), 3), "strikeout": round(float(row["p_so_ge1"]), 3),
                },
            }
        if match.is_pitcher:
            spec = pd.DataFrame([{
                "date": date, "player_id": match.player_id, "team_id": match.team_id,
                "opp_team_id": ctx["opp_team_id"] if ctx["opp_team_id"] is not None else -1,
                "is_home": ctx["is_home"],
            }])
            feats = assemble_pitcher_features(spec, self.states)
            pred = self.models.pitcher.predict(feats)
            row = pred.iloc[0]
            outs = float(row["outs"])
            result["pitching"] = {
                "expected": {t: round(float(row[t]), 3) for t in PITCHING_TARGETS},
                "innings_pitched": f"{int(outs // 3)}.{int(round(outs % 3))}",
                "probabilities": {
                    "quality_start": round(float(_quality_start_prob(outs, float(row["er"]))), 3),
                    "strikeouts_ge6": round(float(1 - _poisson_cdf(5, float(row["so"]))), 3),
                    "strikeouts_ge8": round(float(1 - _poisson_cdf(7, float(row["so"]))), 3),
                },
            }
        return result

    def _pitcher_name(self, pid: int | None) -> str | None:
        if pid is None or pid not in self._index.index:
            return None
        return str(self._index.loc[pid, "name"])

    # -------------------------------------------------------------- games
    def predict_games(self, date: str | pd.Timestamp, schedule: pd.DataFrame | None = None,
                      home: str | int | None = None, away: str | int | None = None) -> pd.DataFrame:
        date = pd.Timestamp(date).normalize()
        if home is not None and away is not None:
            h, a = resolve_team(home), resolve_team(away)
            games = pd.DataFrame([{
                "game_pk": None, "date": date, "home_team_id": h, "away_team_id": a,
                "home_sp_id": self._probable_starter(h, date), "away_sp_id": self._probable_starter(a, date),
            }])
        else:
            games = self.games_on(date, schedule)
            if games.empty:
                raise KeyError(f"No games found on {date.date()}. Pass --home/--away or ingest that day's schedule.")
            games = games.copy()
            for side in ("home", "away"):
                col = f"{side}_sp_id"
                missing = games[col].isna()
                games.loc[missing, col] = [self._probable_starter(int(t), date)
                                           for t in games.loc[missing, f"{side}_team_id"]]
        spec = games[["game_pk", "date", "home_team_id", "away_team_id", "home_sp_id", "away_sp_id"]].copy()
        feats = assemble_game_features(spec, self.states)
        pred = summarise_game_predictions(self.models.game.predict(feats))
        out = pd.DataFrame({
            "date": date.strftime("%Y-%m-%d"),
            "home": [team_label(t) for t in games["home_team_id"]],
            "away": [team_label(t) for t in games["away_team_id"]],
            "home_sp": [self._pitcher_name(None if pd.isna(p) else int(p)) for p in games["home_sp_id"]],
            "away_sp": [self._pitcher_name(None if pd.isna(p) else int(p)) for p in games["away_sp_id"]],
            "home_win_prob": pred["home_win_prob"].round(3),
            "away_win_prob": (1 - pred["home_win_prob"]).round(3),
            "exp_home_runs": pred["home_runs"].round(2),
            "exp_away_runs": pred["away_runs"].round(2),
            "exp_total_runs": pred["total_runs"].round(2),
            "predicted_winner": np.where(pred["home_win_prob"] >= 0.5,
                                         [team_label(t) for t in games["home_team_id"]],
                                         [team_label(t) for t in games["away_team_id"]]),
        })
        if "home_score" in games.columns and games["home_score"].notna().any():
            out["actual"] = [
                f"{int(hs)}-{int(as_)}" if not (pd.isna(hs) or pd.isna(as_)) else ""
                for hs, as_ in zip(games["home_score"], games["away_score"])]
        return out


def _poisson_cdf(k: int, mu: float) -> float:
    from scipy import stats as sps
    return float(sps.poisson.cdf(k, max(mu, 1e-6)))


def _quality_start_prob(exp_outs: float, exp_er: float) -> float:
    """P(outs >= 18 and ER <= 3) treating the two as independent Poissons (rough but useful)."""
    from scipy import stats as sps
    p_outs = 1 - sps.poisson.cdf(17, max(exp_outs, 1e-6))
    p_er = sps.poisson.cdf(3, max(exp_er, 1e-6))
    return float(p_outs * p_er)
