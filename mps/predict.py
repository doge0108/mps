"""High-level prediction API: player stat lines and game outcomes for a given date."""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import (BATTING_TARGETS, DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, PITCHING_TARGETS, resolve_team,
                     team_label)
from .data.store import Dataset
from .features.build import assemble_batter_features, assemble_game_features, assemble_pitcher_features, States
from .features.lineups import latest_lineup
from .models.game_model import summarise_game_predictions
from .models.player_model import add_probabilities
from .models.registry import ModelBundle

WEATHER_COLS = ["temp_f", "wind_mph", "wind_dir", "condition", "day_night"]


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
            name=("player_name", "last"), team_id=("team_id", "last"), last_seen=("date", "max"))
        bat["is_batter"] = True
        pit = self.ds.pitching_lines
        pit = pit[pit["is_starter"] == 1].groupby("player_id").agg(
            name=("player_name", "last"), team_id=("team_id", "last"), last_seen=("date", "max"))
        pit["is_pitcher"] = True
        idx = bat.join(pit, how="outer", lsuffix="_b", rsuffix="_p")
        idx["name"] = idx["name_b"].fillna(idx["name_p"])
        idx["last_seen"] = idx[["last_seen_b", "last_seen_p"]].max(axis=1)
        floor = pd.Timestamp.min
        idx["team_id"] = np.where(
            idx["last_seen_p"].notna() & (idx["last_seen_p"].fillna(floor) >= idx["last_seen_b"].fillna(floor)),
            idx["team_id_p"], idx["team_id_b"])
        idx["is_batter"] = idx["is_batter"].fillna(False).astype(bool)
        idx["is_pitcher"] = idx["is_pitcher"].fillna(False).astype(bool)
        idx = idx[["name", "team_id", "last_seen", "is_batter", "is_pitcher"]]
        # players.csv knows about trades / call-ups that box scores have not caught up with yet
        players = self.ds.players
        if players is not None and len(players) and "team_id" in players.columns:
            cur = players.dropna(subset=["team_id"]).set_index("player_id")["team_id"]
            cur = cur[cur.index.isin(idx.index)]
            idx.loc[cur.index, "team_id"] = cur.astype(int).to_numpy()
        return idx

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
        """Games on a date: from an explicit schedule frame or the stored games table (played + upcoming)."""
        date = pd.Timestamp(date).normalize()
        src = schedule if schedule is not None else self.ds.games
        out = src[pd.to_datetime(src["date"]).dt.normalize() == date].reset_index(drop=True)
        for col in WEATHER_COLS:
            if col not in out.columns:
                out[col] = np.nan
        return out

    def _probable_starter(self, team_id: int, date: pd.Timestamp) -> int | None:
        """Guess the next starter from the rotation order (the pitcher who started five games ago)."""
        pit = self.ds.pitching_lines
        starts = pit[(pit["team_id"] == team_id) & (pit["is_starter"] == 1) & (pit["date"] < date)]
        starts = starts.sort_values(["date", "game_pk"])
        if starts.empty:
            return None
        return int(starts["player_id"].tail(5).iloc[0])

    def _pitcher_name(self, pid: int | None) -> str | None:
        if pid is None or pd.isna(pid) or int(pid) not in self._index.index:
            return None
        return str(self._index.loc[int(pid), "name"])

    def _hand(self, pid: int | None, col: str) -> str | None:
        if pid is None or pd.isna(pid):
            return None
        val = self.states.hand(pd.Series([int(pid)]), col).iloc[0]
        return None if pd.isna(val) else str(val)

    # ------------------------------------------------------------ lineups
    def lineup_for(self, game_pk: int | None, team_id: int, date: pd.Timestamp) -> tuple[pd.DataFrame, str]:
        """Announced lineup for the game if stored, else the team's most recent starting nine."""
        lu = self.ds.lineups
        if game_pk is not None and len(lu):
            rows = lu[(lu["game_pk"] == game_pk) & (lu["team_id"] == team_id)]
            if len(rows):
                return rows[["team_id", "player_id", "batting_order"]].sort_values("batting_order"), "announced"
        return latest_lineup(self.ds.batting_lines, team_id, date), "previous_game"

    def _lineup_rows(self, game_key: int, team_id: int, date: pd.Timestamp, game_pk: int | None) -> pd.DataFrame:
        rows, _ = self.lineup_for(game_pk, team_id, date)
        rows = rows.copy()
        rows["game_pk"] = game_key
        rows["date"] = date
        return rows[["game_pk", "date", "team_id", "player_id"]]

    # ------------------------------------------------------- game context
    def resolve_game_context(self, team_id: int, date: pd.Timestamp, opponent: str | int | None = None,
                             is_home: int | None = None, schedule: pd.DataFrame | None = None) -> dict:
        games = self.games_on(date, schedule)
        game = games[(games["home_team_id"] == team_id) | (games["away_team_id"] == team_id)]
        if len(game):
            g = game.iloc[0]
            home = int(g["home_team_id"]) == team_id
            opp = int(g["away_team_id"] if home else g["home_team_id"])
            opp_sp = g["away_sp_id" if home else "home_sp_id"]
            own_sp = g["home_sp_id" if home else "away_sp_id"]
            return {"source": "schedule", "game_pk": int(g["game_pk"]), "is_home": int(home), "opp_team_id": opp,
                    "opp_sp_id": self._probable_starter(opp, date) if pd.isna(opp_sp) else int(opp_sp),
                    "own_sp_id": self._probable_starter(team_id, date) if pd.isna(own_sp) else int(own_sp),
                    "weather": {c: (None if pd.isna(g[c]) else g[c]) for c in WEATHER_COLS},
                    "status": g.get("status")}
        if opponent is not None:
            opp = resolve_team(opponent)
            home = 1 if is_home is None else int(is_home)
            return {"source": "manual", "game_pk": None, "is_home": home, "opp_team_id": opp,
                    "opp_sp_id": self._probable_starter(opp, date),
                    "own_sp_id": self._probable_starter(team_id, date),
                    "weather": {c: None for c in WEATHER_COLS}, "status": None}
        return {"source": "unknown", "game_pk": None, "is_home": 1 if is_home is None else int(is_home),
                "opp_team_id": None, "opp_sp_id": None, "own_sp_id": None,
                "weather": {c: None for c in WEATHER_COLS}, "status": None}

    # --------------------------------------------------------------- form
    def batter_form(self, player_id: int, date: pd.Timestamp) -> dict | None:
        st = self.states.batters
        rows = st[(st["player_id"] == player_id) & (st["date"] < date)]
        if rows.empty:
            return None
        s = rows.iloc[-1]
        lines = self.ds.batting_lines
        last5 = lines[(lines["player_id"] == player_id) & (lines["date"] < date)].sort_values("date").tail(5)
        form_slg = float(s["b_form_slg"]) if pd.notna(s["b_form_slg"]) else 0.0
        label = "hot" if form_slg > 0.10 else "cold" if form_slg < -0.10 else "steady"
        return {
            "label": label,
            "last5": {"games": int(len(last5)), "ab": int(last5["ab"].sum()), "h": int(last5["h"].sum()),
                      "hr": int(last5["hr"].sum()), "rbi": int(last5["rbi"].sum()), "bb": int(last5["bb"].sum()),
                      "so": int(last5["so"].sum()), "sb": int(last5["sb"].sum()),
                      "avg": _safe_round(s["b_avg_r5"]), "slg": _safe_round(s["b_slg_r5"])},
            "baseline30": {"avg": _safe_round(s["b_avg_r30"]), "slg": _safe_round(s["b_slg_r30"]),
                           "hr_rate": _safe_round(s["b_hr_rate_r30"]), "k_rate": _safe_round(s["b_k_rate_r30"])},
            "slg_vs_baseline": round(form_slg, 3),
            "hit_streak": int(s["b_hit_streak"]),
            "hitless_streak": int(s["b_hitless_streak"]),
            "hr_drought": int(s["b_hr_drought"]),
            "days_since_last_game": int((date - pd.Timestamp(s["b_last_date"])).days),
        }

    def pitcher_form(self, player_id: int, date: pd.Timestamp) -> dict | None:
        st = self.states.pitchers
        rows = st[(st["player_id"] == player_id) & (st["date"] < date)]
        if rows.empty:
            return None
        s = rows.iloc[-1]
        form = float(s["p_form_era"]) if pd.notna(s["p_form_era"]) else 0.0
        label = "hot" if form < -1.0 else "cold" if form > 1.0 else "steady"
        return {
            "label": label,
            "last3": {"era": _safe_round(s["p_era_r3"], 2), "k_rate": _safe_round(s["p_k_rate_r3"]),
                      "outs_per_start": _safe_round(s["p_outs_r3"], 1)},
            "baseline30": {"era": _safe_round(s["p_era_r30"], 2), "k_rate": _safe_round(s["p_k_rate_r30"]),
                           "outs_per_start": _safe_round(s["p_outs_r30"], 1)},
            "era_vs_baseline": round(form, 2),
            "quality_start_streak": int(s["p_qs_streak"]),
            "days_rest": int((date - pd.Timestamp(s["p_last_date"])).days),
        }

    def batter_split(self, player_id: int, hand: str | None, date: pd.Timestamp) -> dict | None:
        if hand not in ("L", "R") or self.states.splits.empty:
            return None
        from .features.states import split_key
        key = int(split_key(pd.Series([player_id]), pd.Series([hand])).iloc[0])
        st = self.states.splits
        rows = st[(st["split_key"] == key) & (st["date"] < date)]
        if rows.empty:
            return None
        s = rows.iloc[-1]
        return {"vs_hand": hand, "games": int(s["sp_games"]), "avg": _safe_round(s["sp_avg"]),
                "slg": _safe_round(s["sp_slg"]), "hr_rate": _safe_round(s["sp_hr_rate"]),
                "k_rate": _safe_round(s["sp_k_rate"])}

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
            "weather": ctx["weather"] if any(v is not None for v in ctx["weather"].values()) else None,
        }
        wx = {c: ctx["weather"].get(c) for c in WEATHER_COLS}
        opp_team = ctx["opp_team_id"] if ctx["opp_team_id"] is not None else -1
        if match.is_batter:
            lineup, lineup_source = self.lineup_for(ctx["game_pk"], match.team_id, date)
            in_lineup = None
            announced_order = None
            if lineup_source == "announced":
                slot = lineup[lineup["player_id"] == match.player_id]
                in_lineup = bool(len(slot))
                announced_order = int(slot["batting_order"].iloc[0]) if len(slot) else None
            last_order = self.states.batters.loc[self.states.batters["player_id"] == match.player_id, "b_last_order"]
            order = (batting_order if batting_order is not None else announced_order
                     if announced_order is not None else int(last_order.iloc[-1]) if len(last_order) else 5)
            spec = pd.DataFrame([{
                "date": date, "player_id": match.player_id, "team_id": match.team_id, "opp_team_id": opp_team,
                "opp_sp_id": ctx["opp_sp_id"], "is_home": ctx["is_home"], "batting_order": order, **wx,
            }])
            feats = assemble_batter_features(spec, self.states)
            pred = add_probabilities(self.models.batter.predict(feats))
            row = pred.iloc[0]
            opp_hand = self._hand(ctx["opp_sp_id"], "throws")
            bats = self._hand(match.player_id, "bats")
            result["batting"] = {
                "opposing_starter": self._pitcher_name(ctx["opp_sp_id"]),
                "opposing_starter_hand": opp_hand,
                "bats": bats,
                "platoon_advantage": (None if bats is None or opp_hand is None
                                      else bats == "S" or bats != opp_hand),
                "split_vs_hand": self.batter_split(match.player_id, opp_hand, date),
                "batting_order": order,
                "lineup": {"source": lineup_source, "in_lineup": in_lineup},
                "form": self.batter_form(match.player_id, date),
                "expected": {t: round(float(row[t]), 3) for t in BATTING_TARGETS},
                "probabilities": {
                    "hit": round(float(row["p_h_ge1"]), 3), "multi_hit": round(float(row["p_h_ge2"]), 3),
                    "home_run": round(float(row["p_hr_ge1"]), 3), "rbi": round(float(row["p_rbi_ge1"]), 3),
                    "run": round(float(row["p_r_ge1"]), 3), "walk": round(float(row["p_bb_ge1"]), 3),
                    "stolen_base": round(float(row["p_sb_ge1"]), 3), "strikeout": round(float(row["p_so_ge1"]), 3),
                },
            }
        if match.is_pitcher:
            key = -1
            lineups = None
            if ctx["opp_team_id"] is not None:
                lineups = self._lineup_rows(key, ctx["opp_team_id"], date, ctx["game_pk"])
            spec = pd.DataFrame([{
                "game_pk": key, "date": date, "player_id": match.player_id, "team_id": match.team_id,
                "opp_team_id": opp_team, "is_home": ctx["is_home"], **wx,
            }])
            feats = assemble_pitcher_features(spec, self.states, lineups)
            row = self.models.pitcher.predict(feats).iloc[0]
            outs = float(row["outs"])
            result["pitching"] = {
                "throws": self._hand(match.player_id, "throws"),
                "form": self.pitcher_form(match.player_id, date),
                "opposing_lineup_lhb_share": _safe_round(feats["opplu_lhb_share"].iloc[0], 2),
                "expected": {t: round(float(row[t]), 3) for t in PITCHING_TARGETS},
                "innings_pitched": f"{int(outs // 3)}.{int(round(outs % 3))}",
                "probabilities": {
                    "quality_start": round(_quality_start_prob(outs, float(row["er"])), 3),
                    "strikeouts_ge6": round(float(sps.poisson.sf(5, max(float(row["so"]), 1e-6))), 3),
                    "strikeouts_ge8": round(float(sps.poisson.sf(7, max(float(row["so"]), 1e-6))), 3),
                },
            }
        return result

    # -------------------------------------------------------------- games
    def predict_games(self, date: str | pd.Timestamp, schedule: pd.DataFrame | None = None,
                      home: str | int | None = None, away: str | int | None = None) -> pd.DataFrame:
        date = pd.Timestamp(date).normalize()
        if home is not None and away is not None:
            h, a = resolve_team(home), resolve_team(away)
            games = pd.DataFrame([{
                "game_pk": np.nan, "date": date, "home_team_id": h, "away_team_id": a,
                "home_sp_id": self._probable_starter(h, date), "away_sp_id": self._probable_starter(a, date),
                **{c: np.nan for c in WEATHER_COLS},
            }])
        else:
            games = self.games_on(date, schedule)
            if games.empty:
                raise KeyError(f"No games found on {date.date()}. Run `mps update`, pass --live, or --home/--away.")
            games = games.copy()
            for side in ("home", "away"):
                col = f"{side}_sp_id"
                missing = games[col].isna()
                games.loc[missing, col] = [self._probable_starter(int(t), date)
                                           for t in games.loc[missing, f"{side}_team_id"]]
        games = games.reset_index(drop=True)
        keys = -(np.arange(len(games)) + 1)  # private keys so manual matchups also get lineup features
        lineup_rows = []
        lineup_src = []
        for key, g in zip(keys, games.itertuples(index=False)):
            pk = None if pd.isna(g.game_pk) else int(g.game_pk)
            for tid in (int(g.home_team_id), int(g.away_team_id)):
                lineup_rows.append(self._lineup_rows(key, tid, date, pk))
                lineup_src.append(self.lineup_for(pk, tid, date)[1])
        lineups = pd.concat(lineup_rows, ignore_index=True) if lineup_rows else None
        spec = games[["date", "home_team_id", "away_team_id", "home_sp_id", "away_sp_id"] + WEATHER_COLS].copy()
        spec.insert(0, "game_pk", keys)
        feats = assemble_game_features(spec, self.states, lineups)
        pred = summarise_game_predictions(self.models.game.predict(feats))
        home_lbl = [team_label(t) for t in games["home_team_id"]]
        away_lbl = [team_label(t) for t in games["away_team_id"]]
        out = pd.DataFrame({
            "date": date.strftime("%Y-%m-%d"),
            "home": home_lbl, "away": away_lbl,
            "home_sp": [self._pitcher_name(p) for p in games["home_sp_id"]],
            "away_sp": [self._pitcher_name(p) for p in games["away_sp_id"]],
            "home_win_prob": pred["home_win_prob"].round(3),
            "away_win_prob": (1 - pred["home_win_prob"]).round(3),
            "exp_home_runs": pred["home_runs"].round(2),
            "exp_away_runs": pred["away_runs"].round(2),
            "exp_total_runs": pred["total_runs"].round(2),
            "predicted_winner": np.where(pred["home_win_prob"] >= 0.5, home_lbl, away_lbl),
            "lineups": [f"{lineup_src[2 * i][:4]}/{lineup_src[2 * i + 1][:4]}" for i in range(len(games))],
            "weather": [_weather_text(g) for g in games.to_dict("records")],
        })
        if "home_score" in games.columns and games["home_score"].notna().any():
            out["actual"] = [
                f"{int(hs)}-{int(as_)}" if not (pd.isna(hs) or pd.isna(as_)) else ""
                for hs, as_ in zip(games["home_score"], games["away_score"])]
        return out


def _safe_round(x, nd: int = 3):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def _weather_text(g: dict) -> str:
    parts = []
    if g.get("temp_f") is not None and not pd.isna(g.get("temp_f")):
        parts.append(f"{int(g['temp_f'])}F")
    wd = g.get("wind_dir")
    if wd is not None and not pd.isna(wd):
        mph = g.get("wind_mph")
        parts.append(f"wind {'' if mph is None or pd.isna(mph) else str(int(mph)) + 'mph '}{wd}")
    cond = g.get("condition")
    if cond is not None and not pd.isna(cond):
        parts.append(str(cond))
    return ", ".join(parts)


def _quality_start_prob(exp_outs: float, exp_er: float) -> float:
    """P(outs >= 18 and ER <= 3) treating the two as independent Poissons (rough but useful)."""
    p_outs = 1 - sps.poisson.cdf(17, max(exp_outs, 1e-6))
    p_er = sps.poisson.cdf(3, max(exp_er, 1e-6))
    return float(p_outs * p_er)
