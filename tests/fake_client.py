"""An MLBStatsClient stand-in that serves a simulated Dataset through the real API shapes.

Used to test ingestion end to end (schedule -> boxscore -> weather -> players) without network.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from mps.data.mlb_api import MLBStatsClient
from mps.data.store import Dataset

_BAT_KEYS = {"pa": "plateAppearances", "ab": "atBats", "r": "runs", "h": "hits", "d": "doubles", "t": "triples",
             "hr": "homeRuns", "rbi": "rbi", "bb": "baseOnBalls", "so": "strikeOuts", "sb": "stolenBases",
             "hbp": "hitByPitch", "tb": "totalBases"}
_PIT_KEYS = {"h": "hits", "r": "runs", "er": "earnedRuns", "bb": "baseOnBalls", "so": "strikeOuts",
             "hr": "homeRuns", "bf": "battersFaced", "pitches": "numberOfPitches"}


class FakeClient(MLBStatsClient):
    def __init__(self, ds: Dataset, today: date, upcoming: pd.DataFrame | None = None,
                 lineups: pd.DataFrame | None = None, announce_in_schedule: bool = True):
        super().__init__(cache_dir=None, pause=0, today=today)
        self.ds = ds
        self.upcoming = upcoming if upcoming is not None else pd.DataFrame()
        self.lineups = lineups if lineups is not None else pd.DataFrame()
        self.announce_in_schedule = announce_in_schedule
        self.calls: list[str] = []

    # ----------------------------------------------------------- payloads
    def _game_json(self, g: pd.Series, final: bool, with_lineups: bool) -> dict:
        d = g["date"].strftime("%Y-%m-%d")
        out = {
            "gamePk": int(g["game_pk"]), "officialDate": d, "season": str(int(g["season"])), "gameType": "R",
            "dayNight": g.get("day_night"),
            "status": {"abstractGameState": "Final" if final else "Preview"},
            "venue": {"id": int(g["home_team_id"])},
            "teams": {"home": {"team": {"id": int(g["home_team_id"])}},
                      "away": {"team": {"id": int(g["away_team_id"])}}},
        }
        if final:
            out["teams"]["home"]["score"] = int(g["home_score"])
            out["teams"]["away"]["score"] = int(g["away_score"])
        for side in ("home", "away"):
            sp = g.get(f"{side}_sp_id")
            if sp is not None and not pd.isna(sp):
                out["teams"][side]["probablePitcher"] = {"id": int(sp)}
        if with_lineups and self.announce_in_schedule and len(self.lineups):
            lu = self.lineups[self.lineups["game_pk"] == g["game_pk"]]
            if len(lu):
                out["lineups"] = {}
                for side in ("home", "away"):
                    rows = lu[lu["team_id"] == g[f"{side}_team_id"]].sort_values("batting_order")
                    out["lineups"][f"{side}Players"] = [{"id": int(p)} for p in rows["player_id"]]
        return out

    def schedule(self, start_date, end_date, game_types=("R",), with_lineups=False):
        self.calls.append(f"schedule {start_date} {end_date}")
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        played = self.ds.played_games()
        played = played[(played["date"] >= start) & (played["date"] <= end)]
        games = [self._game_json(g, True, with_lineups) for _, g in played.iterrows()]
        if len(self.upcoming):
            up = self.upcoming[(self.upcoming["date"] >= start) & (self.upcoming["date"] <= end)]
            games += [self._game_json(g, False, with_lineups) for _, g in up.iterrows()]
        by_day: dict[str, list] = {}
        for gj in games:
            by_day.setdefault(gj["officialDate"], []).append(gj)
        return {"dates": [{"date": d, "games": gs} for d, gs in sorted(by_day.items())]}

    def boxscore(self, game_pk, final=True):
        self.calls.append(f"boxscore {game_pk}")
        g = self.ds.games[self.ds.games["game_pk"] == game_pk]
        if g.empty:  # upcoming game: lineups only (if announced) and no stats
            up = self.upcoming[self.upcoming["game_pk"] == game_pk].iloc[0]
            teams = {}
            for side in ("home", "away"):
                tid = int(up[f"{side}_team_id"])
                players = {}
                lu = self.lineups[(self.lineups["game_pk"] == game_pk) & (self.lineups["team_id"] == tid)]
                for _, r in lu.iterrows():
                    players[f"ID{int(r['player_id'])}"] = {"person": {"id": int(r["player_id"]), "fullName": "x"},
                                                           "battingOrder": str(int(r["batting_order"]) * 100),
                                                           "stats": {}}
                teams[side] = {"team": {"id": tid}, "pitchers": [], "players": players}
            return {"teams": teams}
        g = g.iloc[0]
        teams = {}
        for side in ("home", "away"):
            tid = int(g[f"{side}_team_id"])
            bl = self.ds.batting_lines[(self.ds.batting_lines["game_pk"] == game_pk) & (self.ds.batting_lines["team_id"] == tid)]
            pl = self.ds.pitching_lines[(self.ds.pitching_lines["game_pk"] == game_pk) & (self.ds.pitching_lines["team_id"] == tid)]
            players = {}
            for _, r in bl.iterrows():
                players[f"ID{int(r['player_id'])}"] = {
                    "person": {"id": int(r["player_id"]), "fullName": r["player_name"]},
                    "battingOrder": str(int(r["batting_order"]) * 100),
                    "stats": {"batting": {api: int(r[c]) for c, api in _BAT_KEYS.items()}},
                }
            pitchers = pl.sort_values("is_starter", ascending=False)
            for _, r in pitchers.iterrows():
                entry = players.setdefault(f"ID{int(r['player_id'])}", {
                    "person": {"id": int(r["player_id"]), "fullName": r["player_name"]}, "stats": {}})
                outs = int(r["outs"])
                entry["stats"]["pitching"] = {"inningsPitched": f"{outs // 3}.{outs % 3}",
                                              **{api: int(r[c]) for c, api in _PIT_KEYS.items()}}
            teams[side] = {"team": {"id": tid}, "pitchers": [int(p) for p in pitchers["player_id"]],
                           "players": players}
        ump = g.get("hp_umpire_id")
        officials = [] if ump is None or pd.isna(ump) else [
            {"official": {"id": int(ump), "fullName": g.get("hp_umpire_name")}, "officialType": "Home Plate"}]
        return {"teams": teams, "officials": officials}

    def weather(self, game_pk, final=True):
        self.calls.append(f"weather {game_pk}")
        src = self.ds.games if final else self.upcoming
        g = src[src["game_pk"] == game_pk]
        if g.empty:
            return {"condition": None, "temp_f": None, "wind_mph": None, "wind_dir": None, "day_night": None,
                    "hp_umpire_id": None, "hp_umpire_name": None}
        g = g.iloc[0]
        ump = g.get("hp_umpire_id")
        return {"condition": g.get("condition"), "temp_f": g.get("temp_f"), "wind_mph": g.get("wind_mph"),
                "wind_dir": g.get("wind_dir"), "day_night": g.get("day_night"),
                "hp_umpire_id": None if ump is None or pd.isna(ump) else int(ump),
                "hp_umpire_name": g.get("hp_umpire_name")}

    def statcast_range(self, start, end, **kw):
        """Stand-in for ``fetch_statcast_range``: serve the simulator's aggregates for a date window."""
        self.calls.append(f"statcast {start} {end}")
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        b = self.ds.statcast_batting
        p = self.ds.statcast_pitching
        return (b[(b["date"] >= s) & (b["date"] <= e)].reset_index(drop=True),
                p[(p["date"] >= s) & (p["date"] <= e)].reset_index(drop=True))

    def players(self, season):
        self.calls.append(f"players {season}")
        out = self.ds.players.copy()
        out["birth_date"] = out["birth_date"].dt.strftime("%Y-%m-%d")
        return out.to_dict("records")
