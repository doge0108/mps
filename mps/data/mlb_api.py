"""Thin client for the public MLB Stats API (https://statsapi.mlb.com).

No API key is required.  Raw JSON responses for completed games are cached on
disk so repeated ingestion runs do not hammer the API; anything that can still
change (today's schedule, announced lineups, forecasts) is never cached.  All
parsing functions are pure so they can be unit-tested against fixture payloads.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import date as _date
from pathlib import Path
from typing import Any, Iterable

import requests

from ..config import BATTING_STATS, PITCHING_STATS

log = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1"
BASE_URL_V11 = "https://statsapi.mlb.com/api/v1.1"

_BATTING_MAP = {
    "plateAppearances": "pa", "atBats": "ab", "runs": "r", "hits": "h", "doubles": "d",
    "triples": "t", "homeRuns": "hr", "rbi": "rbi", "baseOnBalls": "bb", "strikeOuts": "so",
    "stolenBases": "sb", "hitByPitch": "hbp", "totalBases": "tb",
}
_PITCHING_MAP = {
    "hits": "h", "runs": "r", "earnedRuns": "er", "baseOnBalls": "bb", "strikeOuts": "so",
    "homeRuns": "hr", "battersFaced": "bf", "numberOfPitches": "pitches",
}
WEATHER_FIELDS = "gameData,weather,condition,temp,wind,datetime,dayNight"


def innings_to_outs(ip: str | float | None) -> int:
    """Convert the API's 'innings pitched' notation ('6.1' = 6 innings + 1 out) to outs."""
    if ip is None or ip == "":
        return 0
    text = str(ip)
    whole, _, frac = text.partition(".")
    outs = int(whole or 0) * 3
    if frac:
        outs += int(frac[0])
    return outs


class MLBStatsClient:
    """Small requests-based client with disk caching and polite retries."""

    def __init__(self, cache_dir: Path | None = None, timeout: float = 20.0,
                 max_retries: int = 3, pause: float = 0.15, session: requests.Session | None = None,
                 today: _date | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.pause = pause
        self.today = today or _date.today()
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "mps/0.1 (+https://github.com/doge0108/mps)")

    # ------------------------------------------------------------------ HTTP
    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
        return self.cache_dir / f"{safe}.json"

    def get(self, path: str, params: dict[str, Any] | None = None, cache_key: str | None = None,
            base: str = BASE_URL) -> dict:
        params = params or {}
        cpath = self._cache_path(cache_key) if cache_key else None
        if cpath and cpath.exists():
            with cpath.open() as fh:
                return json.load(fh)
        url = f"{base}/{path.lstrip('/')}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} from {url}")
                resp.raise_for_status()
                payload = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:  # pragma: no cover - network
                last_err = exc
                wait = 2 ** attempt
                log.warning("request failed (%s); retrying in %ss", exc, wait)
                time.sleep(wait)
        else:  # pragma: no cover - network
            raise RuntimeError(f"MLB API request failed after {self.max_retries} attempts: {last_err}")
        if cpath:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            with cpath.open("w") as fh:
                json.dump(payload, fh)
        if self.pause:
            time.sleep(self.pause)
        return payload

    # ------------------------------------------------------------- endpoints
    def schedule(self, start_date: str, end_date: str, game_types: Iterable[str] = ("R",),
                 with_lineups: bool = False) -> dict:
        """Raw schedule payload between two ISO dates (inclusive).

        Only cached when the whole window is in the past (results can no longer change).
        """
        hydrate = "probablePitcher,lineups" if with_lineups else "probablePitcher"
        types = ",".join(game_types)
        cache_key = None
        if _date.fromisoformat(end_date) < self.today:
            cache_key = f"schedule_{start_date}_{end_date}_{types.replace(',', '_')}"
        return self.get(
            "schedule",
            {"sportId": 1, "startDate": start_date, "endDate": end_date, "gameType": types, "hydrate": hydrate},
            cache_key=cache_key,
        )

    def boxscore(self, game_pk: int, final: bool = True) -> dict:
        return self.get(f"game/{game_pk}/boxscore", cache_key=f"boxscore_{game_pk}" if final else None)

    def weather(self, game_pk: int, final: bool = True) -> dict:
        """Weather / time-of-day block from the live feed (small thanks to the ``fields`` filter)."""
        payload = self.get(f"game/{game_pk}/feed/live", {"fields": WEATHER_FIELDS},
                           cache_key=f"weather_{game_pk}" if final else None, base=BASE_URL_V11)
        return parse_weather(payload)

    def teams(self, season: int | None = None) -> list[dict]:
        params: dict[str, Any] = {"sportId": 1}
        if season:
            params["season"] = season
        payload = self.get("teams", params, cache_key=f"teams_{season or 'current'}")
        return [
            {"team_id": t["id"], "abbr": t.get("abbreviation"), "name": t.get("name")}
            for t in payload.get("teams", [])
        ]

    def players(self, season: int) -> list[dict]:
        """All players on MLB rosters for a season with handedness (never cached for the current season)."""
        cache_key = f"players_{season}" if season < self.today.year else None
        payload = self.get("sports/1/players", {"season": season}, cache_key=cache_key)
        return parse_players(payload)


# ---------------------------------------------------------------- parsers
def parse_players(payload: dict) -> list[dict]:
    return [
        {
            "player_id": int(p["id"]),
            "player_name": p.get("fullName"),
            "bats": (p.get("batSide") or {}).get("code"),
            "throws": (p.get("pitchHand") or {}).get("code"),
            "position": (p.get("primaryPosition") or {}).get("abbreviation"),
            "team_id": (p.get("currentTeam") or {}).get("id"),
        }
        for p in payload.get("people", [])
    ]


def parse_schedule(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            home, away = g["teams"]["home"], g["teams"]["away"]
            status = (g.get("status") or {}).get("abstractGameState", "")
            rows.append({
                "game_pk": int(g["gamePk"]),
                "date": g.get("officialDate") or day.get("date"),
                "season": int(str(g.get("season") or g.get("officialDate", "")[:4] or day["date"][:4])),
                "game_type": g.get("gameType", "R"),
                "home_team_id": int(home["team"]["id"]),
                "away_team_id": int(away["team"]["id"]),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "home_sp_id": (home.get("probablePitcher") or {}).get("id"),
                "away_sp_id": (away.get("probablePitcher") or {}).get("id"),
                "venue_id": (g.get("venue") or {}).get("id"),
                "status": status,
                "day_night": g.get("dayNight"),
            })
    return rows


def parse_schedule_lineups(payload: dict) -> list[dict]:
    """Announced lineups from ``hydrate=lineups`` (empty until a lineup is posted)."""
    rows: list[dict] = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            lineups = g.get("lineups") or {}
            date = g.get("officialDate") or day.get("date")
            for side in ("home", "away"):
                players = lineups.get(f"{side}Players") or []
                team_id = int(g["teams"][side]["team"]["id"])
                for order, p in enumerate(players, start=1):
                    rows.append({"game_pk": int(g["gamePk"]), "date": date, "team_id": team_id,
                                 "player_id": int(p["id"]), "batting_order": order})
    return rows


def parse_boxscore_lineups(box: dict, game_pk: int, date: str) -> list[dict]:
    """Batting order from a (possibly pre-game) boxscore: entries whose battingOrder ends in 00."""
    rows: list[dict] = []
    for side in ("home", "away"):
        team = box.get("teams", {}).get(side, {})
        team_id = int(team["team"]["id"])
        for entry in (team.get("players") or {}).values():
            order = entry.get("battingOrder")
            if order and int(order) % 100 == 0:
                rows.append({"game_pk": game_pk, "date": date, "team_id": team_id,
                             "player_id": int(entry["person"]["id"]), "batting_order": int(order) // 100})
    return sorted(rows, key=lambda r: (r["team_id"], r["batting_order"]))


_WIND_RE = re.compile(r"(\d+)\s*mph,?\s*(.*)", re.IGNORECASE)


def parse_wind(text: str | None) -> tuple[float | None, str | None]:
    """'10 mph, Out To CF' -> (10.0, 'Out To CF')."""
    if not text:
        return None, None
    m = _WIND_RE.search(text)
    if not m:
        return None, text.strip() or None
    return float(m.group(1)), (m.group(2).strip() or None)


def parse_weather(payload: dict) -> dict:
    gd = payload.get("gameData") or {}
    w = gd.get("weather") or {}
    temp = w.get("temp")
    try:
        temp = float(temp) if temp not in (None, "") else None
    except ValueError:
        temp = None
    speed, direction = parse_wind(w.get("wind"))
    return {
        "condition": w.get("condition") or None,
        "temp_f": temp,
        "wind_mph": speed,
        "wind_dir": direction,
        "day_night": (gd.get("datetime") or {}).get("dayNight"),
    }


def parse_boxscore(box: dict, game_pk: int, date: str) -> tuple[list[dict], list[dict], dict]:
    """Extract batting lines, pitching lines and starting pitchers from a boxscore payload.

    Returns (batting_rows, pitching_rows, {"home_sp_id":..., "away_sp_id":...}).
    """
    batting: list[dict] = []
    pitching: list[dict] = []
    starters: dict[str, int | None] = {}
    teams = box.get("teams", {})
    for side in ("home", "away"):
        team = teams.get(side, {})
        opp = teams.get("away" if side == "home" else "home", {})
        team_id = int(team["team"]["id"])
        opp_id = int(opp["team"]["id"])
        pitcher_order = [int(p) for p in team.get("pitchers", [])]
        opp_pitchers = [int(p) for p in opp.get("pitchers", [])]
        starter_id = pitcher_order[0] if pitcher_order else None
        opp_starter = opp_pitchers[0] if opp_pitchers else None
        starters[f"{side}_sp_id"] = starter_id
        for entry in (team.get("players") or {}).values():
            person = entry.get("person", {})
            pid = int(person["id"])
            name = person.get("fullName", str(pid))
            stats = entry.get("stats") or {}
            bat = stats.get("batting") or {}
            if bat and (bat.get("plateAppearances", 0) or bat.get("atBats", 0)):
                order = entry.get("battingOrder")
                row = {
                    "game_pk": game_pk, "date": date, "player_id": pid, "player_name": name,
                    "team_id": team_id, "opp_team_id": opp_id, "is_home": int(side == "home"),
                    "batting_order": int(order) // 100 if order else 0,
                    "bat_starter": int(bool(order) and int(order) % 100 == 0),
                    "opp_sp_id": opp_starter,
                }
                for api_key, col in _BATTING_MAP.items():
                    row[col] = int(bat.get(api_key, 0) or 0)
                if not row["pa"]:
                    row["pa"] = row["ab"] + row["bb"] + row["hbp"] + int(bat.get("sacFlies", 0) or 0) \
                        + int(bat.get("sacBunts", 0) or 0)
                batting.append(row)
            pit = stats.get("pitching") or {}
            if pit and (pit.get("battersFaced", 0) or pit.get("inningsPitched")):
                row = {
                    "game_pk": game_pk, "date": date, "player_id": pid, "player_name": name,
                    "team_id": team_id, "opp_team_id": opp_id, "is_home": int(side == "home"),
                    "is_starter": int(pid == starter_id),
                    "outs": innings_to_outs(pit.get("inningsPitched")),
                }
                for api_key, col in _PITCHING_MAP.items():
                    row[col] = int(pit.get(api_key, 0) or 0)
                pitching.append(row)
    for row in batting:
        for col in BATTING_STATS:
            row.setdefault(col, 0)
    for row in pitching:
        for col in PITCHING_STATS:
            row.setdefault(col, 0)
    return batting, pitching, starters
