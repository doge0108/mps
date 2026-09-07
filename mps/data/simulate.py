"""Plate-appearance level baseball simulator.

Generates realistic box-score data (games, batting lines, pitching lines)
from latent player skills.  It is used to develop and test the pipeline when
the MLB Stats API is unreachable, and doubles as a source of extra training
data.  Outputs use the exact schema produced by ``mps.data.ingest``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BATTING_STATS, PITCHING_STATS, TEAMS
from .store import Dataset, normalise

FIRST_NAMES = ["Aaron", "Bo", "Carlos", "Dansby", "Elly", "Freddie", "Gunnar", "Hunter", "Ian", "Jose",
               "Kyle", "Luis", "Mookie", "Nolan", "Ozzie", "Pete", "Rafael", "Shohei", "Trea", "Vlad",
               "Will", "Xander", "Yordan", "Zack", "Bryce", "Corbin", "Julio", "Marcus", "Ronald", "Tyler"]
LAST_NAMES = ["Judge", "Bichette", "Correa", "Swanson", "De La Cruz", "Freeman", "Henderson", "Greene",
              "Happ", "Ramirez", "Tucker", "Robert", "Betts", "Arenado", "Albies", "Alonso", "Devers",
              "Ohtani", "Turner", "Guerrero", "Smith", "Bogaerts", "Alvarez", "Wheeler", "Harper",
              "Carroll", "Rodriguez", "Semien", "Acuna", "Glasnow", "Cole", "Skubal", "Burnes", "Sale"]


WIND_DIRS = ["Out To CF", "Out To LF", "Out To RF", "In From CF", "In From LF", "In From RF",
             "L To R", "R To L", "Calm", "Varies"]
DOME_TEAMS = {139, 141, 117, 109, 146, 158, 140}  # roofed parks: neutral weather


@dataclass
class Batter:
    player_id: int
    name: str
    team_id: int
    bats: str
    # per-plate-appearance event probabilities
    p_k: float
    p_bb: float
    p_hr: float
    p_2b: float
    p_3b: float
    p_1b: float
    speed: float          # steal attempt propensity
    regular: bool


@dataclass
class Pitcher:
    player_id: int
    name: str
    team_id: int
    throws: str
    k_mult: float
    bb_mult: float
    hr_mult: float
    hit_mult: float
    stamina: float        # expected outs per start (starters) / per outing (relievers)
    is_starter: bool


@dataclass
class Team:
    team_id: int
    batters: list[Batter]
    rotation: list[Pitcher]
    bullpen: list[Pitcher]
    park_hit: float       # park factor on balls in play (>1 = hitter friendly)
    park_hr: float
    rotation_idx: int = 0
    last_game_date: pd.Timestamp | None = None


@dataclass
class _BatLine:
    stats: dict = field(default_factory=lambda: {k: 0 for k in BATTING_STATS})


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Simulator:
    """Season simulator. ``seed`` makes the generated data reproducible."""

    def __init__(self, seed: int = 7, team_ids: list[int] | None = None, home_adv: float = 1.03):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.team_ids = team_ids or sorted(TEAMS)
        self.home_adv = home_adv
        self._next_pid = 900000
        self.teams: dict[int, Team] = {tid: self._make_team(tid) for tid in self.team_ids}
        self.names: dict[int, str] = {}
        for t in self.teams.values():
            for p in t.batters + t.rotation + t.bullpen:
                self.names[p.player_id] = p.name

    # ------------------------------------------------------------ rosters
    def _name(self) -> str:
        return f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"

    def _pid(self) -> int:
        self._next_pid += 1
        return self._next_pid

    def _make_batter(self, team_id: int, regular: bool, team_q: float = 0.0) -> Batter:
        g = self.rng.gauss
        quality = (-0.15 if not regular else 0.0) + 0.12 * team_q
        u = self.rng.random()
        bats = "L" if u < 0.35 else "S" if u < 0.45 else "R"
        return Batter(
            player_id=self._pid(), name=self._name(), team_id=team_id, bats=bats,
            p_k=_clip(g(0.225 - quality * 0.2, 0.05), 0.08, 0.40),
            p_bb=_clip(g(0.085 + quality * 0.1, 0.025), 0.03, 0.18),
            p_hr=_clip(g(0.032 + quality * 0.06, 0.014), 0.004, 0.09),
            p_2b=_clip(g(0.045 + quality * 0.03, 0.01), 0.015, 0.08),
            p_3b=_clip(abs(g(0.004, 0.003)), 0.0005, 0.015),
            p_1b=_clip(g(0.142 + quality * 0.1, 0.022), 0.08, 0.24),
            speed=_clip(abs(g(0.04, 0.05)), 0.0, 0.25),
            regular=regular,
        )

    def _make_pitcher(self, team_id: int, is_starter: bool, team_q: float = 0.0) -> Pitcher:
        g = self.rng.gauss
        return Pitcher(
            player_id=self._pid(), name=self._name(), team_id=team_id,
            throws="L" if self.rng.random() < 0.28 else "R",
            k_mult=float(np.exp(g(0.08 * team_q, 0.18))),
            bb_mult=float(np.exp(g(-0.08 * team_q, 0.22))),
            hr_mult=float(np.exp(g(-0.10 * team_q, 0.25))),
            hit_mult=float(np.exp(g(-0.04 * team_q, 0.08))),
            stamina=_clip(g(17.0, 2.5), 10, 24) if is_starter else _clip(g(3.0, 1.0), 1, 6),
            is_starter=is_starter,
        )

    def _make_team(self, team_id: int) -> Team:
        team_q = self.rng.gauss(0.0, 1.0)  # organisational quality: separates contenders from rebuilders
        batters = [self._make_batter(team_id, True, team_q) for _ in range(9)]
        batters += [self._make_batter(team_id, False, team_q) for _ in range(4)]
        # sort regulars into a lineup: best on-base/power up top
        batters[:9] = sorted(batters[:9], key=lambda b: -(b.p_1b + b.p_2b + b.p_bb + 2 * b.p_hr))
        rotation = [self._make_pitcher(team_id, True, team_q) for _ in range(5)]
        bullpen = [self._make_pitcher(team_id, False, team_q) for _ in range(7)]
        return Team(
            team_id=team_id, batters=batters, rotation=rotation, bullpen=bullpen,
            park_hit=_clip(self.rng.gauss(1.0, 0.04), 0.9, 1.12),
            park_hr=_clip(self.rng.gauss(1.0, 0.12), 0.7, 1.4),
        )

    def _drift_skills(self) -> None:
        """Small year-over-year skill changes so seasons are not identical."""
        for t in self.teams.values():
            for b in t.batters:
                b.p_k = _clip(b.p_k * np.exp(self.rng.gauss(0, 0.06)), 0.08, 0.40)
                b.p_hr = _clip(b.p_hr * np.exp(self.rng.gauss(0, 0.12)), 0.004, 0.09)
                b.p_1b = _clip(b.p_1b * np.exp(self.rng.gauss(0, 0.05)), 0.08, 0.24)
                b.p_bb = _clip(b.p_bb * np.exp(self.rng.gauss(0, 0.06)), 0.03, 0.18)
            for p in t.rotation + t.bullpen:
                p.k_mult *= float(np.exp(self.rng.gauss(0, 0.06)))
                p.hr_mult *= float(np.exp(self.rng.gauss(0, 0.08)))
                p.bb_mult *= float(np.exp(self.rng.gauss(0, 0.06)))

    # ----------------------------------------------------------- schedule
    def _schedule(self, season: int, games_per_team: int) -> list[tuple[pd.Timestamp, int, int]]:
        """Round-robin rounds, one round per day with occasional off days."""
        ids = list(self.team_ids)
        if len(ids) % 2:
            ids.append(-1)  # bye
        n = len(ids)
        rounds: list[list[tuple[int, int]]] = []
        arr = ids[:]
        for r in range(n - 1):
            pairs = []
            for i in range(n // 2):
                a, b = arr[i], arr[n - 1 - i]
                if a != -1 and b != -1:
                    pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
            rounds.append(pairs)
            arr = [arr[0]] + [arr[-1]] + arr[1:-1]
        games_per_round = 1  # each team plays once per round
        n_rounds = int(np.ceil(games_per_team / games_per_round))
        date = pd.Timestamp(year=season, month=3, day=28)
        out = []
        for k in range(n_rounds):
            pairs = rounds[k % len(rounds)]
            if k and k % len(rounds) == 0:
                self.rng.shuffle(rounds)
            # flip home/away between cycles so home games balance out
            flip = (k // len(rounds)) % 2 == 1
            for home, away in pairs:
                out.append((date, away, home) if flip else (date, home, away))
            date += pd.Timedelta(days=1)
            if self.rng.random() < 0.12:
                date += pd.Timedelta(days=1)
        return out

    # ---------------------------------------------------------- game sim
    @staticmethod
    def platoon_edge(bats: str, throws: str) -> int:
        """+1 when the batter has the platoon advantage, -1 when the pitcher does, 0 for switch hitters."""
        if bats == "S":
            return 0
        return 1 if bats != throws else -1

    def _pa(self, batter: Batter, pitcher: Pitcher, park: Team, home_batting: bool,
            hr_env: float = 1.0) -> str:
        hit_boost = self.home_adv if home_batting else 1.0
        edge = self.platoon_edge(batter.bats, pitcher.throws)
        hit_boost *= 1.0 + 0.06 * edge
        p_k = batter.p_k * pitcher.k_mult * (1.0 - 0.07 * edge)
        p_bb = batter.p_bb * pitcher.bb_mult * (1.0 + 0.05 * edge)
        p_hr = batter.p_hr * pitcher.hr_mult * park.park_hr * hit_boost * hr_env
        p_2b = batter.p_2b * pitcher.hit_mult * park.park_hit * hit_boost
        p_3b = batter.p_3b * pitcher.hit_mult * park.park_hit * hit_boost
        p_1b = batter.p_1b * pitcher.hit_mult * park.park_hit * hit_boost
        p_hbp = 0.011
        total = p_k + p_bb + p_hr + p_2b + p_3b + p_1b + p_hbp
        if total > 0.97:
            scale = 0.97 / total
            p_k, p_bb, p_hr, p_2b, p_3b, p_1b, p_hbp = (
                x * scale for x in (p_k, p_bb, p_hr, p_2b, p_3b, p_1b, p_hbp))
        u = self.rng.random()
        for prob, name in ((p_k, "K"), (p_bb, "BB"), (p_hbp, "HBP"), (p_hr, "HR"),
                           (p_3b, "3B"), (p_2b, "2B"), (p_1b, "1B")):
            if u < prob:
                return name
            u -= prob
        return "OUT"

    def _lineup(self, team: Team) -> list[Batter]:
        lineup = []
        bench = list(team.batters[9:])
        for reg in team.batters[:9]:
            if self.rng.random() < 0.10 and bench:
                sub = self.rng.choice(bench)
                bench.remove(sub)
                lineup.append(sub)
            else:
                lineup.append(reg)
        return lineup

    def _weather(self, date: pd.Timestamp, home: Team) -> tuple[dict, float]:
        """Game-time weather and the resulting home-run environment multiplier."""
        night = self.rng.random() < 0.7
        if home.team_id in DOME_TEAMS and self.rng.random() < 0.8:
            w = {"day_night": "night" if night else "day", "temp_f": 72.0, "wind_mph": 0.0,
                 "wind_dir": "None", "condition": "Roof Closed"}
            return w, 1.0
        month_base = {3: 55, 4: 60, 5: 68, 6: 76, 7: 82, 8: 81, 9: 73, 10: 62}.get(date.month, 70)
        temp = round(self.rng.gauss(month_base - (6 if night else 0), 8), 0)
        wind = round(abs(self.rng.gauss(7, 5)), 0)
        wind_dir = self.rng.choice(WIND_DIRS)
        cond = self.rng.choice(["Clear", "Clear", "Sunny", "Partly Cloudy", "Cloudy", "Overcast", "Drizzle"])
        hr_env = 1.0 + 0.004 * (temp - 70)
        if wind_dir.startswith("Out"):
            hr_env *= 1.0 + 0.012 * wind
        elif wind_dir.startswith("In"):
            hr_env *= 1.0 - 0.010 * wind
        w = {"day_night": "night" if night else "day", "temp_f": float(temp), "wind_mph": float(wind),
             "wind_dir": wind_dir, "condition": cond}
        return w, max(0.6, hr_env)

    def _play_game(self, date: pd.Timestamp, game_pk: int, season: int, home: Team, away: Team):
        weather, hr_env = self._weather(date, home)
        lineups = {"home": self._lineup(home), "away": self._lineup(away)}
        starters = {"home": home.rotation[home.rotation_idx % 5], "away": away.rotation[away.rotation_idx % 5]}
        home.rotation_idx += 1
        away.rotation_idx += 1
        bat_lines: dict[tuple[str, int], dict] = {}
        pit_lines: dict[tuple[str, int], dict] = {}
        for side in ("home", "away"):
            for order, b in enumerate(lineups[side], start=1):
                bat_lines[(side, b.player_id)] = {k: 0 for k in BATTING_STATS} | {"order": order, "player": b}
        current_pitcher = {"home": starters["home"], "away": starters["away"]}
        used_relievers = {"home": [], "away": []}
        stamina_target = {
            side: max(6, int(round(self.rng.gauss(starters[side].stamina, 2.5)))) for side in starters
        }

        def pit_line(side: str, p: Pitcher) -> dict:
            key = (side, p.player_id)
            if key not in pit_lines:
                pit_lines[key] = {k: 0 for k in PITCHING_STATS} | {
                    "player": p, "is_starter": int(p is starters[side])}
            return pit_lines[key]

        def maybe_change_pitcher(fielding: str) -> None:
            p = current_pitcher[fielding]
            line = pit_line(fielding, p)
            limit = stamina_target[fielding] if p.is_starter else max(2, int(round(self.rng.gauss(p.stamina, 1.0))))
            tired = line["outs"] >= limit or (p.is_starter and line["r"] >= 6) or (not p.is_starter and line["r"] >= 3)
            if tired:
                team = home if fielding == "home" else away
                pool = [r for r in team.bullpen if r not in used_relievers[fielding]] or team.bullpen
                nxt = self.rng.choice(pool)
                used_relievers[fielding].append(nxt)
                current_pitcher[fielding] = nxt
                pit_line(fielding, nxt)

        score = {"home": 0, "away": 0}
        next_up = {"home": 0, "away": 0}
        inning = 1
        park = home
        while True:
            for batting in ("away", "home"):
                if inning >= 9 and batting == "home" and score["home"] > score["away"]:
                    break
                fielding = "home" if batting == "away" else "away"
                if inning > 1 or batting == "home":
                    maybe_change_pitcher(fielding)
                outs = 0
                bases: list[Batter | None] = [None, None, None]
                lineup = lineups[batting]
                while outs < 3:
                    batter = lineup[next_up[batting] % 9]
                    next_up[batting] += 1
                    pitcher = current_pitcher[fielding]
                    bl = bat_lines[(batting, batter.player_id)]
                    pl = pit_line(fielding, pitcher)
                    ev = self._pa(batter, pitcher, park, batting == "home", hr_env)
                    bl["pa"] += 1
                    pl["bf"] += 1
                    pl["pitches"] += 4 if ev in ("K", "BB") else 3
                    runs = 0

                    def score_runner(r: Batter | None) -> int:
                        if r is None:
                            return 0
                        bat_lines[(batting, r.player_id)]["r"] += 1
                        return 1

                    if ev == "K":
                        bl["ab"] += 1; bl["so"] += 1; pl["so"] += 1; outs += 1
                    elif ev == "OUT":
                        bl["ab"] += 1; outs += 1
                        # productive out: runner on third scores with <2 outs (sac fly); no AB for sac fly
                        if outs < 3 and bases[2] is not None and self.rng.random() < 0.30:
                            runs += score_runner(bases[2]); bases[2] = None
                            bl["ab"] -= 1
                        elif outs < 3 and bases[0] is not None and self.rng.random() < 0.10:
                            bases[0] = None; outs += 1  # double play
                    elif ev in ("BB", "HBP"):
                        bl["bb" if ev == "BB" else "hbp"] += 1
                        if ev == "BB":
                            pl["bb"] += 1
                        if bases[0] is not None:
                            if bases[1] is not None:
                                if bases[2] is not None:
                                    runs += score_runner(bases[2])
                                bases[2] = bases[1]
                            bases[1] = bases[0]
                        bases[0] = batter
                    else:  # hit
                        bl["ab"] += 1; bl["h"] += 1; pl["h"] += 1
                        if ev == "HR":
                            bl["hr"] += 1; bl["tb"] += 4; pl["hr"] += 1
                            for i in range(3):
                                runs += score_runner(bases[i]); bases[i] = None
                            runs += score_runner(batter)
                        elif ev == "3B":
                            bl["t"] += 1; bl["tb"] += 3
                            for i in range(3):
                                runs += score_runner(bases[i]); bases[i] = None
                            bases[2] = batter
                        elif ev == "2B":
                            bl["d"] += 1; bl["tb"] += 2
                            runs += score_runner(bases[2]); runs += score_runner(bases[1])
                            bases[2] = bases[1] = None
                            if bases[0] is not None:
                                if self.rng.random() < 0.45:
                                    runs += score_runner(bases[0])
                                else:
                                    bases[2] = bases[0]
                                bases[0] = None
                            bases[1] = batter
                        else:  # 1B
                            bl["tb"] += 1
                            runs += score_runner(bases[2]); bases[2] = None
                            if bases[1] is not None:
                                if self.rng.random() < 0.60:
                                    runs += score_runner(bases[1])
                                else:
                                    bases[2] = bases[1]
                                bases[1] = None
                            if bases[0] is not None:
                                if bases[2] is None and self.rng.random() < 0.28:
                                    bases[2] = bases[0]
                                else:
                                    bases[1] = bases[0]
                                bases[0] = None
                            bases[0] = batter
                            # stolen base attempt
                            if bases[1] is None and self.rng.random() < batter.speed:
                                if self.rng.random() < 0.76:
                                    bl["sb"] += 1; bases[1] = batter; bases[0] = None
                                else:
                                    bases[0] = None; outs += 1
                    if runs:
                        bl["rbi"] += runs
                        pl["r"] += runs; pl["er"] += runs
                        score[batting] += runs
                    if outs >= 3:
                        break
                    if inning >= 9 and batting == "home" and score["home"] > score["away"]:
                        break  # walk-off
                pl_end = pit_line(fielding, current_pitcher[fielding])
                pl_end["outs"] += min(outs, 3) if outs <= 3 else 3
                # attribute outs in the inning to the pitcher who finished it (good enough for box scores)
                if inning >= 9 and batting == "home" and score["home"] > score["away"]:
                    break
            if inning >= 9 and score["home"] != score["away"]:
                break
            inning += 1
            if inning > 18:  # safety valve
                score["home"] += 1
                break

        bat_rows, pit_rows = [], []
        for (side, pid), line in bat_lines.items():
            b = line["player"]
            row = {"game_pk": game_pk, "date": date, "player_id": pid, "player_name": b.name,
                   "team_id": b.team_id,
                   "opp_team_id": away.team_id if side == "home" else home.team_id,
                   "is_home": int(side == "home"), "batting_order": line["order"], "bat_starter": 1,
                   "opp_sp_id": starters["away" if side == "home" else "home"].player_id}
            row.update({k: line[k] for k in BATTING_STATS})
            bat_rows.append(row)
        for (side, pid), line in pit_lines.items():
            p = line["player"]
            row = {"game_pk": game_pk, "date": date, "player_id": pid, "player_name": p.name,
                   "team_id": p.team_id,
                   "opp_team_id": away.team_id if side == "home" else home.team_id,
                   "is_home": int(side == "home"), "is_starter": line["is_starter"]}
            row.update({k: line[k] for k in PITCHING_STATS})
            pit_rows.append(row)
        game = {"game_pk": game_pk, "date": date, "season": season, "game_type": "R",
                "home_team_id": home.team_id, "away_team_id": away.team_id,
                "home_score": score["home"], "away_score": score["away"],
                "home_sp_id": starters["home"].player_id, "away_sp_id": starters["away"].player_id,
                "venue_id": home.team_id, "status": "Final", **weather}
        return game, bat_rows, pit_rows

    # ------------------------------------------------------------ driver
    def simulate(self, seasons: list[int], games_per_team: int = 162) -> Dataset:
        games, bats, pits = [], [], []
        game_pk = 700000
        for i, season in enumerate(seasons):
            if i:
                self._drift_skills()
            for t in self.teams.values():
                t.rotation_idx = self.rng.randrange(5)
            for date, home_id, away_id in self._schedule(season, games_per_team):
                game_pk += 1
                g, b, p = self._play_game(date, game_pk, season, self.teams[home_id], self.teams[away_id])
                games.append(g); bats.extend(b); pits.extend(p)
        frames = normalise({
            "games": pd.DataFrame(games),
            "batting_lines": pd.DataFrame(bats),
            "pitching_lines": pd.DataFrame(pits),
            "players": self.players_table(),
        })
        return Dataset(**frames)

    def players_table(self) -> pd.DataFrame:
        rows = []
        for t in self.teams.values():
            for b in t.batters:
                rows.append({"player_id": b.player_id, "player_name": b.name, "bats": b.bats, "throws": "R",
                             "position": "IF", "team_id": t.team_id})
            for p in t.rotation + t.bullpen:
                rows.append({"player_id": p.player_id, "player_name": p.name, "bats": "R", "throws": p.throws,
                             "position": "P", "team_id": t.team_id})
        return pd.DataFrame(rows)


def simulate_dataset(seasons: list[int], games_per_team: int = 162, seed: int = 7,
                     n_teams: int | None = None) -> Dataset:
    team_ids = sorted(TEAMS)[:n_teams] if n_teams else None
    return Simulator(seed=seed, team_ids=team_ids).simulate(seasons, games_per_team)
