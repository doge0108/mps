from datetime import date

from mps.data.mlb_api import (MLBStatsClient, innings_to_outs, parse_boxscore, parse_boxscore_lineups,
                              parse_players, parse_schedule, parse_schedule_lineups, parse_weather, parse_wind)


def test_innings_to_outs():
    assert innings_to_outs("6.1") == 19
    assert innings_to_outs("7.0") == 21
    assert innings_to_outs("0.2") == 2
    assert innings_to_outs(None) == 0
    assert innings_to_outs(5) == 15


def _schedule_payload(with_lineups=False):
    game = {
        "gamePk": 777, "officialDate": "2025-07-04", "season": "2025", "gameType": "R", "dayNight": "night",
        "status": {"abstractGameState": "Preview"},
        "venue": {"id": 3313},
        "teams": {
            "home": {"team": {"id": 147}, "probablePitcher": {"id": 543037}},
            "away": {"team": {"id": 111}},
        },
    }
    if with_lineups:
        game["lineups"] = {
            "homePlayers": [{"id": 1, "fullName": "A"}, {"id": 2, "fullName": "B"}],
            "awayPlayers": [{"id": 3, "fullName": "C"}],
        }
    return {"dates": [{"date": "2025-07-04", "games": [game]}]}


def test_parse_schedule_extracts_probable_pitchers():
    rows = parse_schedule(_schedule_payload())
    assert rows == [{
        "game_pk": 777, "date": "2025-07-04", "season": 2025, "game_type": "R",
        "home_team_id": 147, "away_team_id": 111, "home_score": None, "away_score": None,
        "home_sp_id": 543037, "away_sp_id": None, "venue_id": 3313, "status": "Preview", "day_night": "night",
    }]


def test_parse_schedule_lineups():
    assert parse_schedule_lineups(_schedule_payload()) == []
    rows = parse_schedule_lineups(_schedule_payload(with_lineups=True))
    assert [(r["team_id"], r["player_id"], r["batting_order"]) for r in rows] == [(147, 1, 1), (147, 2, 2), (111, 3, 1)]
    assert all(r["game_pk"] == 777 and r["date"] == "2025-07-04" for r in rows)


def test_parse_wind_and_weather():
    assert parse_wind("10 mph, Out To CF") == (10.0, "Out To CF")
    assert parse_wind("0 mph, None") == (0.0, "None")
    assert parse_wind(None) == (None, None)
    assert parse_wind("Calm") == (None, "Calm")
    payload = {"gameData": {"weather": {"condition": "Partly Cloudy", "temp": "78", "wind": "12 mph, L To R"},
                            "datetime": {"dayNight": "day"}}}
    assert parse_weather(payload) == {"condition": "Partly Cloudy", "temp_f": 78.0, "wind_mph": 12.0,
                                      "wind_dir": "L To R", "day_night": "day"}
    assert parse_weather({}) == {"condition": None, "temp_f": None, "wind_mph": None, "wind_dir": None,
                                 "day_night": None}


def test_parse_players_handedness():
    payload = {"people": [{"id": 5, "fullName": "Lefty Lou", "batSide": {"code": "L"}, "pitchHand": {"code": "R"},
                           "primaryPosition": {"abbreviation": "1B"}, "currentTeam": {"id": 111},
                           "birthDate": "1996-04-02"}]}
    assert parse_players(payload) == [{"player_id": 5, "player_name": "Lefty Lou", "bats": "L", "throws": "R",
                                       "position": "1B", "team_id": 111, "birth_date": "1996-04-02"}]


def test_parse_umpire_and_game_meta():
    from mps.data.mlb_api import parse_umpire
    payload = {"liveData": {"boxscore": {"officials": [
        {"official": {"id": 427044, "fullName": "Joe West"}, "officialType": "First Base"},
        {"official": {"id": 484183, "fullName": "Pat Hoberg"}, "officialType": "Home Plate"},
    ]}}}
    assert parse_umpire(payload) == {"hp_umpire_id": 484183, "hp_umpire_name": "Pat Hoberg"}
    assert parse_umpire({}) == {"hp_umpire_id": None, "hp_umpire_name": None}
    # the boxscore endpoint lists officials at the top level
    box = {"officials": payload["liveData"]["boxscore"]["officials"], "teams": {}}
    assert parse_umpire(box)["hp_umpire_id"] == 484183


def _player(pid, name, batting=None, pitching=None, order=None):
    entry = {"person": {"id": pid, "fullName": name}, "stats": {}}
    if batting is not None:
        entry["stats"]["batting"] = batting
    if pitching is not None:
        entry["stats"]["pitching"] = pitching
    if order:
        entry["battingOrder"] = order
    return entry


BOX = {"teams": {
    "home": {
        "team": {"id": 147},
        "pitchers": [10, 11],
        "players": {
            "ID1": _player(1, "Aaron Judge", batting={
                "plateAppearances": 5, "atBats": 4, "runs": 1, "hits": 2, "doubles": 1, "triples": 0,
                "homeRuns": 1, "rbi": 3, "baseOnBalls": 1, "strikeOuts": 1, "stolenBases": 0,
                "hitByPitch": 0, "totalBases": 6}, order="200"),
            "ID5": _player(5, "Pinch Hitter", batting={"plateAppearances": 1, "atBats": 1, "hits": 0}, order="201"),
            "ID10": _player(10, "Gerrit Cole", batting={}, pitching={
                "inningsPitched": "6.2", "hits": 5, "runs": 2, "earnedRuns": 2, "baseOnBalls": 1,
                "strikeOuts": 8, "homeRuns": 1, "battersFaced": 26, "numberOfPitches": 98}),
            "ID11": _player(11, "Closer Guy", pitching={
                "inningsPitched": "2.1", "hits": 0, "runs": 0, "earnedRuns": 0, "baseOnBalls": 0,
                "strikeOuts": 3, "homeRuns": 0, "battersFaced": 7, "numberOfPitches": 30}),
            "ID99": _player(99, "Bench Bat", batting={}),  # did not play
        },
    },
    "away": {
        "team": {"id": 111},
        "pitchers": [20],
        "players": {
            "ID2": _player(2, "Rafael Devers", batting={
                "atBats": 4, "runs": 0, "hits": 1, "doubles": 0, "triples": 0, "homeRuns": 0, "rbi": 0,
                "baseOnBalls": 0, "strikeOuts": 2, "stolenBases": 1, "hitByPitch": 0, "totalBases": 1},
                order="300"),
            "ID20": _player(20, "Away Starter", pitching={
                "inningsPitched": "5.0", "hits": 7, "runs": 4, "earnedRuns": 4, "baseOnBalls": 3,
                "strikeOuts": 4, "homeRuns": 1, "battersFaced": 24, "numberOfPitches": 95}),
        },
    },
}}


def test_parse_boxscore_lines_and_starters():
    batting, pitching, starters = parse_boxscore(BOX, 777, "2025-07-04")
    assert starters == {"home_sp_id": 10, "away_sp_id": 20}
    assert {r["player_id"] for r in batting} == {1, 2, 5}
    judge = next(r for r in batting if r["player_id"] == 1)
    assert judge["batting_order"] == 2 and judge["bat_starter"] == 1 and judge["is_home"] == 1
    assert judge["opp_sp_id"] == 20 and judge["h"] == 2 and judge["hr"] == 1 and judge["tb"] == 6 and judge["pa"] == 5
    ph = next(r for r in batting if r["player_id"] == 5)
    assert ph["batting_order"] == 2 and ph["bat_starter"] == 0
    devers = next(r for r in batting if r["player_id"] == 2)
    assert devers["pa"] == 4 and devers["opp_sp_id"] == 10 and devers["opp_team_id"] == 147
    cole = next(r for r in pitching if r["player_id"] == 10)
    assert cole["outs"] == 20 and cole["is_starter"] == 1 and cole["so"] == 8
    closer = next(r for r in pitching if r["player_id"] == 11)
    assert closer["is_starter"] == 0 and closer["outs"] == 7


def test_parse_boxscore_lineups_only_keeps_starters():
    rows = parse_boxscore_lineups(BOX, 777, "2025-07-04")
    assert [(r["team_id"], r["player_id"], r["batting_order"]) for r in rows] == [(111, 2, 3), (147, 1, 2)]


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload, self.calls, self.headers = payload, [], {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return _FakeResponse(self.payload)


def test_client_caches_past_schedules_but_not_current_ones(tmp_path):
    payload = {"dates": [{"date": "2025-05-01", "games": [{
        "gamePk": 1, "officialDate": "2025-05-01", "season": "2025", "gameType": "R",
        "status": {"abstractGameState": "Final"},
        "teams": {"home": {"team": {"id": 147}, "score": 4}, "away": {"team": {"id": 111}, "score": 2}},
    }]}]}
    session = _FakeSession(payload)
    client = MLBStatsClient(cache_dir=tmp_path, pause=0, session=session, today=date(2025, 6, 1))
    first = parse_schedule(client.schedule("2025-05-01", "2025-05-01"))
    second = parse_schedule(client.schedule("2025-05-01", "2025-05-01"))
    assert first == second and first[0]["home_score"] == 4 and first[0]["status"] == "Final"
    assert len(session.calls) == 1  # second call served from cache
    assert session.calls[0][0].endswith("/schedule")
    assert session.calls[0][1]["hydrate"] == "probablePitcher"
    assert list(tmp_path.glob("*.json"))
    # a window that reaches today can still change -> never cached, lineups hydrated on request
    client.schedule("2025-05-30", "2025-06-01", with_lineups=True)
    client.schedule("2025-05-30", "2025-06-01", with_lineups=True)
    assert len(session.calls) == 3
    assert session.calls[-1][1]["hydrate"] == "probablePitcher,lineups"
    # weather goes through the v1.1 live feed with a field filter
    session.payload = {"gameData": {"weather": {"temp": "70", "wind": "5 mph, In From CF", "condition": "Clear"}}}
    wx = client.weather(1)
    assert wx["temp_f"] == 70 and wx["wind_dir"] == "In From CF"
    assert "/v1.1/game/1/feed/live" in session.calls[-1][0] and "fields" in session.calls[-1][1]
