from mps.data.mlb_api import innings_to_outs, parse_boxscore, parse_schedule


def test_innings_to_outs():
    assert innings_to_outs("6.1") == 19
    assert innings_to_outs("7.0") == 21
    assert innings_to_outs("0.2") == 2
    assert innings_to_outs(None) == 0
    assert innings_to_outs(5) == 15


def test_parse_schedule_extracts_probable_pitchers():
    payload = {"dates": [{"date": "2025-07-04", "games": [{
        "gamePk": 777, "officialDate": "2025-07-04", "season": "2025", "gameType": "R",
        "status": {"abstractGameState": "Preview"},
        "venue": {"id": 3313},
        "teams": {
            "home": {"team": {"id": 147}, "probablePitcher": {"id": 543037}},
            "away": {"team": {"id": 111}},
        },
    }]}]}
    rows = parse_schedule(payload)
    assert rows == [{
        "game_pk": 777, "date": "2025-07-04", "season": 2025, "game_type": "R",
        "home_team_id": 147, "away_team_id": 111, "home_score": None, "away_score": None,
        "home_sp_id": 543037, "away_sp_id": None, "venue_id": 3313, "status": "Preview",
    }]


def _player(pid, name, batting=None, pitching=None, order=None):
    entry = {"person": {"id": pid, "fullName": name}, "stats": {}}
    if batting is not None:
        entry["stats"]["batting"] = batting
    if pitching is not None:
        entry["stats"]["pitching"] = pitching
    if order:
        entry["battingOrder"] = order
    return entry


def test_parse_boxscore_lines_and_starters():
    box = {"teams": {
        "home": {
            "team": {"id": 147},
            "pitchers": [10, 11],
            "players": {
                "ID1": _player(1, "Aaron Judge", batting={
                    "plateAppearances": 5, "atBats": 4, "runs": 1, "hits": 2, "doubles": 1, "triples": 0,
                    "homeRuns": 1, "rbi": 3, "baseOnBalls": 1, "strikeOuts": 1, "stolenBases": 0,
                    "hitByPitch": 0, "totalBases": 6}, order="200"),
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
    batting, pitching, starters = parse_boxscore(box, 777, "2025-07-04")
    assert starters == {"home_sp_id": 10, "away_sp_id": 20}
    assert {r["player_id"] for r in batting} == {1, 2}
    judge = next(r for r in batting if r["player_id"] == 1)
    assert judge["batting_order"] == 2 and judge["is_home"] == 1 and judge["opp_sp_id"] == 20
    assert judge["h"] == 2 and judge["hr"] == 1 and judge["tb"] == 6 and judge["pa"] == 5
    devers = next(r for r in batting if r["player_id"] == 2)
    assert devers["pa"] == 4 and devers["opp_sp_id"] == 10 and devers["opp_team_id"] == 147
    cole = next(r for r in pitching if r["player_id"] == 10)
    assert cole["outs"] == 20 and cole["is_starter"] == 1 and cole["so"] == 8
    closer = next(r for r in pitching if r["player_id"] == 11)
    assert closer["is_starter"] == 0 and closer["outs"] == 7


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


def test_client_caches_responses_on_disk(tmp_path):
    from mps.data.mlb_api import MLBStatsClient
    payload = {"dates": [{"date": "2025-05-01", "games": [{
        "gamePk": 1, "officialDate": "2025-05-01", "season": "2025", "gameType": "R",
        "status": {"abstractGameState": "Final"},
        "teams": {"home": {"team": {"id": 147}, "score": 4}, "away": {"team": {"id": 111}, "score": 2}},
    }]}]}
    session = _FakeSession(payload)
    client = MLBStatsClient(cache_dir=tmp_path, pause=0, session=session)
    first = client.schedule("2025-05-01", "2025-05-01")
    second = client.schedule("2025-05-01", "2025-05-01")
    assert first == second and first[0]["home_score"] == 4 and first[0]["status"] == "Final"
    assert len(session.calls) == 1  # second call served from cache
    assert session.calls[0][0].endswith("/schedule")
    assert session.calls[0][1]["hydrate"] == "probablePitcher"
    assert list(tmp_path.glob("*.json"))
