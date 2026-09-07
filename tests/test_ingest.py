from datetime import date

import pandas as pd
import pytest

from mps.data import ingest
from mps.data.ingest import fetch_seasons, update
from mps.data.store import Dataset
from tests.fake_client import FakeClient


@pytest.fixture
def statcast_via_client(monkeypatch):
    """Route Statcast downloads through the FakeClient of the test instead of Baseball Savant."""
    holder = {}

    def fake_range(start, end, **kw):
        return holder["client"].statcast_range(start, end)

    monkeypatch.setattr(ingest, "fetch_statcast_range", fake_range)
    return holder


def _split(ds: Dataset, cutoff: pd.Timestamp) -> Dataset:
    """Everything the store 'already has' before the cutoff date."""
    games = ds.games[ds.games["date"] < cutoff]
    keep = games["game_pk"]
    return Dataset(games=games, batting_lines=ds.batting_lines[ds.batting_lines["game_pk"].isin(keep)],
                   pitching_lines=ds.pitching_lines[ds.pitching_lines["game_pk"].isin(keep)], players=ds.players,
                   statcast_batting=ds.statcast_batting[ds.statcast_batting["game_pk"].isin(keep)],
                   statcast_pitching=ds.statcast_pitching[ds.statcast_pitching["game_pk"].isin(keep)])


def test_fetch_in_progress_season_stops_at_today(small_dataset, tmp_path, statcast_via_client):
    ds = small_dataset
    today = date(2024, 5, 15)
    client = FakeClient(ds, today=today)
    statcast_via_client["client"] = client
    got = fetch_seasons([2024], tmp_path / "data", client=client, weather=True)
    expected = ds.games[(ds.games["season"] == 2024) & (ds.games["date"] <= pd.Timestamp(today))]
    assert len(got.games) == len(expected) > 0
    assert set(got.games["game_pk"]) == set(expected["game_pk"])
    assert got.games["temp_f"].notna().all() and got.games["hp_umpire_id"].notna().all()
    assert got.batting_lines["bat_starter"].eq(1).all()
    assert len(got.players) == len(ds.players) and got.players["bats"].isin(["L", "R", "S"]).all()
    assert got.players["birth_date"].notna().all()
    assert set(got.statcast_batting["game_pk"]) == set(expected["game_pk"])
    assert any(c.startswith("statcast ") for c in client.calls)
    assert any(c.startswith("schedule 2024-03-15 2024-05-15") for c in client.calls)
    # box scores agree with the simulator's own tables
    sample = got.batting_lines.sort_values(["game_pk", "player_id"]).head(50).reset_index(drop=True)
    orig = ds.batting_lines[ds.batting_lines["game_pk"].isin(sample["game_pk"])]
    orig = orig.sort_values(["game_pk", "player_id"]).head(50).reset_index(drop=True)
    assert (sample[["h", "hr", "so", "tb"]].to_numpy() == orig[["h", "hr", "so", "tb"]].to_numpy()).all()


def test_update_adds_new_finals_and_upcoming_games_with_lineups(small_dataset, tmp_path, statcast_via_client):
    ds = small_dataset
    data_dir = tmp_path / "data"
    all_dates = sorted(ds.games["date"].unique())
    cutoff = pd.Timestamp(all_dates[-6])
    _split(ds, cutoff).save(data_dir)
    stored = Dataset.load(data_dir)
    today = pd.Timestamp(all_dates[-1])  # the last simulated day counts as "today"
    # tomorrow: two upcoming games with announced lineups and a forecast
    g1, g2 = ds.games.iloc[0], ds.games.iloc[1]
    tomorrow = today + pd.Timedelta(days=1)
    upcoming = pd.DataFrame([
        {"game_pk": 888001, "date": tomorrow, "season": 2024, "home_team_id": g1["home_team_id"],
         "away_team_id": g1["away_team_id"], "home_sp_id": g1["home_sp_id"], "away_sp_id": None,
         "temp_f": 91.0, "wind_mph": 3.0, "wind_dir": "Out To LF", "condition": "Sunny", "day_night": "day"},
        {"game_pk": 888002, "date": tomorrow, "season": 2024, "home_team_id": g2["home_team_id"],
         "away_team_id": g2["away_team_id"], "home_sp_id": None, "away_sp_id": None,
         "temp_f": None, "wind_mph": None, "wind_dir": None, "condition": None, "day_night": "night"},
    ])
    lu_rows = []
    for tid in (g1["home_team_id"], g1["away_team_id"]):
        starters = ds.batting_lines[(ds.batting_lines["team_id"] == tid)].tail(9)
        lu_rows += [{"game_pk": 888001, "team_id": tid, "player_id": pid, "batting_order": i}
                    for i, pid in enumerate(starters["player_id"], start=1)]
    lineups = pd.DataFrame(lu_rows)
    client = FakeClient(ds, today=today.date(), upcoming=upcoming, lineups=lineups, announce_in_schedule=False)
    statcast_via_client["client"] = client

    out = update(data_dir, days_ahead=3, client=client)
    # statcast rows for the new games were fetched and merged with the ones already stored
    assert len(out.statcast_batting) == len(ds.statcast_batting)
    assert out.statcast_batting["date"].max() == today
    fetched_sc = [c for c in client.calls if c.startswith("statcast ")]
    assert len(fetched_sc) == 1 and fetched_sc[0].split()[1] >= (cutoff - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    # every simulated game is now stored, plus the two upcoming ones
    assert len(out.played_games()) == len(ds.games)
    assert out.played_games()["date"].max() == today
    assert set(out.upcoming_games()["game_pk"]) == {888001, 888002}
    up1 = out.games[out.games["game_pk"] == 888001].iloc[0]
    assert up1["temp_f"] == 91.0 and up1["wind_dir"] == "Out To LF" and up1["status"] == "Preview"
    assert int(up1["home_sp_id"]) == int(g1["home_sp_id"])
    # lineups came from the pre-game boxscore fallback since the schedule did not carry them
    assert len(out.lineups) == 18 and set(out.lineups["game_pk"]) == {888001}
    assert any(c == "boxscore 888001" for c in client.calls)
    # only games since the last stored day were re-fetched, not the whole season
    fetched = [c for c in client.calls if c.startswith("boxscore") and not c.endswith(("888001", "888002"))]
    assert 0 < len(fetched) < len(ds.games)
    assert len(stored.played_games()) + len(fetched) >= len(ds.games)

    # a second update is idempotent and postponed games drop out
    client2 = FakeClient(ds, today=today.date(), upcoming=upcoming.iloc[:1], lineups=lineups)
    statcast_via_client["client"] = client2
    again = update(data_dir, days_ahead=3, client=client2)
    assert len(again.played_games()) == len(ds.games)
    assert set(again.upcoming_games()["game_pk"]) == {888001}
    assert len(again.lineups) == 18
