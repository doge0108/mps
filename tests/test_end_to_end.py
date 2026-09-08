import json

import pandas as pd
import pytest

from mps.cli import main
from mps.models.registry import ModelBundle, train_all
from mps.predict import Predictor


@pytest.fixture(scope="module")
def trained(small_dataset, tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    bundle = train_all(small_dataset, max_rounds=60, verbose=False)
    bundle.save(models_dir)
    return models_dir


def test_models_round_trip(trained, small_dataset):
    bundle = ModelBundle.load(trained)
    assert set(bundle.batter.boosters) == {"h", "hr", "rbi", "r", "bb", "so", "sb", "tb", "ab"}
    assert set(bundle.game.boosters) == {"home_win", "home_runs", "away_runs"}


def test_predict_player_for_scheduled_game(trained, small_dataset):
    pred = Predictor(ds=small_dataset, models=ModelBundle.load(trained))
    # pick a regular batter and one of their real games in the final season
    last_game = small_dataset.games.sort_values("date").iloc[-1]
    lines = small_dataset.batting_lines[small_dataset.batting_lines["game_pk"] == last_game["game_pk"]]
    pid = int(lines.iloc[0]["player_id"])
    out = pred.predict_player(str(pid), last_game["date"])
    assert out["context_source"] == "schedule"
    assert out["opponent"] is not None
    # a completed game carries the actual box-score line for grading
    actual = out["actual"]
    assert actual["batting"]["h"] == int(lines.iloc[0]["h"]) and actual["batting"]["ab"] == int(lines.iloc[0]["ab"])
    assert actual["final_score"].startswith(pred.ds.games.pipe(lambda g: __import__("mps.config", fromlist=["team_label"]).team_label(last_game["home_team_id"])))
    exp = out["batting"]["expected"]
    assert 0.2 < exp["h"] < 3 and 0 < exp["hr"] < 1 and 2 < exp["ab"] < 5.5
    probs = out["batting"]["probabilities"]
    assert 0 < probs["home_run"] < probs["hit"] < 1
    # fuzzy name lookup resolves the same player
    name = str(lines.iloc[0]["player_name"])
    assert pred.find_player(name.lower()).player_id == pid


def test_predict_pitcher_and_unknown_context(trained, small_dataset):
    pred = Predictor(ds=small_dataset, models=ModelBundle.load(trained))
    sp = small_dataset.pitching_lines[small_dataset.pitching_lines["is_starter"] == 1].iloc[-1]
    future = small_dataset.games["date"].max() + pd.Timedelta(days=3)
    out = pred.predict_player(str(int(sp["player_id"])), future)
    assert out["context_source"] == "unknown"
    assert 5 < out["pitching"]["expected"]["outs"] < 27
    assert out["pitching"]["probable_starter"] is True  # nobody else is listed
    # a scheduled game with a different probable starter is flagged
    g = small_dataset.games.sort_values("date").iloc[-1]
    other = small_dataset.pitching_lines[(small_dataset.pitching_lines["is_starter"] == 1)
                                         & (small_dataset.pitching_lines["team_id"] == g["home_team_id"])
                                         & (small_dataset.pitching_lines["player_id"] != g["home_sp_id"])].iloc[-1]
    out3 = pred.predict_player(str(int(other["player_id"])), g["date"])
    assert out3["pitching"]["probable_starter"] is False and out3["pitching"]["listed_starter"] is not None
    from mps.predict import _weather_text
    assert _weather_text({"day_night": "night", "temp_f": None}) == "night game, forecast not available yet"
    # with an explicit opponent the context becomes manual and a probable starter is guessed
    opp = int(sp["opp_team_id"])
    out2 = pred.predict_player(str(int(sp["player_id"])), future, opponent=opp)
    assert out2["context_source"] == "manual" and out2["opponent"] is not None


def test_predict_games_for_date_and_matchup(trained, small_dataset):
    pred = Predictor(ds=small_dataset, models=ModelBundle.load(trained))
    date = small_dataset.games["date"].max()
    table = pred.predict_games(date)
    assert len(table) == (small_dataset.games["date"] == date).sum()
    assert table["home_win_prob"].between(0.05, 0.95).all()
    assert (table["exp_total_runs"] > 3).all()
    g = small_dataset.games.iloc[0]
    one = pred.predict_games(date + pd.Timedelta(days=1), home=int(g["home_team_id"]), away=int(g["away_team_id"]))
    assert len(one) == 1 and one["predicted_winner"].iloc[0] in (one["home"].iloc[0], one["away"].iloc[0])


def test_cli_simulate_train_predict(tmp_path, capsys):
    data_dir, models_dir = tmp_path / "data", tmp_path / "models"
    from mps.data.simulate import simulate_dataset
    simulate_dataset([2023, 2024], games_per_team=40, seed=5, n_teams=6).save(data_dir)
    assert main(["train", "--data-dir", str(data_dir), "--models-dir", str(models_dir), "--max-rounds", "40"]) == 0
    assert main(["info", "--data-dir", str(data_dir), "--models-dir", str(models_dir)]) == 0
    from mps.data.store import Dataset
    ds = Dataset.load(data_dir)
    date = ds.games["date"].max().strftime("%Y-%m-%d")
    pid = str(int(ds.batting_lines.iloc[-1]["player_id"]))
    capsys.readouterr()
    assert main(["predict-player", pid, "--date", date, "--json",
                 "--data-dir", str(data_dir), "--models-dir", str(models_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["player_id"] == int(pid) and "batting" in payload
    assert main(["predict-game", "--date", date, "--data-dir", str(data_dir), "--models-dir", str(models_dir)]) == 0
    assert "home_win_prob" in capsys.readouterr().out


def test_predict_upcoming_game_with_announced_lineup(trained, small_dataset):
    ds = small_dataset
    tomorrow = ds.games["date"].max() + pd.Timedelta(days=1)
    g = ds.games.iloc[-1]
    upcoming = pd.DataFrame([{
        "game_pk": 555001, "date": tomorrow, "season": 2024, "game_type": "R",
        "home_team_id": g["home_team_id"], "away_team_id": g["away_team_id"],
        "home_score": None, "away_score": None, "home_sp_id": None, "away_sp_id": None,
        "venue_id": g["home_team_id"], "status": "Preview",
        "day_night": "day", "temp_f": 95.0, "wind_mph": 15.0, "wind_dir": "Out To CF", "condition": "Sunny",
        "hp_umpire_id": int(ds.games["hp_umpire_id"].iloc[-1]), "hp_umpire_name": ds.games["hp_umpire_name"].iloc[-1],
    }])
    home_bats = ds.batting_lines[ds.batting_lines["team_id"] == g["home_team_id"]]
    last9 = home_bats[home_bats["game_pk"] == home_bats.sort_values("date")["game_pk"].iloc[-1]]
    last9 = last9.sort_values("batting_order")["player_id"].tolist()[:9]
    lineup = pd.DataFrame([{"game_pk": 555001, "date": tomorrow, "team_id": g["home_team_id"],
                            "player_id": pid, "batting_order": 9 - i} for i, pid in enumerate(last9)])
    from mps.data.store import Dataset
    ds2 = ds.concat(Dataset(games=upcoming, batting_lines=ds.batting_lines.iloc[:0],
                            pitching_lines=ds.pitching_lines.iloc[:0], lineups=lineup))
    pred = Predictor(ds=ds2, models=ModelBundle.load(trained))
    leadoff_last_game = last9[0]
    out = pred.predict_player(str(leadoff_last_game), tomorrow)
    b = out["batting"]
    assert out["context_source"] == "schedule" and out["weather"]["temp_f"] == 95.0
    assert b["lineup"] == {"source": "announced", "in_lineup": True}
    assert b["batting_order"] == 9  # announced slot overrides the last-game slot
    assert b["bats"] in ("L", "R", "S") and b["opposing_starter_hand"] in ("L", "R")
    assert b["form"]["label"] in ("hot", "cold", "steady") and b["form"]["last5"]["games"] == 5
    assert b["split_vs_hand"] is not None and 0 <= b["split_vs_hand"]["avg"] <= 1
    assert out["umpire"]["name"] == ds.games["hp_umpire_name"].iloc[-1] and out["umpire"]["games"] > 0
    assert 18 < out["age"] < 45
    assert b["contact"]["ev_r30"] > 60 and b["opposing_starter_stuff"]["velo_r10"] > 80
    assert b["prior"]["season"] == 2024 and 0 < b["prior"]["reliability"] < 1
    assert out["opp_bullpen"]["arms_used_last_game"] >= 0 and "back_to_back_arms" in out["opp_bullpen"]
    assert b["opposing_starter_source"] == "rotation_guess"   # upcoming row had no probable pitcher
    # the opposing starter can be overridden by name or id
    sp = ds.pitching_lines[(ds.pitching_lines["is_starter"] == 1) & (ds.pitching_lines["team_id"] == g["away_team_id"])].iloc[0]
    out_o = pred.predict_player(str(leadoff_last_game), tomorrow, opp_starter=int(sp["player_id"]))
    assert out_o["batting"]["opposing_starter_source"] == "override"
    assert out_o["batting"]["opposing_starter"] == sp["player_name"]
    # a player who is not in the announced lineup is flagged
    bench = home_bats[~home_bats["player_id"].isin(last9)]["player_id"].iloc[0]
    assert pred.predict_player(str(int(bench)), tomorrow)["batting"]["lineup"]["in_lineup"] is False
    table = pred.predict_games(tomorrow)
    assert len(table) == 1 and table["lineups"].iloc[0] == "anno/prev"
    assert "95F" in table["weather"].iloc[0] and table["home_sp"].iloc[0] is not None
    assert table["umpire"].iloc[0] == ds.games["hp_umpire_name"].iloc[-1]


def test_predict_team_completed_and_upcoming(trained, small_dataset):
    ds = small_dataset
    pred = Predictor(ds=ds, models=ModelBundle.load(trained))
    g = ds.games.sort_values("date").iloc[-1]
    r = pred.predict_team(int(g["home_team_id"]), g["date"])
    assert r["final"] and r["lineup_source"] == "box_score" and r["final_score"]
    assert len(r["batters"]) == 9 and [b["batting_order"] for b in r["batters"]] == list(range(1, 10))
    for b in r["batters"]:
        assert 0 < b["expected"]["h"] < 3 and b["actual"] is not None and b["actual"]["ab"] >= 0
    assert r["pitcher"]["source"] == "box_score" and r["pitcher"]["actual"]["started"] is True
    assert r["pitcher"]["player_id"] == int(g["home_sp_id"])
    # upcoming game with no lineup announced: previous game's nine, no actuals
    tomorrow = ds.games["date"].max() + pd.Timedelta(days=1)
    r2 = pred.predict_team(int(g["away_team_id"]), tomorrow, opponent=int(g["home_team_id"]), is_home=0)
    assert not r2["final"] and r2["lineup_source"] == "previous_game" and len(r2["batters"]) == 9
    assert all(b["actual"] is None for b in r2["batters"]) and r2["pitcher"]["source"] == "rotation_guess"
