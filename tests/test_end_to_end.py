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
