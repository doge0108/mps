import numpy as np
import pandas as pd

from mps.features.build import (States, build_batter_training, build_game_training,
                                build_pitcher_training, feature_columns)
from mps.features.states import asof_join


def test_asof_join_uses_only_strictly_earlier_state():
    state = pd.DataFrame({
        "player_id": [1, 1, 1],
        "date": pd.to_datetime(["2024-04-01", "2024-04-02", "2024-04-05"]),
        "val": [10.0, 20.0, 30.0],
    })
    left = pd.DataFrame({
        "player_id": [1, 1, 1, 2],
        "date": pd.to_datetime(["2024-04-01", "2024-04-02", "2024-04-09", "2024-04-09"]),
    })
    out = asof_join(left, state, "player_id", "player_id", prefix="x_")
    assert out["x_val"].tolist()[:3] == [None, 10.0, 30.0] or (
        np.isnan(out["x_val"].iloc[0]) and out["x_val"].tolist()[1:3] == [10.0, 30.0])
    assert np.isnan(out["x_val"].iloc[3])  # unknown player -> NaN features


def test_batter_features_are_leak_free(small_dataset):
    ds = small_dataset
    states = States.from_dataset(ds)
    feats = build_batter_training(ds, states)
    pid = feats["player_id"].value_counts().index[0]
    lines = ds.batting_lines[ds.batting_lines["player_id"] == pid].sort_values("date").reset_index(drop=True)
    rows = feats[feats["player_id"] == pid].sort_values("date").reset_index(drop=True)
    assert np.isnan(rows["b_h_r7"].iloc[0])  # no history before the first game
    for i in (5, 12, 30):
        expected = lines["h"].iloc[max(0, i - 7):i].mean()
        assert abs(rows["b_h_r7"].iloc[i] - expected) < 1e-9
    # cumulative averages exclude the current game
    i = 25
    exp_avg = lines["h"].iloc[:i].sum() / lines["ab"].iloc[:i].sum()
    assert abs(rows["b_avg_cum"].iloc[i] - exp_avg) < 1e-9


def test_feature_matrices_have_targets_and_numeric_features(small_dataset):
    ds = small_dataset
    states = States.from_dataset(ds)
    bf = build_batter_training(ds, states)
    pf = build_pitcher_training(ds, states)
    gf = build_game_training(ds, states)
    assert len(bf) == len(ds.batting_lines)
    assert len(pf) == int((ds.pitching_lines["is_starter"] == 1).sum())
    assert len(gf) == len(ds.games)
    for frame in (bf, pf, gf):
        cols = feature_columns(frame)
        assert len(cols) > 20
        assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in cols)
        assert not any(c.startswith("y_") for c in cols)
    assert {"y_h", "y_hr", "y_ab"} <= set(bf.columns)
    assert {"y_so", "y_er", "y_outs"} <= set(pf.columns)
    assert {"y_home_win", "y_home_runs", "y_away_runs"} <= set(gf.columns)


def test_elo_and_team_form_are_pre_game(small_dataset):
    ds = small_dataset
    gf = build_game_training(ds)
    first_dates = ds.games.groupby("home_team_id")["date"].min()
    # a team's first game of the dataset has no prior form
    first_game = gf.sort_values("date").iloc[0]
    assert np.isnan(first_game["home_tm_win_r10"]) and np.isnan(first_game["home_tm_elo"])
    # mid-season games have Elo populated and centred near 1500
    mid = gf.dropna(subset=["home_tm_elo"])
    assert abs(mid["home_tm_elo"].mean() - 1500) < 25
