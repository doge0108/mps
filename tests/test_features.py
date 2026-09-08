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
    assert np.isnan(rows["b_h_r5"].iloc[0])  # no history before the first game
    for i in (5, 12, 30):
        expected = lines["h"].iloc[max(0, i - 5):i].mean()
        assert abs(rows["b_h_r5"].iloc[i] - expected) < 1e-9
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
    assert "bat_starter" in feature_columns(bf) and bf["bat_starter"].eq(1).all()
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


def test_streaks_and_form_features(small_dataset):
    from mps.features.states import batter_state
    ds = small_dataset
    st = batter_state(ds.batting_lines)
    pid = ds.batting_lines["player_id"].value_counts().index[0]
    lines = ds.batting_lines[ds.batting_lines["player_id"] == pid].sort_values(["date", "game_pk"]).reset_index(drop=True)
    rows = st[st["player_id"] == pid].sort_values("date").reset_index(drop=True)
    assert len(rows) == len(lines)  # one game per day in the simulator
    hits = (lines["h"] > 0).tolist()
    for i in (0, 7, 19, 33):
        streak = 0
        for j in range(i, -1, -1):
            if hits[j]:
                streak += 1
            else:
                break
        assert rows["b_hit_streak"].iloc[i] == streak
        assert rows["b_hitless_streak"].iloc[i] == (0 if hits[i] else next(
            (k for k in range(1, i + 2) if i - k < 0 or hits[i - k]), i + 1))
    i = 40
    exp_form = lines["tb"].iloc[i - 4:i + 1].sum() / lines["ab"].iloc[i - 4:i + 1].sum() \
        - lines["tb"].iloc[i - 29:i + 1].sum() / lines["ab"].iloc[i - 29:i + 1].sum()
    assert abs(rows["b_form_slg"].iloc[i] - exp_form) < 1e-9


def test_platoon_split_state_is_keyed_by_opposing_hand(small_dataset):
    from mps.features.states import batter_split_state, split_key
    ds = small_dataset
    st = batter_split_state(ds.batting_lines, ds.players)
    throws = ds.players.set_index("player_id")["throws"]
    lines = ds.batting_lines.assign(hand=ds.batting_lines["opp_sp_id"].map(throws))
    pid = lines["player_id"].value_counts().index[0]
    vs_l = lines[(lines["player_id"] == pid) & (lines["hand"] == "L")].sort_values("date")
    assert len(vs_l) > 5
    key = int(split_key(pd.Series([pid]), pd.Series(["L"])).iloc[0])
    rows = st[st["split_key"] == key].sort_values("date").reset_index(drop=True)
    assert len(rows) == len(vs_l)
    n = min(len(vs_l), 40)
    exp = vs_l["h"].tail(n).sum() / vs_l["ab"].tail(n).sum()
    assert abs(rows["sp_avg"].iloc[-1] - exp) < 1e-9
    feats = build_batter_training(ds)
    assert feats["platoon_edge"].isin([-1.0, 1.0]).all()
    # the batter row for a game vs a lefty uses the vs-L split from strictly earlier games
    r = feats[(feats["player_id"] == pid) & (feats["opp_hand"] == "L")].sort_values("date").iloc[3]
    prior = vs_l[vs_l["date"] < r["date"]]
    assert abs(r["sp_avg"] - prior["h"].sum() / prior["ab"].sum()) < 1e-9


def test_weather_features_encoding():
    from mps.features.build import weather_features
    df = pd.DataFrame({
        "temp_f": [85, None, 60], "wind_mph": [12, None, 8],
        "wind_dir": ["Out To CF", None, "In From LF"], "condition": ["Clear", None, "Roof Closed"],
        "day_night": ["night", None, "day"],
    })
    wx = weather_features(df)
    assert wx["wx_wind_out"].tolist()[0] == 1.0 and wx["wx_wind_out_mph"].iloc[0] == 12.0
    assert wx["wx_wind_in"].iloc[2] == 1.0 and wx["wx_wind_in_mph"].iloc[2] == 8.0
    assert wx["wx_dome"].tolist()[2] == 1.0 and wx["wx_dome"].iloc[0] == 0.0
    assert wx["wx_night"].tolist()[0] == 1.0 and wx["wx_night"].iloc[2] == 0.0
    assert wx.iloc[1].isna().all()


def test_lineup_features_aggregate_the_nine_starters(small_dataset):
    from mps.features.lineups import latest_lineup, lineup_features, starters_from_lines
    ds = small_dataset
    states = States.from_dataset(ds)
    starters = starters_from_lines(ds.batting_lines)
    assert starters.groupby(["game_pk", "team_id"]).size().eq(9).all()
    lf = lineup_features(starters, states.batters, ds.players)
    assert len(lf) == 2 * len(ds.games)
    late = lf.merge(ds.games[["game_pk", "date"]], on="game_pk")
    late = late[late["date"] > ds.games["date"].min() + pd.Timedelta(days=20)]
    assert late["lu_known"].eq(9).mean() > 0.9
    assert late["lu_lhb_share"].between(0, 1).all() and late["lu_slg_r30"].between(0.1, 0.9).all()
    team = int(ds.games.iloc[-1]["home_team_id"])
    fallback = latest_lineup(ds.batting_lines, team, ds.games["date"].max() + pd.Timedelta(days=1))
    assert len(fallback) == 9 and fallback["batting_order"].tolist() == list(range(1, 10))
    gf = build_game_training(ds, states)
    assert {"hlu_slg_r30", "alu_lhb_share", "lu_slg_gap", "wx_temp_f", "home_tm_streak"} <= set(gf.columns)
