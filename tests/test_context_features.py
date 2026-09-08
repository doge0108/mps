"""Priors, age, umpire tendencies, bullpen usage and Statcast states."""
import numpy as np
import pandas as pd

from mps.features.build import States, build_batter_training, build_game_training, feature_columns
from mps.features.priors import age_on, batter_priors, pitcher_priors
from mps.features.states import bullpen_usage, statcast_batter_state, statcast_pitcher_state, umpire_state


def test_age_and_priors(small_dataset):
    ds = small_dataset
    bp = batter_priors(ds.batting_lines, ds.games, ds.players)
    # first data season has no prior; the season after the last one is projected too
    assert sorted(bp["date"].dt.year.unique()) == [2024, 2025]
    assert bp["mc_reliability"].between(0, 1).all() and bp["mc_avg"].between(0.15, 0.4).all()
    pid = bp["player_id"].iloc[0]
    hist = ds.batting_lines.merge(ds.games[["game_pk", "season"]], on="game_pk")
    h23 = hist[(hist["player_id"] == pid) & (hist["season"] == 2023)]
    lg = hist[hist["season"] == 2023]
    raw = h23["hr"].sum() / h23["pa"].sum()
    lg_rate = lg["hr"].sum() / lg["pa"].sum()
    row = bp[(bp["player_id"] == pid) & (bp["date"].dt.year == 2024)].iloc[0]
    # projection sits between the player's own rate and the league rate (before the small age factor)
    lo, hi = sorted([raw, lg_rate])
    assert lo * 0.95 <= row["mc_hr_rate"] <= hi * 1.05
    pp = pitcher_priors(ds.pitching_lines, ds.games, ds.players)
    assert {"mcp_era", "mcp_k_rate", "mcp_reliability"} <= set(pp.columns) and pp["mcp_era"].between(1, 12).all()
    ages = age_on(pd.Series(pd.to_datetime(["2024-06-15"])), pd.Series(pd.to_datetime(["1994-06-15"])))
    assert abs(ages.iloc[0] - 30.0) < 0.01
    feats = build_batter_training(ds)
    assert feats["age"].between(18, 45).all()
    in_2024 = feats[feats["date"].dt.year == 2024]
    assert in_2024["mc_hr_rate"].notna().mean() > 0.8      # priors available for returning players
    assert feats[feats["date"].dt.year == 2023]["mc_hr_rate"].isna().all()


def test_umpire_state_tracks_strikeout_tendency(small_dataset):
    ds = small_dataset
    st = umpire_state(ds.games, ds.pitching_lines)
    assert {"ump_k_rate_r100", "ump_bb_rate_r100", "ump_games"} <= set(st.columns)
    ump = st["hp_umpire_id"].value_counts().index[0]
    worked = ds.games[ds.games["hp_umpire_id"] == ump].sort_values("date")
    per_game = ds.pitching_lines.groupby("game_pk")[["so", "bf"]].sum()
    tot = per_game.loc[worked["game_pk"]]
    lg = ds.pitching_lines["so"].sum() / ds.pitching_lines["bf"].sum()
    last = st[st["hp_umpire_id"] == ump].sort_values("date").iloc[-1]
    assert abs(last["ump_k_rate_r100"] - (tot["so"].mean() / tot["bf"].mean() - lg)) < 1e-9
    assert last["ump_games"] == len(worked)


def test_bullpen_usage_counts_back_to_back_arms(small_dataset):
    ds = small_dataset
    use = bullpen_usage(ds.games, ds.pitching_lines)
    assert len(use) == 2 * len(ds.games)
    rp = ds.pitching_lines[ds.pitching_lines["is_starter"] == 0]
    team = rp["team_id"].iloc[0]
    log = ds.games[(ds.games["home_team_id"] == team) | (ds.games["away_team_id"] == team)].sort_values("date")
    # pick a pair of consecutive-day games and verify the overlap count by hand
    dates = log["date"].tolist(); pks = log["game_pk"].tolist()
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days == 1:
            prev = set(rp[(rp["game_pk"] == pks[i - 1]) & (rp["team_id"] == team)]["player_id"])
            cur = set(rp[(rp["game_pk"] == pks[i]) & (rp["team_id"] == team)]["player_id"])
            row = use[(use["team_id"] == team) & (use["game_pk"] == pks[i])].iloc[0]
            assert row["bp_b2b"] == len(prev & cur) and row["bp_arms"] == len(cur)
            break
    states = States.from_dataset(ds)
    assert {"tm_bp_b2b_last", "tm_bp_pitches_r3", "tm_bp_top3_used_last"} <= set(states.teams.columns)
    gf = build_game_training(ds, states)
    assert "bp_fatigue_gap" in gf.columns and gf["home_tm_bp_arms_last"].notna().mean() > 0.9


def test_statcast_states_and_features(small_dataset):
    ds = small_dataset
    sb = statcast_batter_state(ds.statcast_batting)
    sp = statcast_pitcher_state(ds.statcast_pitching)
    pid = ds.statcast_batting["player_id"].value_counts().index[0]
    sc = ds.statcast_batting[ds.statcast_batting["player_id"] == pid].sort_values("date")
    last = sb[sb["player_id"] == pid].sort_values("date").iloc[-1]
    tail = sc.tail(30)
    assert abs(last["sc_ev_r30"] - tail["ev_sum"].sum() / tail["bip"].sum()) < 1e-9
    assert abs(last["sc_barrel_r30"] - tail["barrels"].sum() / tail["bip"].sum()) < 1e-9
    assert sp["scp_velo_r10"].between(80, 105).all() and "scp_velo_delta" in sp.columns
    feats = build_batter_training(ds)
    cols = set(feature_columns(feats))
    assert {"sc_ev_r30", "sc_form_ev", "opp_scp_velo_delta", "opp_scp_csw_r10", "ump_k_rate_r100",
            "opp_tm_bp_b2b_last", "mc_reliability", "age"} <= cols
    # exit velocity carries information about power beyond outcomes: it correlates with HR rate
    per = feats.groupby("player_id").agg(ev=("sc_ev_r60", "mean"), hr=("y_hr", "mean"))
    assert per["ev"].corr(per["hr"]) > 0.5


def test_pipeline_without_statcast_or_umpires(small_dataset, tmp_path):
    from mps.data.store import Dataset
    from mps.models.registry import train_all
    from mps.predict import Predictor
    ds = Dataset(games=small_dataset.games.assign(hp_umpire_id=None, hp_umpire_name=None),
                 batting_lines=small_dataset.batting_lines, pitching_lines=small_dataset.pitching_lines,
                 players=small_dataset.players)
    bundle = train_all(ds, max_rounds=20, verbose=False)
    assert not any(c.startswith("sc_") for c in bundle.batter.feature_cols)
    # a model trained with Statcast still predicts when the live data has none (columns become NaN)
    full = train_all(small_dataset, max_rounds=20, verbose=False)
    pred = Predictor(ds=ds, models=full)
    pid = str(int(ds.batting_lines.iloc[-1]["player_id"]))
    out = pred.predict_player(pid, ds.games["date"].max() + pd.Timedelta(days=1))
    assert out["batting"]["contact"] is None and out["umpire"] is None and 0 < out["batting"]["expected"]["h"] < 3
