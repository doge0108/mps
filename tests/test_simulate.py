import pandas as pd

from mps.config import BATTING_STATS, PITCHING_STATS


def test_schema_and_ranges(small_dataset):
    ds = small_dataset
    assert set(BATTING_STATS) <= set(ds.batting_lines.columns)
    assert set(PITCHING_STATS) <= set(ds.pitching_lines.columns)
    assert ds.seasons() == [2023, 2024]
    per_team = ds.games.groupby(["season", "home_team_id"]).size() + ds.games.groupby(["season", "away_team_id"]).size()
    assert (per_team == 60).all()
    b = ds.batting_lines
    avg = b["h"].sum() / b["ab"].sum()
    assert 0.20 < avg < 0.32
    k_rate = b["so"].sum() / b["pa"].sum()
    assert 0.15 < k_rate < 0.30
    rpg = (ds.games["home_score"] + ds.games["away_score"]).mean()
    assert 6 < rpg < 12


def test_box_scores_are_internally_consistent(small_dataset):
    ds = small_dataset
    runs = ds.batting_lines.groupby("game_pk")["r"].sum()
    totals = (ds.games.set_index("game_pk")["home_score"] + ds.games.set_index("game_pk")["away_score"]).astype(float)
    assert (runs.reindex(totals.index) == totals).mean() > 0.99
    b = ds.batting_lines
    assert (b["tb"] >= b["h"]).all()
    assert (b["h"] <= b["ab"]).all()
    assert ds.pitching_lines.groupby("game_pk")["is_starter"].sum().eq(2).all()


def test_simulation_is_reproducible():
    from mps.data.simulate import simulate_dataset
    a = simulate_dataset([2024], games_per_team=10, seed=3, n_teams=4)
    b = simulate_dataset([2024], games_per_team=10, seed=3, n_teams=4)
    pd.testing.assert_frame_equal(a.games, b.games)
    pd.testing.assert_frame_equal(a.batting_lines, b.batting_lines)
