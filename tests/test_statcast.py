import numpy as np
import pandas as pd

from mps.data.statcast import aggregate_pitches, is_barrel, savant_params


def _pitch(game_pk, batter, pitcher, desc, ptype="FF", speed=95.0, zone=5, typ=None, ev=None, la=None,
           xba=None, xwoba=None, spin=2300):
    return {"game_pk": game_pk, "game_date": "2025-06-01", "batter": batter, "pitcher": pitcher, "events": None,
            "description": desc, "zone": zone, "pitch_type": ptype, "release_speed": speed,
            "release_spin_rate": spin, "launch_speed": ev, "launch_angle": la,
            "type": typ or ("X" if desc.startswith("hit_into_play") else "S"),
            "estimated_ba_using_speedangle": xba, "estimated_woba_using_speedangle": xwoba}


def test_barrel_rule():
    assert is_barrel(pd.Series([98, 100, 116, 97, 105]), pd.Series([26, 40, 50, 26, 3])).tolist() == \
        [True, False, True, False, False]


def test_savant_params_window_and_type():
    p = savant_params("2025-06-01", "2025-06-03", "batter")
    assert p["game_date_gt"] == "2025-06-01" and p["game_date_lt"] == "2025-06-03"
    assert p["type"] == "details" and p["player_type"] == "batter" and p["all"] == "true"


def test_aggregate_pitches_per_batter_and_pitcher():
    rows = [
        _pitch(1, 10, 20, "called_strike", zone=5, speed=96.0),
        _pitch(1, 10, 20, "swinging_strike", zone=13, speed=97.0),            # chase + whiff
        _pitch(1, 10, 20, "hit_into_play", ptype="SL", speed=85.0, zone=4, ev=104.0, la=25.0, xba=0.8, xwoba=1.6),
        _pitch(1, 11, 20, "ball", zone=12, speed=95.0),
        _pitch(1, 11, 20, "hit_into_play", ptype="SI", speed=94.0, zone=5, ev=80.0, la=10.0, xba=0.2, xwoba=0.18),
        _pitch(1, 11, 20, "foul", ptype="CU", speed=80.0, zone=6),
    ]
    bat, pit = aggregate_pitches(pd.DataFrame(rows))
    b10 = bat[bat["player_id"] == 10].iloc[0]
    assert b10["pitches"] == 3 and b10["swings"] == 2 and b10["whiffs"] == 1 and b10["chases"] == 1
    assert b10["bip"] == 1 and b10["ev_sum"] == 104.0 and b10["ev_max"] == 104.0 and b10["hard_hit"] == 1
    assert b10["barrels"] == 1 and b10["xwoba_sum"] == 1.6 and b10["xba_sum"] == 0.8 and b10["la_sum"] == 25.0
    b11 = bat[bat["player_id"] == 11].iloc[0]
    assert b11["swings"] == 2 and b11["whiffs"] == 0 and b11["hard_hit"] == 0 and b11["barrels"] == 0
    assert b11["out_zone_pitches"] == 1
    p = pit.iloc[0]
    assert p["player_id"] == 20 and p["pitches"] == 6 and p["fastballs"] == 4  # FF, FF, FF (the ball), SI
    assert abs(p["fb_velo_sum"] - (96 + 97 + 95 + 94)) < 1e-9 and p["fb_velo_max"] == 97.0
    assert p["whiffs"] == 1 and p["called_strikes"] == 1 and p["swings"] == 4 and p["chases"] == 1
    assert p["bip"] == 2 and p["barrels"] == 1 and p["hard_hit"] == 1
    assert str(bat["date"].iloc[0])[:10] == "2025-06-01"


def test_aggregate_handles_missing_columns_and_empty():
    bat, pit = aggregate_pitches(pd.DataFrame())
    assert bat.empty and pit.empty
    df = pd.DataFrame([{"game_pk": 1, "game_date": "2025-06-01", "batter": 1, "pitcher": 2,
                        "description": "ball", "type": "B"}])
    bat, pit = aggregate_pitches(df)
    assert len(bat) == 1 and bat.iloc[0]["bip"] == 0 and np.isnan(bat.iloc[0]["ev_max"])
