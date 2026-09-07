"""Post-game "state" tables and the as-of join that makes features leak-free.

A state row summarises everything known about a player/team *after* the games
on a given date.  Features for a game on date D are obtained by joining the
latest state row with date strictly before D (``asof_join``).  Training and
live prediction therefore share exactly the same code path: to predict a
future game we simply look up the latest available states.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HAND_CODES
from .common import safe_div
from .team_features import ELO_HOME, ELO_K, ELO_SEASON_REGRESS, ELO_START, team_game_log

BATTER_ROLL_COLS = ["pa", "ab", "h", "hr", "rbi", "r", "bb", "so", "sb", "tb", "d"]
BATTER_WINDOWS = (3, 5, 15, 30, 60)
SPLIT_WINDOW = 40          # games vs. same-handed starters used for platoon splits
PITCHER_ROLL_COLS = ["outs", "h", "er", "bb", "so", "hr", "bf", "pitches"]
PITCHER_WINDOWS = (3, 5, 10, 30)
TEAM_WINDOWS = (5, 10, 30, 162)


def _rolling_inclusive(df: pd.DataFrame, by: str, cols: list[str], window: int, prefix: str) -> pd.DataFrame:
    grp = df.groupby(by, sort=False)[cols]
    out = grp.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
    out.columns = [f"{prefix}{c}_r{window}" for c in cols]
    return out


def _streak(flag: pd.Series, by: pd.Series) -> pd.Series:
    """Length of the current run of True values within each group (inclusive of the row)."""
    flag = flag.astype(bool)
    block = (~flag).groupby(by).cumsum()
    return flag.groupby([by, block]).cumsum().astype(float)


def _collapse_same_day(state: pd.DataFrame, key: str) -> pd.DataFrame:
    """Keep the last row per (key, date) so doubleheaders yield one state per day."""
    return state.drop_duplicates(subset=[key, "date"], keep="last").reset_index(drop=True)


def hand_code(series: pd.Series) -> pd.Series:
    return series.map(HAND_CODES).astype("float")


# ---------------------------------------------------------------- batters
def batter_state(batting_lines: pd.DataFrame) -> pd.DataFrame:
    df = batting_lines.sort_values(["player_id", "date", "game_pk"]).reset_index(drop=True)
    parts = [df[["player_id", "date", "team_id", "player_name", "batting_order"]].rename(
        columns={"team_id": "b_team_id", "player_name": "b_name", "batting_order": "b_last_order"})]
    for w in BATTER_WINDOWS:
        parts.append(_rolling_inclusive(df, "player_id", BATTER_ROLL_COLS, w, "b_"))
    grp = df.groupby("player_id", sort=False)
    cum = grp[BATTER_ROLL_COLS].cumsum()
    cum.columns = [f"b_{c}_cum" for c in BATTER_ROLL_COLS]
    parts.append(cum)
    st = pd.concat(parts, axis=1)
    st["b_games"] = (grp.cumcount() + 1).clip(upper=150)
    st["b_last_date"] = df["date"]
    for w in (5, 15, 30, 60):
        st[f"b_avg_r{w}"] = safe_div(st[f"b_h_r{w}"], st[f"b_ab_r{w}"])
        st[f"b_hr_rate_r{w}"] = safe_div(st[f"b_hr_r{w}"], st[f"b_pa_r{w}"])
        st[f"b_bb_rate_r{w}"] = safe_div(st[f"b_bb_r{w}"], st[f"b_pa_r{w}"])
        st[f"b_k_rate_r{w}"] = safe_div(st[f"b_so_r{w}"], st[f"b_pa_r{w}"])
        st[f"b_slg_r{w}"] = safe_div(st[f"b_tb_r{w}"], st[f"b_ab_r{w}"])
        st[f"b_sb_rate_r{w}"] = safe_div(st[f"b_sb_r{w}"], st[f"b_pa_r{w}"])
    st["b_avg_cum"] = safe_div(st["b_h_cum"], st["b_ab_cum"])
    st["b_hr_rate_cum"] = safe_div(st["b_hr_cum"], st["b_pa_cum"])
    st["b_k_rate_cum"] = safe_div(st["b_so_cum"], st["b_pa_cum"])
    st["b_bb_rate_cum"] = safe_div(st["b_bb_cum"], st["b_pa_cum"])
    st["b_slg_cum"] = safe_div(st["b_tb_cum"], st["b_ab_cum"])
    # --- form: hot / cold relative to the batter's own 30-game baseline
    st["b_form_avg"] = st["b_avg_r5"] - st["b_avg_r30"]
    st["b_form_slg"] = st["b_slg_r5"] - st["b_slg_r30"]
    st["b_form_k"] = st["b_k_rate_r5"] - st["b_k_rate_r30"]
    st["b_form_hr"] = st["b_hr_rate_r5"] - st["b_hr_rate_r30"]
    st["b_form_tb3"] = st["b_tb_r3"] - st["b_tb_r30"]
    st["b_hit_streak"] = _streak(df["h"] > 0, df["player_id"])
    st["b_hitless_streak"] = _streak(df["h"] == 0, df["player_id"])
    st["b_hr_drought"] = _streak(df["hr"] == 0, df["player_id"])
    st["b_multi_hit_r5"] = _rolling_inclusive(df.assign(mh=(df["h"] >= 2).astype(float)), "player_id",
                                              ["mh"], 5, "x_")["x_mh_r5"]
    # raw cumulative counts are non-stationary, keep only rates
    st = st.drop(columns=[f"b_{c}_cum" for c in BATTER_ROLL_COLS])
    st = st.sort_values(["date", "player_id"]).reset_index(drop=True)
    return _collapse_same_day(st, "player_id")


def batter_split_state(batting_lines: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Rolling performance vs. left- and right-handed starters, keyed by (player_id, opp hand)."""
    if players is None or players.empty or "throws" not in players.columns:
        return pd.DataFrame(columns=["split_key", "date"])
    hands = players[["player_id", "throws"]].rename(columns={"player_id": "opp_sp_id", "throws": "opp_hand"})
    df = batting_lines.merge(hands, on="opp_sp_id", how="inner")
    df = df[df["opp_hand"].isin(["L", "R"])].copy()
    if df.empty:
        return pd.DataFrame(columns=["split_key", "date"])
    df["split_key"] = split_key(df["player_id"], df["opp_hand"])
    df = df.sort_values(["split_key", "date", "game_pk"]).reset_index(drop=True)
    cols = ["pa", "ab", "h", "hr", "bb", "so", "tb"]
    roll = _rolling_inclusive(df, "split_key", cols, SPLIT_WINDOW, "sp_")
    st = pd.concat([df[["split_key", "date"]], roll], axis=1)
    st["sp_games"] = (df.groupby("split_key", sort=False).cumcount() + 1).clip(upper=SPLIT_WINDOW).astype(float)
    st["sp_avg"] = safe_div(st[f"sp_h_r{SPLIT_WINDOW}"], st[f"sp_ab_r{SPLIT_WINDOW}"])
    st["sp_slg"] = safe_div(st[f"sp_tb_r{SPLIT_WINDOW}"], st[f"sp_ab_r{SPLIT_WINDOW}"])
    st["sp_hr_rate"] = safe_div(st[f"sp_hr_r{SPLIT_WINDOW}"], st[f"sp_pa_r{SPLIT_WINDOW}"])
    st["sp_k_rate"] = safe_div(st[f"sp_so_r{SPLIT_WINDOW}"], st[f"sp_pa_r{SPLIT_WINDOW}"])
    st["sp_bb_rate"] = safe_div(st[f"sp_bb_r{SPLIT_WINDOW}"], st[f"sp_pa_r{SPLIT_WINDOW}"])
    st = st.drop(columns=[f"sp_{c}_r{SPLIT_WINDOW}" for c in cols])
    st = st.sort_values(["date", "split_key"]).reset_index(drop=True)
    return _collapse_same_day(st, "split_key")


def split_key(player_id: pd.Series, hand: pd.Series) -> pd.Series:
    """Composite as-of key: player_id * 4 + hand code (L=1, R=2)."""
    code = pd.Series(hand).map({"L": 1, "R": 2}).fillna(0).astype("int64")
    return pd.to_numeric(player_id, errors="coerce").fillna(-1).astype("int64") * 4 + code.to_numpy()


# --------------------------------------------------------------- pitchers
def pitcher_state(pitching_lines: pd.DataFrame) -> pd.DataFrame:
    df = pitching_lines.sort_values(["player_id", "date", "game_pk"]).reset_index(drop=True)
    parts = [df[["player_id", "date", "team_id", "player_name"]].rename(
        columns={"team_id": "p_team_id", "player_name": "p_name"})]
    for w in PITCHER_WINDOWS:
        parts.append(_rolling_inclusive(df, "player_id", PITCHER_ROLL_COLS, w, "p_"))
    grp = df.groupby("player_id", sort=False)
    cum = grp[PITCHER_ROLL_COLS].cumsum()
    cum.columns = [f"p_{c}_cum" for c in PITCHER_ROLL_COLS]
    parts.append(cum)
    st = pd.concat(parts, axis=1)
    st["p_apps"] = (grp.cumcount() + 1).clip(upper=60)
    st["p_starts"] = grp["is_starter"].cumsum().clip(upper=40)
    st["p_start_share_r10"] = _rolling_inclusive(df, "player_id", ["is_starter"], 10, "x_")["x_is_starter_r10"]
    st["p_last_date"] = df["date"]
    for w in (3, 5, 10, 30):
        st[f"p_k_rate_r{w}"] = safe_div(st[f"p_so_r{w}"], st[f"p_bf_r{w}"])
        st[f"p_bb_rate_r{w}"] = safe_div(st[f"p_bb_r{w}"], st[f"p_bf_r{w}"])
        st[f"p_hr_rate_r{w}"] = safe_div(st[f"p_hr_r{w}"], st[f"p_bf_r{w}"])
        st[f"p_h_rate_r{w}"] = safe_div(st[f"p_h_r{w}"], st[f"p_bf_r{w}"])
        st[f"p_era_r{w}"] = safe_div(st[f"p_er_r{w}"] * 27.0, st[f"p_outs_r{w}"])
    st["p_k_rate_cum"] = safe_div(st["p_so_cum"], st["p_bf_cum"])
    st["p_bb_rate_cum"] = safe_div(st["p_bb_cum"], st["p_bf_cum"])
    st["p_hr_rate_cum"] = safe_div(st["p_hr_cum"], st["p_bf_cum"])
    st["p_era_cum"] = safe_div(st["p_er_cum"] * 27.0, st["p_outs_cum"])
    # form vs. own baseline
    st["p_form_era"] = st["p_era_r3"] - st["p_era_r30"]
    st["p_form_k"] = st["p_k_rate_r3"] - st["p_k_rate_r30"]
    st["p_form_outs"] = st["p_outs_r3"] - st["p_outs_r30"]
    st["p_qs_streak"] = _streak((df["outs"] >= 18) & (df["er"] <= 3), df["player_id"])
    st = st.drop(columns=[f"p_{c}_cum" for c in PITCHER_ROLL_COLS])
    st = st.sort_values(["date", "player_id"]).reset_index(drop=True)
    return _collapse_same_day(st, "player_id")


# ------------------------------------------------------------------ teams
def _post_game_elo(games: pd.DataFrame) -> pd.DataFrame:
    g = games.sort_values(["date", "game_pk"]).reset_index(drop=True)
    ratings: dict[int, float] = {}
    season_seen = None
    rows = []
    for row in g.itertuples(index=False):
        season = None if pd.isna(row.season) else int(row.season)
        if season is not None and season_seen is not None and season != season_seen:
            for tid in ratings:
                ratings[tid] += ELO_SEASON_REGRESS * (ELO_START - ratings[tid])
        if season is not None:
            season_seen = season
        h, a = int(row.home_team_id), int(row.away_team_id)
        rh, ra = ratings.get(h, ELO_START), ratings.get(a, ELO_START)
        if not (pd.isna(row.home_score) or pd.isna(row.away_score)):
            exp_home = 1.0 / (1.0 + 10 ** ((ra - (rh + ELO_HOME)) / 400.0))
            hs, as_ = float(row.home_score), float(row.away_score)
            result = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
            margin = abs(hs - as_)
            delta = ELO_K * (np.log1p(margin) if margin > 0 else 1.0) * (result - exp_home)
            rh, ra = rh + delta, ra - delta
            ratings[h], ratings[a] = rh, ra
        rows.append((row.game_pk, h, rh))
        rows.append((row.game_pk, a, ra))
    return pd.DataFrame(rows, columns=["game_pk", "team_id", "tm_elo"])


def team_state(games: pd.DataFrame, pitching_lines: pd.DataFrame | None = None) -> pd.DataFrame:
    """Post-game team state: form, streak, Elo, park factor, bullpen quality, last game date."""
    played = games[games["home_score"].notna() & games["away_score"].notna()]
    log = team_game_log(played)
    log = log.sort_values(["team_id", "date", "game_pk"]).reset_index(drop=True)
    parts = [log[["team_id", "date", "game_pk"]]]
    for w in TEAM_WINDOWS:
        parts.append(_rolling_inclusive(log, "team_id", ["rs", "ra", "win", "run_diff"], w, "tm_"))
    st = pd.concat(parts, axis=1)
    st["tm_games"] = (log.groupby("team_id", sort=False).cumcount() + 1).clip(upper=162)
    st["tm_last_date"] = log["date"]
    win_streak = _streak(log["win"] == 1, log["team_id"])
    loss_streak = _streak(log["win"] == 0, log["team_id"])
    st["tm_streak"] = win_streak - loss_streak  # +3 = won three straight, -2 = lost two straight
    league_avg = log["total_runs"].mean()
    is_home = log["is_home"] == 1
    home_tr = log["total_runs"].where(is_home)
    home_roll = home_tr.groupby(log["team_id"]).transform(lambda s: s.rolling(81, min_periods=5).mean())
    home_roll = home_roll.groupby(log["team_id"]).ffill()
    st["tm_park_factor"] = (home_roll / league_avg).to_numpy()
    st = st.merge(_post_game_elo(played), on=["game_pk", "team_id"], how="left")
    if pitching_lines is not None and len(pitching_lines):
        rp = pitching_lines[pitching_lines["is_starter"] == 0]
        agg = rp.groupby(["team_id", "game_pk"], sort=False)[["outs", "er", "so", "bf"]].sum().reset_index()
        agg = agg.merge(log[["team_id", "game_pk", "date"]], on=["team_id", "game_pk"], how="inner")
        agg = agg.sort_values(["team_id", "date", "game_pk"]).reset_index(drop=True)
        roll = _rolling_inclusive(agg, "team_id", ["outs", "er", "so", "bf"], 30, "bp_")
        agg = pd.concat([agg[["team_id", "game_pk"]], roll], axis=1)
        agg["tm_bullpen_era_r30"] = safe_div(agg["bp_er_r30"] * 27.0, agg["bp_outs_r30"])
        agg["tm_bullpen_k_rate_r30"] = safe_div(agg["bp_so_r30"], agg["bp_bf_r30"])
        st = st.merge(agg[["team_id", "game_pk", "tm_bullpen_era_r30", "tm_bullpen_k_rate_r30"]],
                      on=["team_id", "game_pk"], how="left")
        st = st.sort_values(["team_id", "date", "game_pk"]).reset_index(drop=True)
        for col in ("tm_bullpen_era_r30", "tm_bullpen_k_rate_r30"):
            st[col] = st.groupby("team_id", sort=False)[col].ffill()
        # bullpen workload: what the pen threw in the last game(s) and back-to-back arms
        use = bullpen_usage(played, pitching_lines)
        st = st.merge(use, on=["team_id", "game_pk"], how="left").sort_values(["team_id", "date", "game_pk"]).reset_index(drop=True)
        for col in ("bp_pitches", "bp_arms", "bp_b2b", "bp_top3_used"):
            st[f"tm_{col}_last"] = st[col].fillna(0.0)
        st["tm_bp_pitches_r3"] = _rolling_inclusive(st.rename(columns={"bp_pitches": "v"}).assign(v=lambda d: d["v"].fillna(0.0)), "team_id", ["v"], 3, "x_")["x_v_r3"] * 3
        st["tm_bp_arms_r3"] = _rolling_inclusive(st.rename(columns={"bp_arms": "v"}).assign(v=lambda d: d["v"].fillna(0.0)), "team_id", ["v"], 3, "x_")["x_v_r3"] * 3
        st = st.drop(columns=["bp_pitches", "bp_arms", "bp_b2b", "bp_top3_used"])
    st = st.drop(columns=["game_pk"]).sort_values(["date", "team_id"]).reset_index(drop=True)
    return _collapse_same_day(st, "team_id")


# --------------------------------------------------------------- statcast
SC_BAT_WINDOWS = (5, 15, 30, 60)
SC_PIT_WINDOWS = (3, 10, 30)


def statcast_batter_state(sc: pd.DataFrame) -> pd.DataFrame:
    """Rolling quality-of-contact and plate-discipline rates per batter (prefix ``sc_``)."""
    if sc is None or sc.empty:
        return pd.DataFrame(columns=["player_id", "date"])
    df = sc.sort_values(["player_id", "date", "game_pk"]).reset_index(drop=True)
    cols = ["pitches", "swings", "whiffs", "chases", "out_zone_pitches", "bip", "ev_sum", "la_sum", "hard_hit",
            "barrels", "xba_sum", "xwoba_sum"]
    parts = [df[["player_id", "date"]]]
    for w in SC_BAT_WINDOWS:
        parts.append(_rolling_inclusive(df, "player_id", cols, w, "x_"))
    st = pd.concat(parts, axis=1)
    for w in SC_BAT_WINDOWS:
        st[f"sc_ev_r{w}"] = safe_div(st[f"x_ev_sum_r{w}"], st[f"x_bip_r{w}"])
        st[f"sc_hard_hit_r{w}"] = safe_div(st[f"x_hard_hit_r{w}"], st[f"x_bip_r{w}"])
        st[f"sc_barrel_r{w}"] = safe_div(st[f"x_barrels_r{w}"], st[f"x_bip_r{w}"])
        st[f"sc_xwoba_r{w}"] = safe_div(st[f"x_xwoba_sum_r{w}"], st[f"x_bip_r{w}"])
        st[f"sc_whiff_r{w}"] = safe_div(st[f"x_whiffs_r{w}"], st[f"x_swings_r{w}"])
    st["sc_xba_r30"] = safe_div(st["x_xba_sum_r30"], st["x_bip_r30"])
    st["sc_la_r30"] = safe_div(st["x_la_sum_r30"], st["x_bip_r30"])
    st["sc_chase_r30"] = safe_div(st["x_chases_r30"], st["x_out_zone_pitches_r30"])
    st["sc_ev_max_r15"] = df.groupby("player_id", sort=False)["ev_max"].rolling(15, min_periods=1).max() \
        .reset_index(level=0, drop=True).sort_index()
    st["sc_form_ev"] = st["sc_ev_r5"] - st["sc_ev_r60"]
    st["sc_form_barrel"] = st["sc_barrel_r15"] - st["sc_barrel_r60"]
    st["sc_games"] = (df.groupby("player_id", sort=False).cumcount() + 1).clip(upper=100).astype(float)
    st = st.drop(columns=[c for c in st.columns if c.startswith("x_")])
    st = st.sort_values(["date", "player_id"]).reset_index(drop=True)
    return _collapse_same_day(st, "player_id")


def statcast_pitcher_state(sc: pd.DataFrame) -> pd.DataFrame:
    """Rolling velocity, whiff / CSW and contact-allowed rates per pitcher (prefix ``scp_``)."""
    if sc is None or sc.empty:
        return pd.DataFrame(columns=["player_id", "date"])
    df = sc.sort_values(["player_id", "date", "game_pk"]).reset_index(drop=True)
    cols = ["pitches", "fastballs", "fb_velo_sum", "spin_sum", "swings", "whiffs", "called_strikes", "chases",
            "out_zone_pitches", "bip", "ev_sum", "hard_hit", "barrels", "xwoba_sum"]
    parts = [df[["player_id", "date"]]]
    for w in SC_PIT_WINDOWS:
        parts.append(_rolling_inclusive(df, "player_id", cols, w, "x_"))
    st = pd.concat(parts, axis=1)
    for w in SC_PIT_WINDOWS:
        st[f"scp_velo_r{w}"] = safe_div(st[f"x_fb_velo_sum_r{w}"], st[f"x_fastballs_r{w}"])
        st[f"scp_whiff_r{w}"] = safe_div(st[f"x_whiffs_r{w}"], st[f"x_swings_r{w}"])
        st[f"scp_csw_r{w}"] = safe_div(st[f"x_called_strikes_r{w}"] + st[f"x_whiffs_r{w}"], st[f"x_pitches_r{w}"])
        st[f"scp_ev_allowed_r{w}"] = safe_div(st[f"x_ev_sum_r{w}"], st[f"x_bip_r{w}"])
        st[f"scp_barrel_allowed_r{w}"] = safe_div(st[f"x_barrels_r{w}"], st[f"x_bip_r{w}"])
        st[f"scp_xwoba_allowed_r{w}"] = safe_div(st[f"x_xwoba_sum_r{w}"], st[f"x_bip_r{w}"])
    st["scp_hard_hit_allowed_r30"] = safe_div(st["x_hard_hit_r30"], st["x_bip_r30"])
    st["scp_chase_r30"] = safe_div(st["x_chases_r30"], st["x_out_zone_pitches_r30"])
    st["scp_spin_r10"] = safe_div(st["x_spin_sum_r10"], st["x_pitches_r10"])
    st["scp_velo_last"] = safe_div(df["fb_velo_sum"], df["fastballs"]).to_numpy()
    st["scp_velo_max_last"] = df["fb_velo_max"].to_numpy()
    st["scp_velo_delta"] = st["scp_velo_r3"] - st["scp_velo_r30"]      # fatigue / injury signal
    st["scp_velo_delta_last"] = st["scp_velo_last"] - st["scp_velo_r30"]
    st["scp_form_whiff"] = st["scp_whiff_r3"] - st["scp_whiff_r30"]
    st["scp_games"] = (df.groupby("player_id", sort=False).cumcount() + 1).clip(upper=100).astype(float)
    st = st.drop(columns=[c for c in st.columns if c.startswith("x_")])
    st = st.sort_values(["date", "player_id"]).reset_index(drop=True)
    return _collapse_same_day(st, "player_id")


# ---------------------------------------------------------------- umpires
def umpire_state(games: pd.DataFrame, pitching_lines: pd.DataFrame) -> pd.DataFrame:
    """Rolling strikeout / walk rates in the games each home-plate umpire has worked (prefix ``ump_``)."""
    if games is None or "hp_umpire_id" not in games.columns or games["hp_umpire_id"].notna().sum() == 0:
        return pd.DataFrame(columns=["hp_umpire_id", "date"])
    per_game = pitching_lines.groupby("game_pk", sort=False)[["so", "bb", "bf", "er", "outs"]].sum().reset_index()
    g = games.dropna(subset=["hp_umpire_id"])[["game_pk", "date", "hp_umpire_id"]].merge(per_game, on="game_pk")
    if g.empty:
        return pd.DataFrame(columns=["hp_umpire_id", "date"])
    g["hp_umpire_id"] = g["hp_umpire_id"].astype("int64")
    g = g.sort_values(["hp_umpire_id", "date", "game_pk"]).reset_index(drop=True)
    parts = [g[["hp_umpire_id", "date"]]]
    for w in (30, 100):
        parts.append(_rolling_inclusive(g, "hp_umpire_id", ["so", "bb", "bf", "er", "outs"], w, "x_"))
    st = pd.concat(parts, axis=1)
    lg_k = g["so"].sum() / g["bf"].sum()
    lg_bb = g["bb"].sum() / g["bf"].sum()
    for w in (30, 100):
        st[f"ump_k_rate_r{w}"] = safe_div(st[f"x_so_r{w}"], st[f"x_bf_r{w}"]) - lg_k
        st[f"ump_bb_rate_r{w}"] = safe_div(st[f"x_bb_r{w}"], st[f"x_bf_r{w}"]) - lg_bb
    st["ump_runs_r100"] = safe_div(st["x_er_r100"] * 54.0, st["x_outs_r100"])
    st["ump_games"] = (g.groupby("hp_umpire_id", sort=False).cumcount() + 1).clip(upper=200).astype(float)
    st = st.drop(columns=[c for c in st.columns if c.startswith("x_")])
    st = st.sort_values(["date", "hp_umpire_id"]).reset_index(drop=True)
    return _collapse_same_day(st, "hp_umpire_id")


# ---------------------------------------------------------------- bullpen
def bullpen_usage(games: pd.DataFrame, pitching_lines: pd.DataFrame) -> pd.DataFrame:
    """Per (team, game_pk): bullpen workload in that game and back-to-back usage (for team_state)."""
    rp = pitching_lines[pitching_lines["is_starter"] == 0]
    if rp.empty:
        return pd.DataFrame(columns=["team_id", "game_pk", "bp_pitches", "bp_arms", "bp_b2b", "bp_top3_used"])
    log = pd.concat([
        games[["game_pk", "date", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
        games[["game_pk", "date", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
    ], ignore_index=True).sort_values(["team_id", "date", "game_pk"]).reset_index(drop=True)
    usage = rp.groupby(["team_id", "game_pk"], sort=False).agg(
        bp_pitches=("pitches", "sum"), bp_arms=("player_id", "count"),
        arms=("player_id", lambda s: frozenset(s.astype(int)))).reset_index()
    log = log.merge(usage, on=["team_id", "game_pk"], how="left")
    log["bp_pitches"] = log["bp_pitches"].fillna(0.0)
    log["bp_arms"] = log["bp_arms"].fillna(0.0)
    log["arms"] = log["arms"].apply(lambda x: x if isinstance(x, frozenset) else frozenset())
    prev_arms = log.groupby("team_id", sort=False)["arms"].shift(1)
    prev_date = log.groupby("team_id", sort=False)["date"].shift(1)
    consecutive = (log["date"] - prev_date).dt.days == 1
    log["bp_b2b"] = [
        float(len(a & b)) if (c and isinstance(b, frozenset)) else 0.0
        for a, b, c in zip(log["arms"], prev_arms, consecutive)]
    # "top 3" relievers = most appearances over the team's previous 30 games; how many pitched today
    top_used = np.zeros(len(log))
    for tid, idx in log.groupby("team_id", sort=False).indices.items():
        history: list[frozenset] = []
        for k, i in enumerate(idx):
            if history:
                counts: dict[int, int] = {}
                for arms in history[-30:]:
                    for pid in arms:
                        counts[pid] = counts.get(pid, 0) + 1
                top3 = {pid for pid, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]}
                top_used[i] = float(len(top3 & log.at[i, "arms"]))
            history.append(log.at[i, "arms"])
    log["bp_top3_used"] = top_used
    return log[["team_id", "game_pk", "bp_pitches", "bp_arms", "bp_b2b", "bp_top3_used"]]


# ---------------------------------------------------------------- as-of
def asof_join(left: pd.DataFrame, state: pd.DataFrame, left_key: str, state_key: str,
              prefix: str = "") -> pd.DataFrame:
    """Attach the latest state row with ``state.date < left.date`` for each left row.

    ``left`` may contain NaN keys (unknown starting pitcher); those rows get NaN features.
    Row order of ``left`` is preserved.
    """
    left = left.copy()
    left["_order"] = np.arange(len(left))
    left["_key"] = pd.to_numeric(left[left_key], errors="coerce").fillna(-1).astype("int64")
    feat_cols = [c for c in state.columns if c not in (state_key, "date", "_key")]
    if state.empty:
        for c in feat_cols:
            left[f"{prefix}{c}"] = np.nan
        return left.drop(columns=["_order", "_key"])
    st = state.copy()
    st["_key"] = st[state_key].astype("int64")
    if prefix:
        st = st.rename(columns={c: f"{prefix}{c}" for c in feat_cols})
        feat_cols = [f"{prefix}{c}" for c in feat_cols]
    left_sorted = left.sort_values("date", kind="mergesort")
    st_sorted = st.sort_values("date", kind="mergesort")
    merged = pd.merge_asof(
        left_sorted, st_sorted[["_key", "date"] + feat_cols], on="date", by="_key",
        allow_exact_matches=False, direction="backward",
    )
    merged = merged.sort_values("_order").drop(columns=["_order", "_key"]).reset_index(drop=True)
    return merged
