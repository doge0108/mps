"""Command-line interface: ``mps <command> [options]``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR


def _add_dirs(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="where games/batting/pitching CSVs live")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="where trained models are stored")


def parse_seasons(tokens: list[str]) -> list[int]:
    """'2018-2021 2024' -> [2018, 2019, 2020, 2021, 2024]."""
    out: list[int] = []
    for tok in tokens:
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mps", description="MLB player-stat and game-outcome prediction")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download real seasons from the MLB Stats API (in-progress seasons OK)")
    p.add_argument("seasons", nargs="+", help="e.g. 2018-2026 or 2024 2025 2026")
    p.add_argument("--postseason", action="store_true", help="include postseason games")
    p.add_argument("--no-weather", action="store_true", help="skip the per-game weather/umpire request")
    p.add_argument("--no-statcast", action="store_true", help="skip Baseball Savant Statcast downloads")
    _add_dirs(p)

    p = sub.add_parser("update", help="incremental refresh: new results since last stored game + upcoming schedule")
    p.add_argument("--days-ahead", type=int, default=7, help="how many days of upcoming games to store")
    p.add_argument("--no-weather", action="store_true")
    p.add_argument("--no-statcast", action="store_true")
    _add_dirs(p)

    p = sub.add_parser("statcast", help="(re)download Statcast aggregates for stored seasons")
    p.add_argument("seasons", nargs="*", help="seasons to fetch (default: every stored season)")
    _add_dirs(p)

    p = sub.add_parser("simulate", help="generate simulated seasons (offline development / testing)")
    p.add_argument("seasons", nargs="+")
    p.add_argument("--games-per-team", type=int, default=162)
    p.add_argument("--seed", type=int, default=7)
    _add_dirs(p)

    p = sub.add_parser("train", help="train batter, pitcher and game models on all stored data")
    p.add_argument("--max-rounds", type=int, default=1500)
    _add_dirs(p)

    p = sub.add_parser("evaluate", help="backtest: train on earlier seasons, test on the latest")
    p.add_argument("--test-season", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=800)
    p.add_argument("--report", type=Path, default=None, help="write JSON report here")
    _add_dirs(p)

    p = sub.add_parser("predict-player", help="predict a player's stat line for a game")
    p.add_argument("player", help="player name (fuzzy) or MLB player id")
    p.add_argument("--date", default=None, help="game date YYYY-MM-DD (default: today)")
    p.add_argument("--opponent", default=None, help="opponent team if the game is not in the stored schedule")
    p.add_argument("--away", action="store_true", help="player's team is the away team (with --opponent)")
    p.add_argument("--order", type=int, default=None, help="batting order slot 1-9")
    p.add_argument("--live", action="store_true", help="look up that day's schedule/probable pitchers from the MLB API")
    p.add_argument("--json", action="store_true")
    _add_dirs(p)

    p = sub.add_parser("predict-game", help="predict game outcomes for a date or matchup")
    p.add_argument("--date", default=None, help="game date YYYY-MM-DD (default: today)")
    p.add_argument("--home", default=None)
    p.add_argument("--away", default=None)
    p.add_argument("--live", action="store_true", help="look up that day's schedule/probable pitchers from the MLB API")
    p.add_argument("--json", action="store_true")
    _add_dirs(p)

    p = sub.add_parser("players", help="search players in the dataset")
    p.add_argument("query")
    _add_dirs(p)

    p = sub.add_parser("info", help="summarise the stored dataset")
    _add_dirs(p)
    return parser


def _today() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def _live_schedule(date: str) -> pd.DataFrame | None:
    from .data.ingest import fetch_schedule
    try:
        sched = fetch_schedule(date)
        return sched if len(sched) else None
    except Exception as exc:  # network failure -> fall back to stored data
        print(f"warning: could not fetch live schedule ({exc}); using stored games", file=sys.stderr)
        return None


def cmd_fetch(args) -> int:
    from .data.ingest import fetch_seasons
    types = ("R", "P") if args.postseason else ("R",)
    ds = fetch_seasons(parse_seasons(args.seasons), args.data_dir, game_types=types, weather=not args.no_weather,
                       statcast=not args.no_statcast)
    print(f"Saved {len(ds.games)} games / {len(ds.batting_lines)} batting / {len(ds.pitching_lines)} pitching lines "
          f"to {args.data_dir}")
    return 0


def cmd_update(args) -> int:
    from .data.ingest import update
    ds = update(args.data_dir, days_ahead=args.days_ahead, weather=not args.no_weather, statcast=not args.no_statcast)
    up = ds.upcoming_games()
    print(f"Dataset now has {len(ds.played_games())} completed games (latest {ds.played_games()['date'].max().date()}) "
          f"and {len(up)} upcoming games -> {args.data_dir}")
    print("Re-run `mps train` to refresh the models with the new games.")
    return 0


def cmd_statcast(args) -> int:
    from .data.ingest import fetch_statcast
    from .data.store import Dataset, normalise
    ds = Dataset.load(args.data_dir)
    seasons = parse_seasons(args.seasons) if args.seasons else ds.seasons()
    played = ds.played_games()
    for season in seasons:
        g = played[played["season"] == season]
        if g.empty:
            print(f"no stored games for {season}; run `mps fetch {season}` first")
            continue
        bat, pit = fetch_statcast(g["date"].min().date(), g["date"].max().date(), args.data_dir)
        add = Dataset(games=ds.games.iloc[:0], batting_lines=ds.batting_lines.iloc[:0],
                      pitching_lines=ds.pitching_lines.iloc[:0], statcast_batting=bat, statcast_pitching=pit)
        ds = ds.concat(add)
    ds.save(args.data_dir)
    print(f"Statcast rows stored: {len(ds.statcast_batting)} batter-games, {len(ds.statcast_pitching)} pitcher-games")
    return 0


def cmd_simulate(args) -> int:
    from .data.simulate import simulate_dataset
    from .data.store import Dataset
    ds = simulate_dataset(parse_seasons(args.seasons), args.games_per_team, seed=args.seed)
    if (args.data_dir / "games.csv").exists():
        ds = Dataset.load(args.data_dir).concat(ds)
    ds.save(args.data_dir)
    print(f"Simulated seasons {parse_seasons(args.seasons)}: {len(ds.games)} games, {len(ds.batting_lines)} batting lines, "
          f"{len(ds.pitching_lines)} pitching lines -> {args.data_dir}")
    return 0


def cmd_train(args) -> int:
    from .data.store import Dataset
    from .models.registry import train_all
    ds = Dataset.load(args.data_dir)
    print(f"Training on seasons {ds.seasons()} ({len(ds.games)} games)")
    bundle = train_all(ds, max_rounds=args.max_rounds)
    bundle.save(args.models_dir)
    print(f"Saved models to {args.models_dir}")
    return 0


def cmd_evaluate(args) -> int:
    from .data.store import Dataset
    from .models.evaluate import backtest, format_report, save_report
    ds = Dataset.load(args.data_dir)
    report = backtest(ds, args.test_season, max_rounds=args.max_rounds)
    print(format_report(report))
    if args.report:
        save_report(report, args.report)
        print(f"\nReport written to {args.report}")
    return 0


def _predictor(args):
    from .predict import Predictor
    return Predictor.load(args.data_dir, args.models_dir)


def cmd_predict_player(args) -> int:
    pred = _predictor(args)
    date = args.date or _today()
    schedule = _live_schedule(date) if args.live else None
    result = pred.predict_player(args.player, date, opponent=args.opponent,
                                 is_home=None if args.opponent is None else int(not args.away),
                                 batting_order=args.order, schedule=schedule)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    where = "vs" if result["is_home"] else "@"
    opp = result["opponent"] or "unknown opponent"
    print(f"{result['player']} ({result['team']}) {where} {opp} on {result['date']}  [context: {result['context_source']}]")
    if result.get("weather"):
        from .predict import _weather_text
        print(f"  Weather: {_weather_text(result['weather'])}")
    if result["context_source"] == "schedule" and not result.get("umpire"):
        print("  Umpire: not announced yet (run `mps update` closer to first pitch)")
    if result.get("umpire"):
        u = result["umpire"]
        tend = ""
        if u.get("k_rate_vs_league") is not None:
            tend = (f" -- K rate {u['k_rate_vs_league']:+.3f}, BB rate {u['bb_rate_vs_league']:+.3f} vs league "
                    f"over {u['games']} games")
        print(f"  Umpire: {u['name'] or u['id']}{tend}")
    if result.get("opp_bullpen"):
        bp = result["opp_bullpen"]
        print(f"  Opposing bullpen: {bp['arms_used_last_game']} arms / {bp['pitches_last_game']} pitches last game, "
              f"{bp['pitches_last_3']} pitches over last 3, {bp['back_to_back_arms']} arms on back-to-back days, "
              f"{bp['top3_used_last_game']}/3 top relievers used yesterday")
    if "batting" in result:
        b = result["batting"]
        hand = f" ({b['opposing_starter_hand']}HP)" if b["opposing_starter_hand"] else ""
        print(f"  Opposing starter: {b['opposing_starter'] or 'unknown'}{hand}")
        lineup = b["lineup"]
        slot = f"batting {b['batting_order']}"
        if lineup["source"] == "announced":
            slot += " (announced lineup)" if lineup["in_lineup"] else " -- NOT in the announced lineup"
        else:
            slot += " (based on last game)"
        print(f"  Lineup: {slot}")
        if b["bats"]:
            adv = {True: "platoon advantage", False: "platoon disadvantage", None: ""}[b["platoon_advantage"]]
            split = b["split_vs_hand"]
            split_txt = (f", vs {split['vs_hand']}HP last {split['games']} g: AVG {split['avg']:.3f} SLG {split['slg']:.3f}"
                         if split and split["avg"] is not None else "")
            print(f"  Platoon: bats {b['bats']}{', ' + adv if adv else ''}{split_txt}")
        if result.get("age"):
            pr = b.get("prior")
            prior_txt = (f"; preseason projection AVG {pr['avg']:.3f} SLG {pr['slg']:.3f} HR/PA {pr['hr_rate']:.3f} "
                         f"(reliability {pr['reliability']:.2f})") if pr else ""
            print(f"  Age {result['age']}{prior_txt}")
        c = b.get("contact")
        if c and c["ev_r30"] is not None:
            print(f"  Contact (Statcast): EV {c['ev_r30']} mph (last 5 g {c['ev_r5']}, form {c['ev_form']:+.1f}), "
                  f"hard-hit {c['hard_hit_r30']:.0%}, barrel {c['barrel_r30']:.1%}, xwOBA {c['xwoba_r30']:.3f}, "
                  f"whiff {c['whiff_r30']:.0%}, chase {c['chase_r30']:.0%}")
        st = b.get("opposing_starter_stuff")
        if st and st["velo_r10"] is not None:
            print(f"  Opposing starter stuff: FB {st['velo_r10']} mph (last start {st['velo_last']}, "
                  f"{st['velo_delta_vs_r30']:+.2f} vs season), whiff {st['whiff_r10']:.0%}, CSW {st['csw_r10']:.0%}, "
                  f"EV allowed {st['ev_allowed_r30']}, barrel allowed {st['barrel_allowed_r30']:.1%}")
        f = b["form"]
        if f:
            l5 = f["last5"]
            print(f"  Form: {f['label'].upper()} -- last {l5['games']} g: {l5['h']}-for-{l5['ab']}, {l5['hr']} HR, "
                  f"{l5['rbi']} RBI, AVG {l5['avg'] or 0:.3f} SLG {l5['slg'] or 0:.3f} "
                  f"(30-g baseline AVG {f['baseline30']['avg'] or 0:.3f} SLG {f['baseline30']['slg'] or 0:.3f}); "
                  f"hit streak {f['hit_streak']}, hitless streak {f['hitless_streak']}, HR drought {f['hr_drought']} g")
        print("  Expected line: " + ", ".join(f"{k.upper()} {v:.2f}" for k, v in b["expected"].items()))
        print("  Probabilities: " + ", ".join(f"{k} {v:.0%}" for k, v in b["probabilities"].items()))
    if "pitching" in result:
        p = result["pitching"]
        if not p.get("probable_starter", True):
            listed = p.get("listed_starter") or "another pitcher"
            print(f"  Pitching: NOT the probable starter for this game ({listed} is listed). "
                  f"Hypothetical line if they started:")
        f = p["form"]
        if f:
            print(f"  Pitching form: {f['label'].upper()} -- last 3 starts ERA {f['last3']['era']}, "
                  f"K% {f['last3']['k_rate']} (30-app baseline ERA {f['baseline30']['era']}, "
                  f"K% {f['baseline30']['k_rate']}); QS streak {f['quality_start_streak']}, rest {f['days_rest']} d")
        if result.get("age"):
            pr = p.get("prior")
            prior_txt = (f"; preseason projection ERA {pr['era']:.2f} K% {pr['k_rate']:.3f} BB% {pr['bb_rate']:.3f} "
                         f"(reliability {pr['reliability']:.2f})") if pr else ""
            print(f"  Age {result['age']}{prior_txt}")
        st = p.get("stuff")
        if st and st["velo_r10"] is not None:
            print(f"  Stuff (Statcast): FB {st['velo_r10']} mph (last start {st['velo_last']}, "
                  f"{st['velo_delta_vs_r30']:+.2f} vs season), whiff {st['whiff_r10']:.0%}, CSW {st['csw_r10']:.0%}, "
                  f"EV allowed {st['ev_allowed_r30']}, barrel allowed {st['barrel_allowed_r30']:.1%}, "
                  f"xwOBA allowed {st['xwoba_allowed_r30']:.3f}")
        if p.get("opposing_lineup_lhb_share") is not None:
            print(f"  Opposing lineup: {p['opposing_lineup_lhb_share']:.0%} left-handed bats")
        print(f"  Pitching, expected: IP {p['innings_pitched']}, " +
              ", ".join(f"{k.upper()} {v:.2f}" for k, v in p["expected"].items() if k != "outs"))
        print("  Probabilities: " + ", ".join(f"{k} {v:.0%}" for k, v in p["probabilities"].items()))
    if result["context_source"] == "unknown":
        print("  note: no game found for this date; run `mps update`, or pass --opponent / --live for matchup-aware predictions.")
    return 0


def cmd_predict_game(args) -> int:
    pred = _predictor(args)
    date = args.date or _today()
    schedule = _live_schedule(date) if args.live else None
    table = pred.predict_games(date, schedule=schedule, home=args.home, away=args.away)
    if args.json:
        print(table.to_json(orient="records", indent=2))
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(table.to_string(index=False))
    return 0


def cmd_players(args) -> int:
    pred_index = _predictor(args).search_players(args.query, n=15)
    if pred_index.empty:
        print("no matches")
    else:
        from .config import team_label
        out = pred_index.copy()
        out["team"] = out["team_id"].map(team_label)
        print(out[["name", "team", "last_seen", "is_batter", "is_pitcher"]].to_string())
    return 0


def cmd_info(args) -> int:
    from .data.store import Dataset
    ds = Dataset.load(args.data_dir)
    played, up = ds.played_games(), ds.upcoming_games()
    print(f"seasons: {ds.seasons()}")
    print(f"completed games: {len(played)}  ({played['date'].min().date()} .. {played['date'].max().date()})")
    if len(up):
        print(f"upcoming games: {len(up)}  ({up['date'].min().date()} .. {up['date'].max().date()}), "
              f"announced lineup slots: {len(ds.lineups)}")
    print(f"players with handedness: {int(ds.players['bats'].notna().sum()) if len(ds.players) else 0}, "
          f"with birth date: {int(ds.players['birth_date'].notna().sum()) if len(ds.players) else 0}")
    wx = played["temp_f"].notna().mean() if len(played) else 0
    ump = played["hp_umpire_id"].notna().mean() if len(played) else 0
    print(f"games with weather: {wx:.0%}, with home-plate umpire: {ump:.0%}")
    sc = ds.statcast_batting
    if len(sc):
        print(f"statcast: {len(sc)} batter-games, {len(ds.statcast_pitching)} pitcher-games "
              f"({sc['date'].min().date()} .. {sc['date'].max().date()})")
    else:
        print("statcast: none (run `mps statcast` to download quality-of-contact data)")
    print(f"batting lines: {len(ds.batting_lines)}  players: {ds.batting_lines['player_id'].nunique()}")
    print(f"pitching lines: {len(ds.pitching_lines)}  pitchers: {ds.pitching_lines['player_id'].nunique()}")
    return 0


COMMANDS = {
    "fetch": cmd_fetch, "update": cmd_update, "statcast": cmd_statcast, "simulate": cmd_simulate, "train": cmd_train, "evaluate": cmd_evaluate,
    "predict-player": cmd_predict_player, "predict-game": cmd_predict_game, "players": cmd_players,
    "info": cmd_info,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
