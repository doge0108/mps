"""Backtest: train on earlier seasons, evaluate on the most recent one, compare with baselines."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error, mean_poisson_deviance,
                             roc_auc_score)

from ..config import BATTING_TARGETS, PITCHING_TARGETS
from ..data.store import Dataset
from ..features.build import (States, build_batter_training, build_game_training,
                              build_pitcher_training, feature_columns)
from .game_model import make_game_model, summarise_game_predictions
from .player_model import make_batter_model, make_pitcher_model


def _split_by_season(feats: pd.DataFrame, ds: Dataset, test_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = ds.games.set_index("game_pk")["season"]
    s = feats["game_pk"].map(seasons).astype("Int64")
    return feats[s < test_season].reset_index(drop=True), feats[s == test_season].reset_index(drop=True)


def _mae_table(test: pd.DataFrame, pred: pd.DataFrame, targets: list[str], baseline_prefix: str,
               window: int) -> dict:
    rows = {}
    for t in targets:
        y = test[f"y_{t}"].to_numpy(dtype=float)
        base_col = f"{baseline_prefix}{t}_r{window}"
        base = test[base_col].fillna(np.nanmean(y)).to_numpy() if base_col in test else np.full_like(y, y.mean())
        mean_base = np.full_like(y, y.mean())
        model = np.clip(pred[t].to_numpy(), 1e-6, None)
        base = np.clip(base, 1e-6, None)
        rows[t] = {
            "mae_model": float(mean_absolute_error(y, model)),
            "mae_rolling_baseline": float(mean_absolute_error(y, base)),
            "mae_mean_baseline": float(mean_absolute_error(y, mean_base)),
            "rmse_model": float(np.sqrt(np.mean((y - model) ** 2))),
            "rmse_rolling_baseline": float(np.sqrt(np.mean((y - base) ** 2))),
            "rmse_mean_baseline": float(np.sqrt(np.mean((y - mean_base) ** 2))),
            "poisson_dev_model": float(mean_poisson_deviance(y, model)),
            "poisson_dev_rolling_baseline": float(mean_poisson_deviance(y, base)),
            "poisson_dev_mean_baseline": float(mean_poisson_deviance(y, mean_base)),
            "mean_actual": float(y.mean()),
            "mean_pred": float(np.mean(model)),
        }
    return rows


def backtest(ds: Dataset, test_season: int | None = None, max_rounds: int = 1500) -> dict:
    seasons = ds.seasons()
    if len(seasons) < 2:
        raise ValueError("backtest needs at least two seasons of data")
    test_season = test_season or seasons[-1]
    states = States.from_dataset(ds)
    report: dict = {"test_season": test_season, "train_seasons": [s for s in seasons if s < test_season]}

    # ---- batters
    bf = build_batter_training(ds, states)
    bf = bf[bf["b_games"].notna()]  # need some history to be a fair comparison
    cols = feature_columns(bf)
    train, test = _split_by_season(bf, ds, test_season)
    model = make_batter_model(cols, max_rounds=max_rounds).fit(train)
    pred = model.predict(test)
    report["batters"] = {"n_test": int(len(test)), "stats": _mae_table(test, pred, BATTING_TARGETS, "b_", 30)}

    # ---- pitchers
    pf = build_pitcher_training(ds, states)
    pf = pf[pf["p_apps"].notna()]
    cols = feature_columns(pf)
    train, test = _split_by_season(pf, ds, test_season)
    model = make_pitcher_model(cols, max_rounds=max_rounds).fit(train)
    pred = model.predict(test)
    report["pitchers"] = {"n_test": int(len(test)), "stats": _mae_table(test, pred, PITCHING_TARGETS, "p_", 10)}

    # ---- games
    gf = build_game_training(ds, states)
    cols = feature_columns(gf)
    train, test = _split_by_season(gf, ds, test_season)
    model = make_game_model(cols, max_rounds=max_rounds).fit(train)
    pred = summarise_game_predictions(model.predict(test))
    y = test["y_home_win"].to_numpy()
    p = pred["home_win_prob"].clip(1e-4, 1 - 1e-4).to_numpy()
    elo_p = test["elo_home_prob"].fillna(0.54).clip(1e-4, 1 - 1e-4).to_numpy()
    home_rate = float(train["y_home_win"].mean())
    report["games"] = {
        "n_test": int(len(test)),
        "accuracy_model": float(((p >= 0.5) == y).mean()),
        "accuracy_elo": float(((elo_p >= 0.5) == y).mean()),
        "accuracy_always_home": float(y.mean()),
        "log_loss_model": float(log_loss(y, p)),
        "log_loss_elo": float(log_loss(y, elo_p)),
        "log_loss_home_rate": float(log_loss(y, np.full_like(p, home_rate))),
        "brier_model": float(brier_score_loss(y, p)),
        "brier_elo": float(brier_score_loss(y, elo_p)),
        "auc_model": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "mae_total_runs_model": float(mean_absolute_error(test["y_total_runs"], pred["total_runs"])),
        "mae_total_runs_mean": float(mean_absolute_error(
            test["y_total_runs"], np.full(len(test), train["y_total_runs"].mean()))),
    }
    return report


def format_report(report: dict) -> str:
    lines = [f"Backtest: train {report['train_seasons']} -> test {report['test_season']}", ""]
    for section in ("batters", "pitchers"):
        sec = report[section]
        lines.append(f"{section.title()} ({sec['n_test']} player-games)")
        lines.append(f"  {'stat':>5} | {'Poisson deviance (model/rolling/mean)':^38} | "
                     f"{'RMSE (model/rolling/mean)':^30} | actual vs pred mean")
        for stat, m in sec["stats"].items():
            lines.append(
                f"  {stat:>5} | {m['poisson_dev_model']:.4f} / {m['poisson_dev_rolling_baseline']:.4f} / "
                f"{m['poisson_dev_mean_baseline']:.4f}{'':>10} | {m['rmse_model']:.4f} / "
                f"{m['rmse_rolling_baseline']:.4f} / {m['rmse_mean_baseline']:.4f}{'':>4} | "
                f"{m['mean_actual']:.3f} vs {m['mean_pred']:.3f}")
        lines.append("")
    g = report["games"]
    lines += [
        f"Games ({g['n_test']} games)",
        f"  accuracy : model {g['accuracy_model']:.3f} | elo {g['accuracy_elo']:.3f} | always-home {g['accuracy_always_home']:.3f}",
        f"  log loss : model {g['log_loss_model']:.4f} | elo {g['log_loss_elo']:.4f} | home-rate {g['log_loss_home_rate']:.4f}",
        f"  brier    : model {g['brier_model']:.4f} | elo {g['brier_elo']:.4f}",
        f"  auc      : model {g['auc_model']}",
        f"  total runs MAE: model {g['mae_total_runs_model']:.3f} | mean {g['mae_total_runs_mean']:.3f}",
    ]
    return "\n".join(lines)


def save_report(report: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        json.dump(report, fh, indent=2)
