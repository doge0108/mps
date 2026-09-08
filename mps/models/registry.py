"""Train / persist / load the three model bundles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.store import Dataset
from ..features.build import (States, build_batter_training, build_game_training,
                              build_pitcher_training, feature_columns)
from .base import MultiTargetBooster
from .game_model import make_game_model
from .player_model import make_batter_model, make_pitcher_model

FILES = {"batter": "batter_model.joblib", "pitcher": "pitcher_model.joblib", "game": "game_model.joblib"}


@dataclass
class ModelBundle:
    batter: MultiTargetBooster
    pitcher: MultiTargetBooster
    game: MultiTargetBooster

    def save(self, models_dir: Path) -> None:
        models_dir = Path(models_dir)
        for name, fname in FILES.items():
            getattr(self, name).save(models_dir / fname)

    @classmethod
    def load(cls, models_dir: Path) -> "ModelBundle":
        models_dir = Path(models_dir)
        missing = [f for f in FILES.values() if not (models_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing model files in {models_dir}: {missing}. Run `mps train` first.")
        return cls(**{name: MultiTargetBooster.load(models_dir / fname) for name, fname in FILES.items()})


def train_all(ds: Dataset, max_rounds: int = 1500, verbose: bool = True, stack: bool = True) -> ModelBundle:
    from .stacking import attach_stack, oof_stack_features
    states = States.from_dataset(ds)
    bf = build_batter_training(ds, states)
    bf = bf[bf["b_games"].notna()].reset_index(drop=True)
    batter = make_batter_model(feature_columns(bf), max_rounds=max_rounds).fit(bf)
    if verbose:
        print(f"batter model: {len(bf)} rows, best iterations {batter.best_iters}")
    pf = build_pitcher_training(ds, states)
    pf = pf[pf["p_apps"].notna()].reset_index(drop=True)
    pitcher = make_pitcher_model(feature_columns(pf), max_rounds=max_rounds).fit(pf)
    if verbose:
        print(f"pitcher model: {len(pf)} rows, best iterations {pitcher.best_iters}")
    gf = build_game_training(ds, states)
    if stack:
        if verbose:
            print("cross-fitting player models for bottom-up game features ...")
        gf = attach_stack(gf, oof_stack_features(bf, pf, ds.played_games(), max_rounds=min(max_rounds, 400),
                                                 verbose=verbose))
    game = make_game_model(feature_columns(gf), max_rounds=max_rounds).fit(gf)
    if verbose:
        print(f"game model: {len(gf)} rows, best iterations {game.best_iters}")
    return ModelBundle(batter=batter, pitcher=pitcher, game=game)
