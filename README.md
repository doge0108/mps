# mps — MLB Prediction System

Machine-learning pipeline that learns from MLB box scores and predicts, for any
game you ask about:

* **Player stat lines** – expected hits, home runs, RBI, runs, walks, strikeouts,
  stolen bases, total bases and at-bats for batters; strikeouts, earned runs,
  innings, hits, walks and home runs allowed for starting pitchers – plus
  probabilities such as *P(at least one hit)*, *P(home run)*, *P(quality start)*.
* **Game results** – home/away win probability, expected runs for each side,
  expected total and the predicted winner.

Models are LightGBM gradient-boosted trees (Poisson objectives for counts,
logistic for win probability) trained on leak-free rolling features.

## Quick start

```bash
pip install -e ".[dev]"

# Option A: real data from the free MLB Stats API (no key required)
mps fetch 2023 2024 2025

# Option B: no network? generate realistic simulated seasons instead
mps simulate 2022 2023 2024

mps train                      # fits batter, pitcher and game models -> models/
mps evaluate                   # backtest on the most recent season vs baselines

mps predict-player "Aaron Judge" --date 2025-09-12
mps predict-player "Tarik Skubal" --date 2025-09-12 --json
mps predict-game --date 2025-09-12                 # every game that day
mps predict-game --date 2025-09-12 --home NYY --away BOS
mps players skubal             # find players / ids in the dataset
mps info
```

For a game that is not yet in the stored schedule, add `--live` to pull that
day's schedule and probable starters from the MLB API, or pass the matchup by
hand with `--opponent BOS [--away]`.

Example output:

```
Hunter Semien (PIT) @ SD on 2024-09-28  [context: schedule]
  Opposing starter: Julio Smith
  Expected line: H 1.34, HR 0.15, RBI 0.84, R 0.77, BB 0.47, SO 0.88, SB 0.01, TB 2.14, AB 4.40
  Probabilities: hit 74%, multi_hit 39%, home_run 14%, rbi 57%, run 54%, walk 37%, stolen_base 1%, strikeout 59%
```

```
      date home away      home_sp       away_sp  home_win_prob  away_win_prob  exp_home_runs  exp_away_runs  exp_total_runs predicted_winner
2024-10-02  LAD  NYY Carlos Alonso Julio Glasnow          0.743          0.257           5.56           3.01            8.57              LAD
```

## How it works

```
MLB Stats API / simulator
        │  games.csv, batting_lines.csv, pitching_lines.csv   (mps/data)
        ▼
state tables  (mps/features/states.py)
  batter_state   rolling 7/15/30/60-game averages, cumulative rates, rest days
  pitcher_state  rolling 5/10/30-appearance K%, BB%, HR%, ERA, outs per start
  team_state     rolling 10/30/162-game form, Elo, park factor, bullpen ERA
        │  as-of join: only states dated strictly BEFORE the game are used
        ▼
feature matrices (mps/features/build.py)
  batter row  = own form + opposing starter + own team + opposing team + home/order
  pitcher row = own form + own team + opposing offence
  game row    = both teams' form/Elo/park + both starters
        ▼
LightGBM boosters (mps/models)
  batter:  Poisson per stat (h, hr, rbi, r, bb, so, sb, tb, ab)
  pitcher: Poisson per stat (so, er, outs, h, bb, hr)
  game:    logistic home_win + Poisson home_runs / away_runs (Skellam blend)
```

Because features are built by joining the latest *pre-game* state, training
and live prediction share one code path: predicting tomorrow's game is just
looking up today's states.  There is no way for a game's own box score to leak
into its features, and `tests/test_features.py` checks this.

### Data

`mps fetch` downloads every regular-season game's box score through the
public MLB Stats API (`statsapi.mlb.com`), caches the raw JSON under
`data/raw/`, and writes three CSV tables:

| table            | one row per          | key columns                                        |
|------------------|----------------------|----------------------------------------------------|
| `games`          | game                 | date, teams, final score, starting pitchers        |
| `batting_lines`  | batter × game        | pa, ab, r, h, 2b, 3b, hr, rbi, bb, so, sb, hbp, tb |
| `pitching_lines` | pitcher × game       | outs, h, r, er, bb, so, hr, batters faced, pitches |

`mps simulate` produces the same tables from a plate-appearance level
simulator with latent batter/pitcher/park/team skills.  It is meant for
offline development and testing; train on real seasons for real predictions.

### Evaluation

`mps evaluate` trains on all but the last season and scores the last season
against two baselines (the player's own rolling average and the league mean).
Counts are scored with Poisson deviance and RMSE; win probabilities with
accuracy, log loss, Brier score and AUC against an Elo baseline.  On three
simulated seasons the models beat both baselines on every stat and reach
0.58 log loss on game winners (Elo 0.584, constant 0.693).

## Python API

```python
from mps.predict import Predictor

p = Predictor.load("data", "models")
p.predict_player("Aaron Judge", "2025-09-12")            # dict with expected line + probabilities
p.predict_games("2025-09-12")                            # DataFrame, one row per game
p.predict_games("2025-09-12", home="NYY", away="BOS")
```

## Layout

```
mps/
  config.py            paths, stat columns, team ids
  data/mlb_api.py      MLB Stats API client + box-score parsers
  data/ingest.py       season download -> CSV tables
  data/simulate.py     offline season simulator
  data/store.py        Dataset load/save/concat
  features/states.py   rolling state tables + as-of join
  features/build.py    feature assembly for batters / pitchers / games
  models/base.py       multi-target LightGBM wrapper (time-based early stopping)
  models/*_model.py    objectives and post-processing per model family
  models/evaluate.py   season backtest and report
  models/registry.py   train / save / load model bundle
  predict.py           Predictor: name lookup, schedule context, predictions
  cli.py               `mps` command line
tests/                 pytest suite (API parsing, simulator, features, end-to-end)
```

## Notes and limitations

* Predictions are only as good as the history available: run `mps fetch` for
  at least two full seasons, and re-run it (it is incremental) before
  predicting so recent games are included.
* When the opposing starter is unknown the model falls back to the pitcher due
  up in a five-man rotation, and to opponent-agnostic features if no game is
  found for the date.
* Weather, injuries, lineup announcements and platoon splits are not modelled
  yet; they are natural next additions to the feature builders.
