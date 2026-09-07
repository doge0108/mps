# mps — MLB Prediction System

Machine-learning pipeline that learns from MLB box scores and predicts, for any
game you ask about:

* **Player stat lines** – expected hits, home runs, RBI, runs, walks, strikeouts,
  stolen bases, total bases and at-bats for batters; strikeouts, earned runs,
  innings, hits, walks and home runs allowed for starting pitchers – plus
  probabilities such as *P(at least one hit)*, *P(home run)*, *P(quality start)*.
* **Game results** – home/away win probability, expected runs for each side,
  expected total and the predicted winner.
* **Context the model actually sees** – current form (hot/cold streaks over the
  last 3–5 games vs. a 30-game baseline), announced lineups and batting-order
  slot, platoon splits (batter side vs. starter hand, with the batter's own
  rolling split against that hand), opposing lineup strength and handedness
  mix, park factor, game-time weather (temperature, wind in/out, dome,
  day/night), **Statcast quality of contact** (exit velocity, hard-hit and
  barrel rates, expected BA / wOBA, whiff and chase rates; pitcher velocity
  trend, CSW, contact allowed), **Marcel-style preseason projections with
  age**, the **home-plate umpire's** strikeout / walk tendencies, and
  **bullpen fatigue** (arms and pitches used in recent games, back-to-back
  appearances, top relievers used yesterday).

Models are LightGBM gradient-boosted trees (Poisson objectives for counts,
logistic for win probability) trained on leak-free rolling features.

## Quick start

```bash
pip install -e ".[dev]"

# Option A: real data from the free MLB Stats API + Baseball Savant (no keys).
# An in-progress season is fine: you get everything played so far.
# More history = better career rates and projections; 2018 onward is a good start.
mps fetch 2018-2026

# Option B: no network? generate realistic simulated seasons instead
mps simulate 2022 2023 2024

mps train                      # fits batter, pitcher and game models -> models/
mps evaluate                   # backtest on the most recent season vs baselines

mps predict-player "Aaron Judge" --date 2026-09-08
mps predict-player "Tarik Skubal" --date 2026-09-08 --json
mps predict-game --date 2026-09-08                 # every game that day
mps predict-game --date 2026-09-08 --home NYY --away BOS
mps players skubal             # find players / ids in the dataset
mps info
```

### Keeping up with the current season

```bash
mps update            # new box scores since the last stored game + next 7 days of schedule
mps train             # refresh the models (a few minutes)
mps predict-game      # defaults to today
```

`mps update` is incremental and safe to run every day.  It stores upcoming
games with their probable pitchers, **announced lineups** (as soon as MLB posts
them, usually a few hours before first pitch), the **home-plate umpire** and the
**weather forecast** from the live feed, plus new Statcast data, so predictions
for today's games use the real matchup.  Once a game finishes, the next update
replaces the placeholder with its box score.

For a game that is not stored yet, add `--live` to pull that day's schedule
from the API on the fly, or pass the matchup by hand with `--opponent BOS
[--away]`.

Example output:

```
Julio Tucker (NYY) vs BOS on 2024-09-30  [context: schedule]
  Weather: 88F, wind 14mph Out To CF, Clear
  Umpire: Ump Luis Betts -- K rate +0.012, BB rate -0.004 vs league over 81 games
  Opposing bullpen: 3 arms / 52 pitches last game, 141 pitches over last 3, 1 arms on back-to-back days, 2/3 top relievers used yesterday
  Opposing starter: Vlad Correa (LHP)
  Lineup: batting 4 (announced lineup)
  Platoon: bats R, platoon advantage, vs LHP last 40 g: AVG 0.193 SLG 0.340
  Age 27.3; preseason projection AVG 0.262 SLG 0.451 HR/PA 0.038 (reliability 0.71)
  Contact (Statcast): EV 88.9 mph (last 5 g 90.4, form +1.5), hard-hit 34%, barrel 5.1%, xwOBA 0.190, whiff 33%, chase 26%
  Opposing starter stuff: FB 92.8 mph (last start 92.1, -0.70 vs season), whiff 35%, CSW 37%, EV allowed 86.9, barrel allowed 2.7%
  Form: STEADY -- last 5 g: 3-for-19, 1 HR, 5 RBI, AVG 0.158 SLG 0.316 (30-g baseline AVG 0.162 SLG 0.252); hit streak 1, hitless streak 0, HR drought 4 g
  Expected line: H 0.85, HR 0.18, RBI 0.41, R 0.43, BB 0.25, SO 1.35, SB 0.02, TB 1.56, AB 3.82
  Probabilities: hit 57%, multi_hit 21%, home_run 16%, rbi 33%, run 35%, walk 22%, stolen_base 2%, strikeout 74%
```

```
      date home away       home_sp     away_sp  home_win_prob  away_win_prob  exp_home_runs  exp_away_runs  exp_total_runs predicted_winner   lineups                          weather
2024-10-06  NYY  BOS  Nolan Burnes Vlad Correa          0.555          0.445           3.93           4.01            7.94              NYY anno/anno 88F, wind 14mph Out To CF, Clear
```

The `lineups` column says where each side's nine came from: `anno` = announced
lineup, `prev` = the team's previous game (fallback).

## How it works

```
MLB Stats API / simulator
        │  games.csv, batting_lines.csv, pitching_lines.csv   (mps/data)
        ▼
state tables  (mps/features/states.py, mps/features/priors.py)
  batter_state   rolling 3/5/15/30/60-game averages, cumulative rates, hot/cold form
                 deltas (last 5 vs last 30), hit / hitless / HR-drought streaks, rest days
  split_state    rolling performance vs LHP and vs RHP separately (platoon splits)
  pitcher_state  rolling 3/5/10/30-appearance K%, BB%, HR%, ERA, form deltas, QS streak
  statcast       batters: EV, hard-hit, barrel, xBA, xwOBA, whiff, chase (5/15/30/60 g)
                 pitchers: fastball velocity + trend, whiff, CSW, EV / barrel / xwOBA allowed
  priors         Marcel projection per season (5/4/3 weights, regression, aging) + age
  umpire_state   home-plate umpire K / BB rates vs league over last 30/100 games
  team_state     rolling 5/10/30/162-game form, W/L streak, Elo, park factor, bullpen ERA,
                 bullpen workload (pitches / arms last game and last 3, back-to-back arms,
                 top-3 relievers used)
        │  as-of join: only states dated strictly BEFORE the game are used
        ▼
feature matrices (mps/features/build.py, mps/features/lineups.py)
  batter row  = own form + Statcast + prior + age + platoon edge & split vs starter hand
                + opposing starter (form, stuff, prior) + teams (incl. opp bullpen fatigue)
                + umpire + home/order + weather
  pitcher row = own form + stuff + prior + age + opposing lineup strength / handedness /
                xwOBA + teams + umpire + weather
  game row    = both teams' form/Elo/park/bullpen fatigue + both starters (form, stuff,
                prior) + both lineups + umpire + weather
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

| table            | one row per          | key columns                                                   |
|------------------|----------------------|---------------------------------------------------------------|
| `games`          | game                 | date, teams, score, starters, day/night, temp, wind, condition|
| `batting_lines`  | batter × game        | pa, ab, r, h, 2b, 3b, hr, rbi, bb, so, sb, hbp, tb, order     |
| `pitching_lines` | pitcher × game       | outs, h, r, er, bb, so, hr, batters faced, pitches            |
| `players`        | player               | name, bats, throws, position, current team, birth date        |
| `lineups`        | announced slot       | game, team, player, batting order (upcoming games only)       |
| `statcast_batting`  | batter × game     | pitches, swings, whiffs, chases, balls in play, EV sum/max, hard-hit, barrels, xBA/xwOBA sums |
| `statcast_pitching` | pitcher × game    | pitches, fastballs, velocity sum/max, spin, whiffs, called strikes, contact allowed |

Endpoints used: `schedule` (with `probablePitcher` and `lineups` hydration),
`game/{pk}/boxscore`, `game/{pk}/feed/live` (weather and officials only, via a
field filter), `sports/1/players` (handedness, birth date), and Baseball
Savant's `statcast_search/csv` (pitch level, fetched in 3-day windows and
reduced to per-game aggregates).  `--no-weather` and `--no-statcast` skip the
extra requests for a faster first download; `mps statcast 2018-2026` backfills
Statcast for stored seasons later.  The `games` table also stores the
home-plate umpire.

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
  data/statcast.py     Baseball Savant client + pitch -> per-game aggregation
  features/states.py   rolling state tables (players, teams, Statcast, umpires, bullpen) + as-of join
  features/priors.py   Marcel-style projections and age
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
  found for the date.  When no lineup is announced yet it uses the team's
  previous starting nine.
* Weather for upcoming games is whatever the live feed reports at update time;
  run `mps update` again closer to first pitch for fresher forecasts and lineups.
* Statcast data starts in 2015; seasons before that simply have NaN contact
  features.  Injuries are not modelled beyond what lineups and velocity trends reveal.
