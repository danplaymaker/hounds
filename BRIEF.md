# UK Greyhound Value-Betting Model — Project Brief

## What we're building

A system that identifies value bets on UK greyhound races by estimating "true" win probabilities for each runner, comparing them to live Betfair Exchange prices in a window **5–20 minutes before the off**, and flagging bets where our edge exceeds a threshold.

**Why this window:** Pre-off greyhound markets are thinly traded until roughly T-5 minutes, then liquidity surges as bots and late money pile in. The strategy is to front-run that money — taking generous prices that we expect to shorten before the off.

**Two phases:**

- **Phase A (MVP):** A "fundamentals" model that predicts win probability from form, trap, race-shape, and connections data. Compares to current available back price. Bets when edge exceeds threshold. Backtested against Betfair Starting Price (BSP).
- **Phase B (Final):** Adds a second model that predicts BSP from the current order book plus fundamentals. Bets are placed only when the fundamentals model says there's value AND the price-movement model expects the price to shorten. Uses tick-level historical data and real order-book reconstruction.

Build Phase A end-to-end first. Do not start Phase B until Phase A is proven (paper-trades break-even or better against BSP over a meaningful sample).

---

## Hard constraints and non-negotiables

These are the constraints that, if violated, invalidate the project. Re-read them every time you start work on a new component.

1. **No look-ahead.** Every feature for a race at time T must be computable using only data with `timestamp < T`. The current race's result is never an input. Final going (which is computed after a meeting starts) is not available before the first race of the day.
2. **Walk-forward backtesting only.** Never random splits. Never k-fold on time series. Train on a strict past window, test on a strict future window, roll forward.
3. **Calibrated probabilities.** A 30% prediction must win 30% of the time on held-out data. Check with reliability diagrams. Apply isotonic or Platt scaling if needed.
4. **Backtest against achievable prices.** BSP is the bet price for Phase A. Phase B uses reconstructed order-book state with realistic fill assumptions, not the best price that ever traded.
5. **Account for commission.** Betfair charges 2–5% on winnings. ROI figures are post-commission or they're meaningless.
6. **Identity by ID, not name.** Dogs are identified by their GBGB ID (microchip / earmark). Never join on names. Normalisation is for matching across sources, not for primary keys.
7. **Reproducibility.** Every backtest must be deterministic given the same data and config. Seed every RNG. Version every model artifact.

---

## Tech stack

- **Language:** Python 3.11+
- **Data:** Polars (preferred) or pandas; Parquet for storage
- **Modelling:** scikit-learn for baselines and calibration; LightGBM for the production model
- **Scraping:** httpx (async), selectolax for HTML parsing, respect robots.txt, 2s minimum between requests to GBGB
- **Betfair:** `betfairlightweight` for API and historic data parsing
- **Orchestration:** Plain Python scripts + Makefile for Phase A. No Airflow.
- **Storage:** Local filesystem (Parquet) for Phase A. Postgres can come later if needed.
- **Testing:** pytest. Every feature function gets a leakage test.
- **Config:** YAML, loaded via Pydantic for typed validation.

Don't add infrastructure (Docker, cloud, databases, queues) until there's a concrete reason. This is a research project until it isn't.

---

## Repository layout

```
greyhound-value/
├── BRIEF.md                    # This file
├── README.md                   # Quickstart
├── pyproject.toml
├── Makefile
├── config/
│   ├── default.yaml            # Paths, dates, model hyperparams
│   └── tracks.yaml             # Track metadata (distances, going adjustments)
├── data/
│   ├── raw/                    # Scraped HTML, downloaded BSP CSVs (gitignored)
│   ├── interim/                # Parsed but not joined (gitignored)
│   └── processed/              # Joined, feature-engineered Parquet (gitignored)
├── src/greyhound/
│   ├── ingest/
│   │   ├── gbgb_scraper.py
│   │   ├── gbgb_parser.py
│   │   └── betfair_bsp.py
│   ├── data/
│   │   ├── identity.py         # Name normalisation, dog ID resolution
│   │   ├── joins.py            # Race-level joins across sources
│   │   └── schemas.py          # Pydantic / Polars schemas
│   ├── features/
│   │   ├── form.py             # Form-snapshot, calculated times
│   │   ├── trap.py             # Trap & track-distance features
│   │   ├── shape.py            # Race-shape & early-pace features
│   │   ├── connections.py      # Trainer/kennel features
│   │   └── pipeline.py         # Orchestrates feature build per race
│   ├── models/
│   │   ├── baseline_logit.py   # Conditional logit baseline
│   │   ├── lgbm_ranker.py      # LambdaRank model
│   │   ├── calibration.py      # Isotonic / Platt + reliability tools
│   │   └── inference.py        # Apply model to a single race
│   ├── betting/
│   │   ├── edge.py             # Probability → edge → bet/no-bet
│   │   ├── staking.py          # Fractional Kelly, caps
│   │   └── backtest.py         # Walk-forward simulator
│   └── live/                   # Empty in Phase A; populated in Phase B
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_eda.ipynb
│   ├── 03_model_diagnostics.ipynb
│   └── 04_backtest_review.ipynb
└── tests/
    ├── test_identity.py
    ├── test_features_no_leakage.py   # The most important test file
    ├── test_calibration.py
    └── test_backtest.py
```

---

## Phase A — MVP, in order

Work in this sequence. Do not skip ahead. Each milestone produces an artifact that the next depends on.

### A1. Data ingestion

**A1.1 GBGB results scraper** (`src/greyhound/ingest/gbgb_scraper.py`)

- Iterate dates from a configured start date (default: 24 months ago) to yesterday.
- For each date, fetch the results index, then each meeting, then each race.
- Cache raw HTML to `data/raw/gbgb/YYYY/MM/DD/<track>/<race_id>.html`. Re-runs must use cache.
- Polite scraping: 2s minimum between requests, exponential backoff on errors, configurable concurrency cap (default 1).
- Resumable: if interrupted, restart should pick up where it left off based on cache presence.

**A1.2 GBGB parser** (`src/greyhound/ingest/gbgb_parser.py`)

Pure function from HTML to typed rows. No I/O. One row per runner per race. Output schema (Parquet, in `data/interim/gbgb_runs.parquet`):

| field | type | notes |
|---|---|---|
| race_id | string | stable GBGB race ID |
| race_datetime | datetime | UTC |
| track | string | canonical name (see `config/tracks.yaml`) |
| distance_m | int | |
| grade | string | e.g. "A3", "OR", "S2" |
| going | float | published going for that meeting/race |
| dog_id | string | GBGB microchip / earmark; **primary key for dogs** |
| dog_name | string | as published |
| trap | int | 1–6 (occasionally 7–8 for handicaps) |
| sp | float | starting price (decimal) |
| finish_position | int | 1 = winner; null = withdrew or DNF |
| run_time | float | seconds |
| sectional_1 | float | seconds to first split; null if not published |
| weight_kg | float | |
| trainer_id | string | |
| trainer_name | string | |
| comment | string | free-text running line (e.g. "EP, Led 2") |
| bf_safe_name | string | derived: dog_name lowercased, apostrophes/dots stripped |

The `bf_safe_name` is for joining to Betfair. Compute it once here.

**A1.3 Betfair BSP ingestion** (`src/greyhound/ingest/betfair_bsp.py`)

- Download Betfair's free PROMO BSP CSVs for UK greyhounds (start with the free tier; paid ADVANCED tier is a Phase B concern).
- Parse to Parquet: one row per (race, runner) with BSP, win flag, matched volume if available.
- Output `data/interim/betfair_bsp.parquet`.

**A1.4 Join** (`src/greyhound/data/joins.py`)

Join GBGB runs to Betfair BSP. Join key: `(date, canonical_track, race_datetime ± 2min, trap)`. Verify with `bf_safe_name` match as a sanity check. Log unmatched rates by track — if any track is >5% unmatched, investigate before proceeding. Output `data/processed/races_with_market.parquet`.

**Milestone A1 done when:** You can run `make ingest` from a clean state and end up with `races_with_market.parquet` containing 18–24 months of UK greyhound races joined to BSP, with >95% join rate.

### A2. Feature engineering

**A2.1 The leakage-proof primitive.**

Build `get_form_snapshot(dog_id: str, as_of: datetime, runs_df) -> dict` in `src/greyhound/features/form.py`. This is the foundation of everything. It returns a dict of features computable from the dog's runs *strictly before* `as_of`. **Every test in `test_features_no_leakage.py` exercises this function with edge cases.**

Test cases that must pass:
- Dog with no prior runs: returns sentinel values (nulls or domain defaults), never crashes.
- Dog with exactly one prior run: returns features based on that one run.
- `as_of` equal to a race datetime: that race must NOT be in the snapshot.
- `as_of` one second after a race: that race must be in the snapshot.

**A2.2 Feature families** (in `src/greyhound/features/`):

*Form* (`form.py`):
- `calc_time_last_n` for n in {1, 3, 6}: mean going-adjusted time at this track/distance
- `calc_time_best_90d`: best calculated time at this track/distance in last 90 days
- `calc_time_trend_6`: OLS slope of last 6 calculated times (improving/declining)
- `calc_time_std_6`: consistency
- `runs_28d`: count of runs in last 28 days
- `days_since_last_run`
- `wins_at_track_dist`: count and rate (with Bayesian shrinkage to population mean)

*Trap & track* (`trap.py`):
- `trap` (categorical 1–8)
- `track_dist_trap_winrate`: long-run win rate of this trap at this track/distance (computed from training set only, not test set)

*Shape* (`shape.py`):
- `early_pace_score`: mean sectional-1 rank in last 6 runs (1 = led, 6 = last)
- `running_style`: derived from comment text patterns ("EP" → early, "RB" → rails-back, etc.) — start with regex, refine later
- Race-level interactions (these require knowing the field):
  - `n_other_early_pace_dogs`: count of other runners with early_pace_score < 2.5
  - `pace_conflict_score`: heuristic combining own style + draw + others' styles + their draws

*Connections* (`connections.py`):
- `trainer_strike_rate_30d`: trainer winners / runners last 30d, Bayesian-shrunk
- `trainer_strike_rate_track_180d`: same but at this track

**A2.3 Feature pipeline** (`features/pipeline.py`):

A function `build_features(races_df) -> features_df` that produces one row per (race, runner) with all features, plus the target `won` (binary). Must be deterministic, fully reproducible, and must NOT use any field that wouldn't be known before the off (no result-derived data on the row itself, no later races' data anywhere).

**Milestone A2 done when:** `make features` produces `data/processed/features.parquet`; `pytest tests/test_features_no_leakage.py` passes (≥10 tests covering the leakage traps).

### A3. Modelling

**A3.1 Baseline: conditional logit** (`models/baseline_logit.py`)

Per-race softmax over runners' linear scores. Trained with cross-entropy on the actual winner. Fast, hard to overfit, interpretable. This is the bar everything else has to beat.

**A3.2 Production: LightGBM with LambdaRank** (`models/lgbm_ranker.py`)

Group by race_id, use LambdaRank objective, convert per-race scores to probabilities via softmax with a learned temperature τ:

```
p_i = softmax(score_i / τ)
```

τ is fit on a validation slice to minimise NLL.

**A3.3 Calibration** (`models/calibration.py`)

After getting raw probabilities:
- Build reliability diagram: bucket predictions into 10 bins, plot predicted vs observed win rate.
- If mis-calibrated, fit isotonic regression on a held-out slice (chronologically after train, before test).
- Save the calibrator with the model.

Acceptance: in the held-out test set, the Brier score must be lower than (a) market-implied probability from BSP after de-overround removal and (b) a uniform 1/N prior.

**A3.4 Inference** (`models/inference.py`)

`predict_race(race_features) -> dict[runner_id, calibrated_prob]`. Probabilities must sum to 1.0 across the field within 1e-6.

**Milestone A3 done when:** `make train` produces a versioned model artifact in `models/artifacts/<timestamp>/`. Reliability diagram saved as PNG. Brier score logged. Probabilities sum to 1 in inference.

### A4. Betting logic and backtest

**A4.1 Edge calculation** (`betting/edge.py`)

```
edge = (model_prob * price) - 1
```

Configurable threshold (default 0.10 = 10% edge). Higher threshold = fewer bets, more confident. Calibrate the threshold on a validation slice, not the test slice.

**A4.2 Staking** (`betting/staking.py`)

Quarter-Kelly with caps:
- Kelly fraction: `f = (p*b - q) / b` where `b = price - 1`, `q = 1 - p`.
- Stake = `bankroll * min(f * 0.25, max_stake_pct)`.
- Hard cap on absolute stake (configurable).
- Minimum stake (Betfair has £2 min).

**A4.3 Walk-forward backtest** (`betting/backtest.py`)

Loop:
1. Train on `[start, t)`.
2. For each race in `[t, t + step)`, generate calibrated probs, compute edge against BSP, place virtual bets that meet threshold.
3. Settle bets against actual BSP outcome; deduct 5% commission on net winnings per market.
4. Roll `t` forward by `step` (default: 1 month). Retrain.

Output `data/processed/backtest_results.parquet` with one row per virtual bet: race_id, runner_id, model_prob, market_prob, edge, stake, price, won, pnl.

Report at the end:
- ROI overall, by month, by track, by grade, by edge bucket, by price bucket.
- Total number of bets, hit rate vs expected hit rate (a calibration check on bet selection).
- Max drawdown, longest losing streak, Sharpe-ish ratio.
- **Reality check:** also run the backtest with `model_prob = market_prob`. ROI should be ≈ -commission. If not, there's a bug.

**Milestone A4 done when:** `make backtest` produces results plus a one-page summary report. ROI is positive net of commission over a multi-month test period with thousands of bets. (If it's not positive, that's still a successful milestone — it means the model isn't there yet, and the diagnostics should point at where.)

### A5. Paper trading hook

A minimal live script (`src/greyhound/live/paper_trade.py`) that:
1. Each morning, pulls today's race cards from GBGB.
2. Pulls current Betfair prices (via API) at scheduled times (T-20, T-15, T-10, T-5 minutes before each race).
3. Runs the trained model, identifies value bets.
4. Logs would-be bets to `data/live/paper_bets.parquet` with timestamp and price snapshot.
5. Reconciles against actual BSP after the race.

Run this for at least 2 months before considering Phase B or real money.

---

## Phase B — Final product

Do not start until Phase A paper-trades break-even or better against BSP over ≥2 months and ≥1,000 bets.

### B1. ADVANCED historical data

Purchase 6–12 months of Betfair ADVANCED historical data for UK greyhounds. Build a parser that, given a race, returns a time-indexed series of order-book snapshots at 1-second resolution from T-30min to off.

### B2. Order-book features

For each race × each runner × each time T ∈ {T-20, T-15, T-10, T-5}:
- best back / lay price
- top-3 ladder sizes
- spread
- total matched volume
- price velocity (Δprice over last 60s, 180s)
- price relative to first observed pre-off price
- cross-runner book metrics (overround, sum of probabilities)

### B3. The BSP-prediction model

Target: `BSP / current_best_back`. Features: order-book features above + fundamentals features from Phase A.

LightGBM regression with quantile loss (we care about the distribution, not just the mean — a dog whose median expected shortening is 0% but whose 10th percentile is -20% is risky).

### B4. Combined bet decision

Bet when:
- Phase A model edge > threshold_A (e.g. 8%), AND
- Phase B expected price shortening > threshold_B (e.g. 5%), AND
- Available liquidity at the current best back ≥ minimum economic stake

### B5. Live execution

Stream the Betfair Exchange API. Event-driven (not polled). Place limit orders at the current best back; cancel and re-evaluate on each order-book change. Conservative fill modelling. Detailed bet journal. Hard daily loss limit.

### B6. Honest live evaluation

Track:
- **Fill rate:** of bets the model wanted to place, what fraction got matched at the desired price?
- **Slippage:** average difference between intended price and filled price.
- **Adverse selection:** when you got matched, was it because the price was about to move against you?

These three metrics will determine whether the strategy works in practice or only in backtest.

---

## What "done" looks like

- **Phase A done:** Reproducible pipeline from raw scrape to backtested results. Positive ROI net of commission over a multi-month held-out test. Paper-trading harness running daily. Calibrated probabilities, leak-free features, walk-forward validation.
- **Phase B done:** Live execution against Betfair Exchange, with 3+ months of real (small-stake) results matching backtest expectations within reasonable bounds. Documented procedures for monitoring, halting, and re-training.

---

## Things to flag before doing

If you encounter any of these, stop and surface the question:

- Any change to the schema of `races_with_market.parquet` after A1 is complete.
- Any feature whose value depends on data from after `race_datetime`.
- Any backtest result that looks too good (>20% ROI). Almost always a leak.
- Any place where you're tempted to fit on test data, even "just to see."
- Any third-party data source whose terms of service might restrict scraping or re-use.
- Any deviation from the leakage-proof primitive (`get_form_snapshot`) — all feature computation must go through it.

---

## Out of scope (for now)

- Forecast / tricast / multi-runner markets
- In-play betting
- Tracks outside the GBGB-licensed 18
- Irish greyhound racing (different regulator, different data sources)
- Other sports
- A UI. Reports are static HTML or notebooks until proven otherwise.

---

## First task

Read this brief end-to-end. Then propose a `pyproject.toml`, `Makefile`, and `config/default.yaml` covering the Phase A directory layout, before writing any production code. Do not start scraping until the config and project skeleton are reviewed.
