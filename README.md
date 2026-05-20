# greyhound-value

UK greyhound value-betting model. See `BRIEF.md` for project intent, hard
constraints, and milestones.

## Quickstart

```bash
make install
make test          # runs everything; leakage tests must always pass
make ingest        # A1 — scrape GBGB + ingest Betfair BSP
make features      # A2 — feature engineering (gated on leakage tests)
make train         # A3 — train + calibrate
make backtest      # A4 — walk-forward backtest
```

All pipeline steps read `config/default.yaml` by default. Override with
`make CONFIG=config/<other>.yaml <target>`.

## Repository layout

See `BRIEF.md` § Repository layout. Top-level dirs:

| dir | purpose |
| --- | --- |
| `config/` | YAML config (paths, dates, hyperparams, track aliases) |
| `data/` | gitignored — raw HTML, interim parquet, processed features |
| `models/artifacts/` | versioned trained models (timestamped subdirs) |
| `notebooks/` | EDA + diagnostics (use `nbstripout` before commit) |
| `reports/` | rendered backtest summaries |
| `src/greyhound/` | source package |
| `tests/` | pytest suite — `test_features_no_leakage.py` is non-negotiable |

## Hard constraints (recap)

1. No look-ahead. Every feature at time `T` uses only data with `timestamp < T`.
2. Walk-forward backtesting only.
3. Calibrated probabilities (reliability diagram + isotonic if needed).
4. Backtest against BSP (Phase A) or reconstructed book (Phase B).
5. Commission accounted for in every ROI figure.
6. Dogs identified by GBGB ID, never by name.
7. Reproducibility: seeded RNGs, versioned artifacts.

If you're tempted to bend any of these, stop and re-read `BRIEF.md` §
"Hard constraints and non-negotiables".
