"""Backtest summary report.

Reads data/processed/backtest_results.parquet and prints a one-page
diagnostic to stdout (also written as a text file to reports/).

Reports:
  - Overall ROI, hit-rate vs expected (calibration sanity)
  - Per-month, per-track, per-grade ROI
  - Per-edge-bucket and per-price-bucket ROI
  - Max drawdown, longest losing streak
  - Brier-score on the model probs (when 'model_prob' present)
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import polars as pl

from greyhound.data.schemas import load_config

log = logging.getLogger(__name__)


def _longest_losing_streak(pnls: np.ndarray) -> int:
    longest = current = 0
    for p in pnls:
        if p < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _max_drawdown(pnls: np.ndarray) -> float:
    if pnls.size == 0:
        return 0.0
    equity = np.cumsum(pnls)
    peaks = np.maximum.accumulate(equity)
    return float((equity - peaks).min())


def overround_diagnostic(features: pl.DataFrame) -> str:
    """Per-race overround stats. Underrounded races (< 1.0) signal voided
    runners — they leak positive 'edge' into the reality check."""
    lines: list[str] = []
    by_race = (
        features
        .filter(pl.col("bsp").is_not_null() & (pl.col("bsp") > 1.0))
        .with_columns((1.0 / pl.col("bsp")).alias("_imp"))
        .group_by("race_id").agg(pl.col("_imp").sum().alias("overround"))
    )
    lines.append("--- Per-race overround distribution ---")
    lines.append(str(by_race["overround"].describe()))
    n_under = int((by_race["overround"] < 1.0).sum())
    lines.append(
        f"Races with overround < 1.0 (likely voided runner): "
        f"{n_under:,} / {by_race.height:,} ({100*n_under/by_race.height:.2f}%)"
    )
    return "\n".join(lines) + "\n"


def summarise(bets: pl.DataFrame) -> str:
    if bets.height == 0:
        return "No bets to summarise.\n"

    pnls = bets["pnl"].to_numpy()
    stakes = bets["stake"].to_numpy()
    won = bets["won"].to_numpy()
    model_probs = bets["model_prob"].to_numpy()

    lines: list[str] = []
    L = lines.append

    L("=" * 72)
    L("BACKTEST SUMMARY")
    L("=" * 72)
    L(f"Bets:            {bets.height:>10,d}")
    L(f"Total stake:     {stakes.sum():>10,.2f}")
    L(f"Total P&L:       {pnls.sum():>10,.2f}")
    L(f"ROI:             {(pnls.sum() / stakes.sum()):>10.4f}")
    L(f"Hit rate:        {won.mean():>10.4f}")
    L(f"Expected hit:    {model_probs.mean():>10.4f}  (calibration sanity)")
    L(f"Max drawdown:    {_max_drawdown(pnls):>10,.2f}")
    L(f"Longest losing:  {_longest_losing_streak(pnls):>10d}")
    brier = float(np.mean((model_probs - won) ** 2))
    L(f"Brier (subset):  {brier:>10.4f}")
    L("")

    # By month
    L("--- ROI by month ---")
    by_month = (
        bets
        .with_columns(pl.col("race_datetime").dt.strftime("%Y-%m").alias("ym"))
        .group_by("ym")
        .agg([
            pl.len().alias("n"),
            pl.col("stake").sum().alias("stake"),
            pl.col("pnl").sum().alias("pnl"),
            pl.col("won").mean().alias("hit"),
        ])
        .with_columns((pl.col("pnl") / pl.col("stake")).alias("roi"))
        .sort("ym")
    )
    L(str(by_month))
    L("")

    # By track
    L("--- ROI by track ---")
    by_track = (
        bets.group_by("track")
            .agg([
                pl.len().alias("n"),
                pl.col("stake").sum().alias("stake"),
                pl.col("pnl").sum().alias("pnl"),
                pl.col("won").mean().alias("hit"),
            ])
            .with_columns((pl.col("pnl") / pl.col("stake")).alias("roi"))
            .sort("roi", descending=True)
    )
    L(str(by_track))
    L("")

    # By edge bucket
    L("--- ROI by edge bucket ---")
    by_edge = (
        bets.with_columns(
            pl.col("edge")
              .cut([0.10, 0.15, 0.20, 0.30, 0.50, 1.00])
              .alias("edge_bucket")
        )
        .group_by("edge_bucket", maintain_order=True)
        .agg([
            pl.len().alias("n"),
            pl.col("stake").sum().alias("stake"),
            pl.col("pnl").sum().alias("pnl"),
            pl.col("won").mean().alias("hit"),
        ])
        .with_columns((pl.col("pnl") / pl.col("stake")).alias("roi"))
        .sort("edge_bucket")
    )
    L(str(by_edge))
    L("")

    # By price bucket
    L("--- ROI by price bucket ---")
    by_price = (
        bets.with_columns(
            pl.col("price")
              .cut([2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
              .alias("price_bucket")
        )
        .group_by("price_bucket", maintain_order=True)
        .agg([
            pl.len().alias("n"),
            pl.col("stake").sum().alias("stake"),
            pl.col("pnl").sum().alias("pnl"),
            pl.col("won").mean().alias("hit"),
        ])
        .with_columns((pl.col("pnl") / pl.col("stake")).alias("roi"))
        .sort("price_bucket")
    )
    L(str(by_price))
    L("=" * 72)

    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    bets_path = cfg.paths.processed_dir / "backtest_results.parquet"
    if not bets_path.exists():
        raise SystemExit(f"Missing {bets_path} — run `make backtest` first.")
    bets = pl.read_parquet(bets_path)

    report = summarise(bets)
    # Add the overround diagnostic if the features file is available.
    features_path = cfg.paths.processed_dir / "features.parquet"
    if features_path.exists():
        features = pl.read_parquet(features_path)
        report += "\n" + overround_diagnostic(features)
    print(report)
    out = cfg.paths.reports_dir / "backtest_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
