"""Join GBGB runs to Betfair BSP (A1.4).

Key: (canonical_track, trap) + race_datetime within ±tolerance.

Match-rate accounting: GBGB publishes rows for non-runners (withdrew,
reserve, vacant). Betfair has no BSP for those — they never traded.
The brief's >95% match-rate gate applies to dogs that *actually ran*
(non-null `run_time`); we still keep non-runner rows in the output for
completeness, but they're excluded from the gate.

Outputs `data/processed/races_with_market.parquet`. Logs match rate per
track for both populations.
"""

from __future__ import annotations

import logging

import polars as pl

from greyhound.data.schemas import load_config

log = logging.getLogger(__name__)


def join_runs_to_bsp(
    runs: pl.DataFrame,
    bsp: pl.DataFrame,
    *,
    time_tolerance_seconds: int = 120,
) -> pl.DataFrame:
    """Asof-join on race_datetime within (track, trap)."""
    runs_s = runs.sort("race_datetime")
    bsp_s = (
        bsp
        .sort("race_time")
        .rename({"race_time": "race_datetime"})
        .select([
            "track", "trap", "race_datetime",
            "market_id", "bsp", "won", "matched_volume",
            "bf_safe_name",
        ])
        .rename({"bf_safe_name": "bsp_bf_safe_name"})
    )

    return runs_s.join_asof(
        bsp_s,
        on="race_datetime",
        by=["track", "trap"],
        strategy="nearest",
        tolerance=f"{time_tolerance_seconds}s",
        suffix="_bsp",
    )


def report_match_rates(joined: pl.DataFrame, min_rate: float) -> bool:
    """Log match rate per track. Returns True if all tracks meet the gate.

    The gate applies only to GBGB rows where the dog actually ran
    (run_time is not null) — non-runners can't have BSP by definition.
    """
    overall = (
        joined
        .group_by("track")
        .agg(
            pl.len().alias("n_total"),
            pl.col("run_time").is_not_null().sum().alias("n_ran"),
            pl.col("bsp").is_not_null().sum().alias("matched"),
        )
        .with_columns([
            (pl.col("matched") / pl.col("n_total")).alias("rate_all"),
            (
                pl.col("matched")
                / pl.when(pl.col("n_ran") == 0).then(1).otherwise(pl.col("n_ran"))
            ).alias("rate_runners"),
        ])
        .sort("rate_runners")
    )
    log.info("Match rate per track (rate_runners excludes withdrawn dogs):\n%s", overall)

    bad = overall.filter(
        (pl.col("n_ran") > 0) & (pl.col("rate_runners") < min_rate)
    )
    if bad.height > 0:
        log.error(
            "%d track(s) below min match rate %.2f%% among actual runners:\n%s",
            bad.height, min_rate * 100, bad,
        )
        return False
    return True


def main() -> None:
    import argparse
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument(
        "--restrict-to-bsp-dates", action="store_true",
        help="Pre-filter runs to dates present in the BSP cache. "
             "Useful during partial backfills.",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero if any track is below the match-rate gate.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    runs_path = cfg.paths.interim_dir / "gbgb_runs.parquet"
    bsp_path = cfg.paths.interim_dir / "betfair_bsp.parquet"
    if not runs_path.exists() or not bsp_path.exists():
        raise SystemExit(f"Missing inputs: {runs_path} or {bsp_path}")

    runs = pl.read_parquet(runs_path)
    bsp = pl.read_parquet(bsp_path)

    if args.restrict_to_bsp_dates:
        bsp_dates = set(bsp["event_date"].unique().to_list())
        runs = runs.filter(
            pl.col("race_datetime").dt.date().is_in(list(bsp_dates))
        )
        log.info("Restricted to BSP-covered dates: %d run rows remain", runs.height)

    joined = join_runs_to_bsp(
        runs, bsp,
        time_tolerance_seconds=cfg.join.time_tolerance_seconds,
    )
    ok = report_match_rates(joined, cfg.join.min_match_rate_per_track)
    if not ok and args.strict:
        raise SystemExit("Match-rate gate failed (strict mode).")

    out = cfg.paths.processed_dir / "races_with_market.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(out)
    log.info("Wrote %d joined rows → %s", joined.height, out)


if __name__ == "__main__":
    main()
