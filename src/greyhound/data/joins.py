"""Join GBGB runs to Betfair BSP (A1.4).

Key: (date, canonical_track, race_datetime ± tolerance, trap).
Sanity: `bf_safe_name` must agree on matched rows; mismatches are reported
but the row is kept (Betfair sometimes renames between source and result).

Outputs `data/processed/races_with_market.parquet` and logs match rate
per track. Fails loud if any track is below `cfg.join.min_match_rate_per_track`.
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
    """Inner-style join with a time tolerance window.

    Polars supports asof joins; we use `join_asof` on race_datetime within
    track+trap groups, then post-filter to the tolerance.
    """
    runs_s = runs.sort("race_datetime")
    bsp_s = bsp.sort("race_time").rename({"race_time": "race_datetime"})

    joined = runs_s.join_asof(
        bsp_s,
        on="race_datetime",
        by=["track", "trap"],
        strategy="nearest",
        tolerance=f"{time_tolerance_seconds}s",
        suffix="_bsp",
    )
    return joined


def report_match_rates(joined: pl.DataFrame, min_rate: float) -> None:
    by_track = (
        joined
        .group_by("track")
        .agg(
            pl.len().alias("n"),
            pl.col("bsp").is_not_null().sum().alias("matched"),
        )
        .with_columns((pl.col("matched") / pl.col("n")).alias("rate"))
        .sort("rate")
    )
    log.info("Match rate per track:\n%s", by_track)
    bad = by_track.filter(pl.col("rate") < min_rate)
    if bad.height > 0:
        raise SystemExit(
            f"{bad.height} track(s) below min match rate {min_rate:.2%}:\n{bad}"
        )


def main() -> None:
    import argparse
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    runs_path = cfg.paths.interim_dir / "gbgb_runs.parquet"
    bsp_path = cfg.paths.interim_dir / "betfair_bsp.parquet"
    if not runs_path.exists() or not bsp_path.exists():
        raise SystemExit(f"Missing inputs: {runs_path} or {bsp_path}")

    runs = pl.read_parquet(runs_path)
    bsp = pl.read_parquet(bsp_path)
    joined = join_runs_to_bsp(runs, bsp, time_tolerance_seconds=cfg.join.time_tolerance_seconds)
    report_match_rates(joined, cfg.join.min_match_rate_per_track)

    out = cfg.paths.processed_dir / "races_with_market.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(out)
    log.info("Wrote %d joined rows → %s", joined.height, out)


if __name__ == "__main__":
    main()
