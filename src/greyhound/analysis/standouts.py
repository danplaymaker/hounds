"""Standout analysis (no betting decisions).

Picks the model's top dog (by calibrated model_prob) in every race,
without looking at BSP. Then reports what BSP that pick actually had,
how often it matched the market favourite, and how well-calibrated the
predicted probabilities are. Use this to gauge whether the model picks
'favourite-quality' dogs that you can manually compare to live bookmaker
markets.

Run: python -m greyhound.analysis.standouts --config config/default.yaml
Output: data/processed/standout_picks.parquet + .csv
"""

from __future__ import annotations

import logging
from datetime import UTC

import polars as pl
from dateutil.relativedelta import relativedelta

from greyhound.data.schemas import load_config
from greyhound.models.calibration import IsotonicCalibrator, renormalise_by_group
from greyhound.models.lgbm_ranker import LgbmRanker, select_feature_cols

log = logging.getLogger(__name__)


def run(cfg) -> pl.DataFrame:
    feat = pl.read_parquet(cfg.paths.processed_dir / "features.parquet").sort("race_datetime")
    start = feat["race_datetime"][0]
    end = feat["race_datetime"][-1]
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)

    train_min = relativedelta(months=cfg.splits.train_min_months)
    val_len = relativedelta(months=cfg.splits.val_months)
    test_len = relativedelta(months=cfg.splits.test_months)
    step = relativedelta(months=cfg.splits.step_months)

    cursor = start + train_min
    all_picks: list[pl.DataFrame] = []
    while cursor + val_len + test_len <= end:
        train = feat.filter(pl.col("race_datetime") < cursor)
        val = feat.filter(
            (pl.col("race_datetime") >= cursor) & (pl.col("race_datetime") < cursor + val_len)
        )
        test = feat.filter(
            (pl.col("race_datetime") >= cursor + val_len)
            & (pl.col("race_datetime") < cursor + val_len + test_len)
        )
        if min(train.height, val.height, test.height) == 0:
            cursor = cursor + step
            continue

        log.info(
            "Window train<%s val<%s test<%s n=%d/%d/%d",
            cursor.date(), (cursor + val_len).date(),
            (cursor + val_len + test_len).date(),
            train.height, val.height, test.height,
        )
        feature_cols = select_feature_cols(feat)
        model = LgbmRanker(feature_cols=feature_cols, params=cfg.model.lgbm.model_dump())
        model.fit(train, val)
        cal = IsotonicCalibrator.fit(model.predict_proba(val), val["won"].to_numpy())
        cal_test = cal.predict(model.predict_proba(test))
        probs = renormalise_by_group(cal_test, test["race_id"].to_numpy())

        t = test.with_columns(pl.Series("model_prob", probs)).with_columns(
            pl.col("bsp").rank(method="ordinal").over("race_id").alias("mkt_rank"),
            pl.col("model_prob").rank(method="ordinal", descending=True).over("race_id").alias("model_rank"),
        )
        # Per-race gap to 2nd-best model prob (a measure of "stand-out-ness")
        t = t.with_columns(
            (pl.col("model_prob") - pl.col("model_prob").sort(descending=True).slice(1, 1).first())
            .over("race_id")
            .alias("prob_gap_to_2nd")
        )
        standouts = t.filter(pl.col("model_rank") == 1)
        all_picks.append(standouts.select([
            "race_id", "race_datetime", "track", "distance_m",
            "dog_id", "model_prob", "prob_gap_to_2nd",
            "bsp", "mkt_rank", "won",
        ]))
        cursor = cursor + step

    picks = pl.concat(all_picks) if all_picks else pl.DataFrame()
    return picks


def report(picks: pl.DataFrame) -> str:
    lines: list[str] = []
    L = lines.append
    picks_b = picks.filter(pl.col("bsp").is_not_null() & (pl.col("bsp") > 1.0))

    L("=" * 72)
    L("STANDOUT PICK ANALYSIS — model selects the field's top dog, BSP-blind")
    L("=" * 72)
    L(f"Total picks (one per race):              {picks.height:,}")
    L(f"With BSP (Betfair had a market):         {picks_b.height:,} "
      f"({100*picks_b.height/max(picks.height,1):.1f}%)")
    L("")
    L(f"Average BSP of picks:                    £{float(picks_b['bsp'].mean()):.2f}")
    L(f"Median BSP:                              £{float(picks_b['bsp'].median()):.2f}")
    L(f"Average market-rank of picks (1=fav):    {float(picks_b['mkt_rank'].mean()):.2f}")
    L(f"Overall hit rate on stand-outs:          {float(picks['won'].mean()):.1%}")
    L("")
    agree = picks_b.filter(pl.col("mkt_rank") == 1)
    disagree = picks_b.filter(pl.col("mkt_rank") > 1)
    L(f"Times model agrees with market favourite: {agree.height:,} "
      f"({100*agree.height/picks_b.height:.1f}%)")
    L(f"  hit rate:                              {float(agree['won'].mean()):.1%}")
    L(f"  market-implied (1/avg_BSP):            {1.0/float(agree['bsp'].mean()):.1%}")
    L(f"Times model disagrees (picks longer dog): {disagree.height:,} "
      f"({100*disagree.height/picks_b.height:.1f}%)")
    L(f"  hit rate:                              {float(disagree['won'].mean()):.1%}")
    L(f"  market-implied (1/avg_BSP):            {1.0/float(disagree['bsp'].mean()):.1%}")
    L("")
    L("--- Calibration: predicted probability vs realised win rate ---")
    cal_view = picks_b.with_columns(
        pl.col("model_prob").cut([0.20, 0.25, 0.30, 0.40, 0.50, 0.70]).alias("p_b")
    )
    cal = (
        cal_view.group_by("p_b", maintain_order=True).agg(
            pl.len().alias("n"),
            pl.col("model_prob").mean().alias("mean_prob"),
            pl.col("won").mean().alias("actual_hit"),
            pl.col("bsp").mean().alias("avg_bsp"),
        ).sort("p_b")
    )
    L(str(cal))
    L("")
    L("--- Hit rate by BSP band ---")
    bsp_view = picks_b.with_columns(
        pl.col("bsp").cut([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0]).alias("bsp_b")
    )
    L(str(
        bsp_view.group_by("bsp_b", maintain_order=True).agg(
            pl.len().alias("n"),
            pl.col("won").mean().alias("hit_rate"),
            pl.col("model_prob").mean().alias("avg_model_prob"),
        ).sort("bsp_b")
    ))
    L("=" * 72)
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    picks = run(cfg)
    out_pq = cfg.paths.processed_dir / "standout_picks.parquet"
    picks.write_parquet(out_pq)
    picks_b = picks.filter(pl.col("bsp").is_not_null())
    picks_b = picks_b.with_columns([
        (1.0 / pl.col("bsp")).alias("mkt_implied_prob"),
        (pl.col("model_prob") - 1.0 / pl.col("bsp")).alias("edge_vs_mkt"),
    ])
    out_csv = cfg.paths.processed_dir / "standout_picks.csv"
    picks_b.sort(["race_datetime", "model_prob"], descending=[False, True]).write_csv(out_csv)

    rep = report(picks)
    print(rep)
    out_txt = cfg.paths.reports_dir / "standout_analysis.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(rep, encoding="utf-8")

    log.info("Wrote %d picks to %s and %s", picks.height, out_pq, out_csv)
    log.info("Wrote report to %s", out_txt)


if __name__ == "__main__":
    main()
