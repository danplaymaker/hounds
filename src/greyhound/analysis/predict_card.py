"""Score a set of upcoming (or recent) races using the latest model.

INPUT options:
  --date YYYY-MM-DD     pull GBGB's published runners for that date from the
                        cached /api/results dump (works once cards have
                        landed in the results endpoint — typically after
                        the morning sched is finalised, before the off).
  --meeting-jsons PATH  glob of meeting JSON files (same format as
                        data/raw/gbgb/meeting/<id>.json).

OUTPUT:
  - data/processed/predictions_<date>.csv  (one row per runner)
  - data/processed/standouts_<date>.csv    (one row per race, the model's
                                           top pick)
  - data/processed/high_conviction_<date>.csv (subset where
                                              model_prob >= HIGH_CONVICTION_MIN_PROB)
  - reports/predictions_<date>.txt         (human-readable summary)

The script trains a fresh model on ALL data with race_datetime < target
date, so no future leakage. Features are computed using history strictly
before the target date.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from greyhound.analysis.standouts import (
    HIGH_CONVICTION_MIN_PROB,
    MEDIUM_VOLUME_MIN_PROB,
)
from greyhound.data.schemas import load_config
from greyhound.features.pipeline_fast import build_features_fast
from greyhound.features.trap import fit_trap_winrate
from greyhound.ingest.gbgb_parser import parse_meeting_json
from greyhound.models.calibration import IsotonicCalibrator, renormalise_by_group
from greyhound.models.lgbm_ranker import LgbmRanker, select_feature_cols

log = logging.getLogger(__name__)


def load_card_rows(cfg, date: datetime, extra_jsons: list[Path] | None = None) -> pl.DataFrame:
    """Build a runs-frame for the target date's races.

    Pulls from the local cache first (data/raw/gbgb/meeting/*.json), then
    optionally adds any extra meeting JSONs supplied by --meeting-jsons.
    Returns rows in the canonical RUN_SCHEMA shape.
    """
    cache = Path(cfg.ingest.gbgb.cache_dir) / "meeting"
    rows: list[dict] = []

    # Determine which meetings are for the target date by checking each
    # cached meeting's date field (cheap — there are only ~28 meetings per day).
    target_date_str = date.strftime("%d/%m/%Y")
    for path in sorted(cache.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        meeting = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(meeting, dict):
            continue
        if meeting.get("meetingDate") == target_date_str:
            rows.extend(parse_meeting_json(payload))

    for extra in extra_jsons or []:
        try:
            payload = json.loads(extra.read_text())
            rows.extend(parse_meeting_json(payload))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Skip %s: %s", extra, e)

    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    return df.filter(pl.col("race_datetime").dt.date() == date.date())


def predict(cfg, target_date: datetime, extra_jsons: list[Path] | None = None) -> pl.DataFrame:
    """Train fresh on history < target_date, score target_date's races."""
    # All races (history + target).
    full_runs = pl.read_parquet(cfg.paths.interim_dir / "gbgb_runs.parquet")
    target_rows = load_card_rows(cfg, target_date, extra_jsons)
    if target_rows.height == 0:
        log.error("No card data found for %s. Check cache or pass --meeting-jsons.",
                  target_date.date())
        return pl.DataFrame()

    log.info(
        "Loaded %d target-day rows across %d races, %d tracks",
        target_rows.height,
        target_rows["race_id"].n_unique(),
        target_rows["track"].n_unique(),
    )

    # Use joined-with-market frame for training so model sees the same
    # rows as in the backtest. Target rows might not be in that frame
    # (no BSP yet), so we feature-engineer them separately and combine.
    joined = pl.read_parquet(cfg.paths.processed_dir / "races_with_market.parquet")
    train = joined.filter(pl.col("race_datetime") < target_date)

    # Build features for the training set, then for the target rows.
    train_trap_table = fit_trap_winrate(
        train, min_sample=cfg.features.trap.trap_winrate_min_sample,
    )
    train_feats = build_features_fast(train, cfg, trap_winrate_table=train_trap_table)

    # Target features need to look at FULL history (every dog's prior runs
    # including before-target ones). Concatenate target rows to history.
    history_plus_target = pl.concat([full_runs, target_rows.cast(full_runs.schema)],
                                    how="vertical_relaxed").unique(
        subset=["race_id", "dog_id"], keep="first",
    )
    target_feats_all = build_features_fast(
        history_plus_target, cfg, trap_winrate_table=train_trap_table,
    )
    target_feats = target_feats_all.filter(
        pl.col("race_datetime").dt.date() == target_date.date()
    )
    log.info("Built %d feature rows for target day", target_feats.height)

    # Fit model on a small held-out slice for calibration (last month of train)
    cutoff = target_date - timedelta(days=30)
    train_part = train_feats.filter(pl.col("race_datetime") < cutoff)
    val_part = train_feats.filter(
        (pl.col("race_datetime") >= cutoff) & (pl.col("race_datetime") < target_date)
    )
    feature_cols = select_feature_cols(train_feats)
    log.info("Training on %d rows, calibrating on %d", train_part.height, val_part.height)

    model = LgbmRanker(feature_cols=feature_cols, params=cfg.model.lgbm.model_dump())
    model.fit(train_part, val_part)
    raw_val = model.predict_proba(val_part)
    cal = IsotonicCalibrator.fit(raw_val, val_part["won"].to_numpy())
    raw_test = model.predict_proba(target_feats)
    probs = renormalise_by_group(cal.predict(raw_test), target_feats["race_id"].to_numpy())

    out = target_feats.with_columns(pl.Series("model_prob", probs)).with_columns(
        pl.col("model_prob")
          .rank(method="ordinal", descending=True)
          .over("race_id")
          .alias("model_rank"),
    )
    out = out.with_columns((1.0 / pl.col("model_prob")).alias("model_fair_odds"))

    # Bring dog_name, trainer_name, sp, finish_position from raw runs.
    # finish_position lets us see how the prediction did when it's available.
    raw = target_rows.select(["race_id", "dog_id", "dog_name", "trainer_name",
                              "sp", "finish_position"])
    out = out.join(raw, on=["race_id", "dog_id"], how="left")
    return out


def write_outputs(cfg, predictions: pl.DataFrame, target_date: datetime) -> None:
    """Write per-runner, per-race-pick, and high-conviction CSVs."""
    ds = target_date.strftime("%Y-%m-%d")
    keep_cols = [
        "race_id", "race_datetime", "track", "distance_m", "grade",
        "dog_id", "dog_name", "trainer_name", "trap",
        "model_rank", "model_prob", "model_fair_odds",
        "sp", "finish_position",
    ]
    full = predictions.select([c for c in keep_cols if c in predictions.columns]).sort(
        ["race_datetime", "race_id", "model_rank"],
    )
    out_full = cfg.paths.processed_dir / f"predictions_{ds}.csv"
    full.write_csv(out_full)
    log.info("Wrote %d per-runner predictions -> %s", full.height, out_full)

    standouts = full.filter(pl.col("model_rank") == 1).sort("race_datetime")
    out_std = cfg.paths.processed_dir / f"standouts_{ds}.csv"
    standouts.write_csv(out_std)
    log.info("Wrote %d stand-outs (one per race) -> %s", standouts.height, out_std)

    high = standouts.filter(pl.col("model_prob") >= HIGH_CONVICTION_MIN_PROB)
    out_high = cfg.paths.processed_dir / f"high_conviction_{ds}.csv"
    high.write_csv(out_high)
    log.info("Wrote %d high-conviction picks -> %s", high.height, out_high)

    medium = standouts.filter(pl.col("model_prob") >= MEDIUM_VOLUME_MIN_PROB)
    out_med = cfg.paths.processed_dir / f"medium_volume_{ds}.csv"
    medium.write_csv(out_med)
    log.info("Wrote %d medium-volume picks -> %s", medium.height, out_med)

    # Human-readable summary
    lines = [
        "=" * 72,
        f"MODEL PREDICTIONS — {ds}",
        "=" * 72,
        f"Races scored:               {standouts.height}",
        f"Tracks:                     {standouts['track'].n_unique()}",
        f"High-conviction picks:      {high.height} (model_prob >= {HIGH_CONVICTION_MIN_PROB})",
        f"Medium-volume picks:        {medium.height} (model_prob >= {MEDIUM_VOLUME_MIN_PROB})",
        "",
        "HIGH-CONVICTION PICKS (sorted by race time)",
        "-" * 72,
    ]
    def fmt(r: dict) -> str:
        tm = r["race_datetime"].astimezone().strftime("%H:%M")
        name = (r.get("dog_name") or "?")[:22]
        fin = f"finished {r['finish_position']}" if r.get("finish_position") else ""
        sp = f"SP {r['sp']}" if r.get("sp") else ""
        return (
            f"  {tm}  {r['track']:13s} {r['distance_m']:>4}m  "
            f"T{r['trap']}  {name:22s}  "
            f"prob={r['model_prob']:.3f}  fair_odds={r['model_fair_odds']:.2f}  "
            f"{sp:9s}  {fin}"
        )

    if high.height:
        for r in high.iter_rows(named=True):
            lines.append(fmt(r))
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("ALL STAND-OUTS")
    lines.append("-" * 72)
    for r in standouts.iter_rows(named=True):
        marker = "**" if r["model_prob"] >= HIGH_CONVICTION_MIN_PROB else "  "
        lines.append(f"{marker}{fmt(r)}")
    lines.append("=" * 72)
    report_text = "\n".join(lines) + "\n"
    out_txt = cfg.paths.reports_dir / f"predictions_{ds}.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(report_text, encoding="utf-8")
    print(report_text)


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--date", required=True, help="Target race date (YYYY-MM-DD)")
    ap.add_argument(
        "--meeting-jsons", default=None,
        help="Optional glob of additional meeting JSON files (same format as the GBGB API).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    target = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=UTC)
    extras = list(Path().glob(args.meeting_jsons)) if args.meeting_jsons else None
    preds = predict(cfg, target, extras)
    if preds.height == 0:
        raise SystemExit("No predictions produced")
    write_outputs(cfg, preds, target)


if __name__ == "__main__":
    main()
