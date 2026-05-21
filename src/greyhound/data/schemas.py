"""Typed config + canonical data schemas.

Loading the YAML through Pydantic gives us:
  - typed access (`cfg.betting.edge_threshold`) with autocomplete
  - fail-loud validation at startup, not deep in a pipeline run
  - one place to add a new knob

`Run` is the canonical row shape produced by the parser (A1.2) and consumed
by every downstream module. Anything that touches a Run touches this schema.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal

import polars as pl
import yaml
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------- config

class ProjectCfg(BaseModel):
    name: str
    random_seed: int = 42
    timezone: str = "Europe/London"


class PathsCfg(BaseModel):
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    live_dir: Path


class BackoffCfg(BaseModel):
    max_attempts: int = 5
    initial_seconds: float = 2.0
    multiplier: float = 2.0


class GbgbCfg(BaseModel):
    start_date: date
    end_date: date | None = None
    base_url: str
    cache_dir: Path
    request_delay_seconds: float = 2.0
    max_concurrency: int = 1
    backoff: BackoffCfg = BackoffCfg()
    user_agent: str
    respect_robots_txt: bool = True


class BetfairBspCfg(BaseModel):
    tier: Literal["promo", "advanced"] = "promo"
    sport: str = "greyhound"
    country: str = "GB"
    cache_dir: Path


class IngestCfg(BaseModel):
    gbgb: GbgbCfg
    betfair_bsp: BetfairBspCfg


class JoinCfg(BaseModel):
    time_tolerance_seconds: int = 120
    min_match_rate_per_track: float = 0.95


class FormFeatCfg(BaseModel):
    last_n_windows: list[int] = [1, 3, 6]
    best_window_days: int = 90
    trend_window: int = 6
    recent_run_window_days: int = 28
    bayesian_shrinkage_prior_runs: int = 20


class TrapFeatCfg(BaseModel):
    trap_winrate_min_sample: int = 50


class ShapeFeatCfg(BaseModel):
    early_pace_threshold: float = 2.5


class ConnectionsFeatCfg(BaseModel):
    trainer_window_days: int = 30
    trainer_track_window_days: int = 180
    shrinkage_prior_runs: int = 50


class FeaturesCfg(BaseModel):
    form: FormFeatCfg = FormFeatCfg()
    trap: TrapFeatCfg = TrapFeatCfg()
    shape: ShapeFeatCfg = ShapeFeatCfg()
    connections: ConnectionsFeatCfg = ConnectionsFeatCfg()


class SplitsCfg(BaseModel):
    train_min_months: int = 12
    val_months: int = 1
    test_months: int = 1
    step_months: int = 1


class LgbmCfg(BaseModel):
    objective: str = "lambdarank"
    metric: str = "ndcg"
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100


class CalibrationCfg(BaseModel):
    method: Literal["isotonic", "platt", "none"] = "isotonic"
    holdout_fraction: float = 0.2


class ModelCfg(BaseModel):
    type: Literal["lgbm_ranker", "baseline_logit"] = "lgbm_ranker"
    lgbm: LgbmCfg = LgbmCfg()
    softmax_temperature_init: float = 1.0
    calibration: CalibrationCfg = CalibrationCfg()


class StakingCfg(BaseModel):
    method: Literal["quarter_kelly", "flat"] = "quarter_kelly"
    bankroll: float = 1000.0
    max_stake_pct: float = 0.02
    min_stake_gbp: float = 2.0
    max_stake_gbp: float = 50.0
    flat_stake_pct: float = 0.02   # for method="flat": fraction of starting bankroll


class BettingCfg(BaseModel):
    selection: Literal["edge_threshold", "stand_out"] = "edge_threshold"
    edge_threshold: float = 0.10
    standout_min_prob: float = 0.20       # for stand_out: min model_prob for the race's top pick
    standout_min_gap: float = 0.05        # for stand_out: min gap to 2nd-best in the race
    staking: StakingCfg = StakingCfg()
    commission_rate: float = 0.05
    reality_check: bool = True

    @field_validator("commission_rate")
    @classmethod
    def _commission_sane(cls, v: float) -> float:
        if not 0.0 <= v <= 0.1:
            raise ValueError("commission_rate must be in [0, 0.1]; Betfair caps at 5%")
        return v


class PaperTradeCfg(BaseModel):
    snapshot_minutes_before_off: list[int] = [20, 15, 10, 5]
    log_path: Path


class LoggingCfg(BaseModel):
    model_config = {"protected_namespaces": ()}
    level: str = "INFO"
    json_format: bool = Field(default=False, alias="json")


class Config(BaseModel):
    project: ProjectCfg
    paths: PathsCfg
    ingest: IngestCfg
    join: JoinCfg
    features: FeaturesCfg
    splits: SplitsCfg
    model: ModelCfg
    betting: BettingCfg
    paper_trade: PaperTradeCfg
    logging: LoggingCfg = LoggingCfg()


def load_config(path: str | Path) -> Config:
    """Read YAML into a typed Config. Fails loud on schema drift."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


# ---------------------------------------------------------------- run schema

# The canonical Run schema. One row per (race, runner). Matches BRIEF § A1.2.
# Kept as a Polars schema so the parser, joiner, and feature pipeline can
# all enforce it identically.
RUN_SCHEMA: dict[str, pl.DataType] = {
    "race_id":         pl.Utf8,
    "race_datetime":   pl.Datetime(time_zone="UTC"),
    "track":           pl.Utf8,
    "distance_m":      pl.Int32,
    "grade":           pl.Utf8,
    "going":           pl.Float64,
    "dog_id":          pl.Utf8,
    "dog_name":        pl.Utf8,
    "trap":            pl.Int8,
    "sp":              pl.Float64,
    "finish_position": pl.Int8,
    "run_time":        pl.Float64,
    "sectional_1":     pl.Float64,
    "weight_kg":       pl.Float64,
    "trainer_id":      pl.Utf8,
    "trainer_name":    pl.Utf8,
    "comment":         pl.Utf8,
    "bf_safe_name":    pl.Utf8,
}


def empty_runs_frame() -> pl.DataFrame:
    """Empty DataFrame with the canonical Run schema. Useful for parser tests."""
    return pl.DataFrame(schema=RUN_SCHEMA)


# --------------------------------------------------------------- bsp schema

BSP_SCHEMA: dict[str, pl.DataType] = {
    "market_id":     pl.Utf8,
    "event_date":    pl.Date,
    "track":         pl.Utf8,
    "race_time":     pl.Datetime(time_zone="UTC"),
    "selection_id":  pl.Int64,
    "selection_name": pl.Utf8,
    "bf_safe_name":  pl.Utf8,
    "trap":          pl.Int8,
    "bsp":           pl.Float64,
    "won":           pl.Boolean,
    "matched_volume": pl.Float64,
}


# --------------------------------------------------------- single-race input

class RunnerInput(BaseModel):
    """A single runner as fed to the model at inference time."""
    dog_id: str
    trap: int
    weight_kg: float | None = None
    features: dict[str, float | None] = Field(default_factory=dict)


class RaceInput(BaseModel):
    race_id: str
    race_datetime: datetime
    track: str
    distance_m: int
    grade: str | None = None
    runners: list[RunnerInput]
