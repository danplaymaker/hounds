"""Single-race inference (A3.4).

`predict_race` accepts a per-runner feature dict for one race and returns
calibrated probabilities that sum to 1.0 within 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from greyhound.models.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    renormalise_by_group,
)
from greyhound.models.lgbm_ranker import LgbmRanker


@dataclass
class CalibratedModel:
    ranker: LgbmRanker
    calibrator: IsotonicCalibrator | PlattCalibrator | None = None

    def predict_race(self, race_runners: pl.DataFrame) -> dict[str, float]:
        if "race_id" not in race_runners.columns:
            race_runners = race_runners.with_columns(pl.lit("_one").alias("race_id"))

        raw_probs = self.ranker.predict_proba(race_runners)
        if self.calibrator is not None:
            raw_probs = self.calibrator.predict(raw_probs)
        group = race_runners["race_id"].to_numpy()
        probs = renormalise_by_group(raw_probs, group)

        s = float(probs.sum())
        if not np.isfinite(s) or abs(s - 1.0) > 1e-6:
            raise RuntimeError(f"Probs do not sum to 1: sum={s}")

        return {
            str(dog_id): float(p)
            for dog_id, p in zip(race_runners["dog_id"].to_list(), probs)
        }
