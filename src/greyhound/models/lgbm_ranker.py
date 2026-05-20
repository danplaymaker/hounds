"""LightGBM LambdaRank model (A3.2).

Per-race ranking with softmax-to-probability and a learned temperature τ
fit on a validation slice. Calibration (isotonic/Platt) is applied
downstream in `calibration.py`.

The training entry point is the CLI `python -m greyhound.models.lgbm_ranker`,
but the class is the primary API.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

log = logging.getLogger(__name__)


@dataclass
class LgbmRanker:
    feature_cols: list[str]
    params: dict = field(default_factory=dict)
    booster_: lgb.Booster | None = None
    temperature_: float = 1.0

    def _groups(self, df: pl.DataFrame) -> np.ndarray:
        # LightGBM expects group sizes in row order.
        return df.group_by("race_id", maintain_order=True).len()["len"].to_numpy()

    def fit(
        self,
        train: pl.DataFrame,
        val: pl.DataFrame | None = None,
        *,
        cat_features: list[str] | None = None,
    ) -> LgbmRanker:
        X_train = train.select(self.feature_cols).to_pandas()
        y_train = train["won"].to_numpy()
        group_train = self._groups(train)

        dtrain = lgb.Dataset(
            X_train, label=y_train, group=group_train,
            categorical_feature=cat_features or "auto",
        )
        valid_sets = [dtrain]
        valid_names = ["train"]
        callbacks = []
        if val is not None and val.height > 0:
            X_val = val.select(self.feature_cols).to_pandas()
            y_val = val["won"].to_numpy()
            group_val = self._groups(val)
            dval = lgb.Dataset(
                X_val, label=y_val, group=group_val,
                categorical_feature=cat_features or "auto",
                reference=dtrain,
            )
            valid_sets.append(dval)
            valid_names.append("val")
            es = self.params.get("early_stopping_rounds", 100)
            callbacks.append(lgb.early_stopping(es))

        params = {
            "objective": self.params.get("objective", "lambdarank"),
            "metric": self.params.get("metric", "ndcg"),
            "learning_rate": self.params.get("learning_rate", 0.05),
            "num_leaves": self.params.get("num_leaves", 63),
            "min_data_in_leaf": self.params.get("min_data_in_leaf", 200),
            "feature_fraction": self.params.get("feature_fraction", 0.9),
            "bagging_fraction": self.params.get("bagging_fraction", 0.9),
            "bagging_freq": self.params.get("bagging_freq", 5),
            "verbose": -1,
        }
        n_round = self.params.get("num_boost_round", 2000)

        self.booster_ = lgb.train(
            params, dtrain,
            num_boost_round=n_round,
            valid_sets=valid_sets, valid_names=valid_names,
            callbacks=callbacks,
        )

        if val is not None and val.height > 0:
            self._fit_temperature(val)
        return self

    def raw_scores(self, df: pl.DataFrame) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("Model not fit")
        X = df.select(self.feature_cols).to_pandas()
        return self.booster_.predict(X, raw_score=True)

    def predict_proba(self, df: pl.DataFrame, *, temperature: float | None = None) -> np.ndarray:
        scores = self.raw_scores(df)
        race_ids = df["race_id"].to_numpy()
        T = temperature if temperature is not None else self.temperature_
        return _softmax_by_group(scores / max(T, 1e-6), race_ids)

    def _fit_temperature(self, val: pl.DataFrame) -> None:
        scores = self.raw_scores(val)
        race_ids = val["race_id"].to_numpy()
        y = val["won"].to_numpy().astype(np.float64)

        def nll(log_T: float) -> float:
            T = np.exp(log_T)
            probs = _softmax_by_group(scores / max(T, 1e-6), race_ids)
            probs = np.clip(probs, 1e-12, 1.0)
            return -float((y * np.log(probs)).sum())

        res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
        self.temperature_ = float(np.exp(res.x))
        log.info("Fit softmax temperature: %.3f", self.temperature_)

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        assert self.booster_ is not None
        self.booster_.save_model(str(dir_path / "model.txt"))
        with open(dir_path / "meta.json", "w") as f:
            json.dump({
                "feature_cols": self.feature_cols,
                "temperature": self.temperature_,
                "params": self.params,
            }, f, indent=2)

    @classmethod
    def load(cls, dir_path: Path) -> LgbmRanker:
        with open(dir_path / "meta.json") as f:
            meta = json.load(f)
        obj = cls(feature_cols=meta["feature_cols"], params=meta.get("params", {}))
        obj.booster_ = lgb.Booster(model_file=str(dir_path / "model.txt"))
        obj.temperature_ = float(meta.get("temperature", 1.0))
        return obj


def _softmax_by_group(scores: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(group_ids, return_inverse=True)
    max_per = np.full(unique.size, -np.inf)
    np.maximum.at(max_per, inverse, scores)
    shifted = scores - max_per[inverse]
    exps = np.exp(shifted)
    denom = np.zeros(unique.size)
    np.add.at(denom, inverse, exps)
    return exps / denom[inverse]


def main() -> None:
    import argparse

    from greyhound.data.schemas import load_config
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    feat_path = cfg.paths.processed_dir / "features.parquet"
    if not feat_path.exists():
        raise SystemExit(f"Missing features at {feat_path} — run `make features` first.")
    features = pl.read_parquet(feat_path)
    # Trivial single-shot fit (no walk-forward). Walk-forward lives in backtest.
    feature_cols = [
        c for c in features.columns
        if c not in {"race_id", "race_datetime", "dog_id", "track", "won", "grade", "running_style"}
        and features[c].dtype.is_numeric()
    ]
    model = LgbmRanker(feature_cols=feature_cols, params=cfg.model.lgbm.model_dump())
    n = features.height
    cutoff = int(n * 0.8)
    train = features.head(cutoff)
    val = features.tail(n - cutoff)
    model.fit(train, val)
    out = cfg.paths.models_dir / dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model.save(out)
    log.info("Saved model to %s", out)


if __name__ == "__main__":
    main()
