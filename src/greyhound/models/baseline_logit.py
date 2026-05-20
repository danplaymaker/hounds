"""Conditional logit baseline (A3.1).

Per-race softmax over runner scores. Trained by maximising log-likelihood
of the actual winner. Fast, hard to overfit, interpretable — this is the
bar the LGBM ranker has to clear.

Implemented in pure NumPy + SciPy. No statsmodels dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize

log = logging.getLogger(__name__)


@dataclass
class ConditionalLogit:
    feature_cols: list[str]
    coef_: np.ndarray | None = None
    fit_loss_: float | None = None

    def _build_groups(self, df: pl.DataFrame) -> list[np.ndarray]:
        return [g.row_count() for _, g in df.group_by("race_id", maintain_order=True)]

    def fit(self, df: pl.DataFrame, *, l2: float = 1e-4, max_iter: int = 200) -> ConditionalLogit:
        if "race_id" not in df.columns or "won" not in df.columns:
            raise ValueError("df must have race_id and won columns")
        # Replace nulls with 0 for the baseline; tree models handle nulls
        # natively but logit does not. Document this divergence in tests.
        X = df.select(self.feature_cols).fill_null(0.0).fill_nan(0.0).to_numpy()
        y = df["won"].to_numpy().astype(np.float64)
        race_ids = df["race_id"].to_numpy()

        unique_races, inverse = np.unique(race_ids, return_inverse=True)
        n_races = unique_races.size

        def nll(beta: np.ndarray) -> float:
            scores = X @ beta
            # softmax within race using inverse index
            max_per_race = np.full(n_races, -np.inf)
            np.maximum.at(max_per_race, inverse, scores)
            shifted = scores - max_per_race[inverse]
            exps = np.exp(shifted)
            denom = np.zeros(n_races)
            np.add.at(denom, inverse, exps)
            log_probs = shifted - np.log(denom[inverse])
            # log-likelihood is sum over winners
            ll = float((y * log_probs).sum())
            reg = l2 * float((beta ** 2).sum())
            return -ll + reg

        x0 = np.zeros(X.shape[1])
        res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": max_iter})
        self.coef_ = res.x
        self.fit_loss_ = float(res.fun)
        log.info("Baseline logit fit: loss=%.4f, n_races=%d, n_feat=%d",
                 self.fit_loss_, n_races, X.shape[1])
        return self

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fit")
        X = df.select(self.feature_cols).fill_null(0.0).fill_nan(0.0).to_numpy()
        race_ids = df["race_id"].to_numpy()
        unique, inverse = np.unique(race_ids, return_inverse=True)
        scores = X @ self.coef_
        max_per = np.full(unique.size, -np.inf)
        np.maximum.at(max_per, inverse, scores)
        shifted = scores - max_per[inverse]
        exps = np.exp(shifted)
        denom = np.zeros(unique.size)
        np.add.at(denom, inverse, exps)
        return exps / denom[inverse]
