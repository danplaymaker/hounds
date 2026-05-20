"""Calibration utilities (A3.3).

Two pieces:
  1. Reliability diagram / Brier-score report.
  2. Isotonic / Platt wrapper that takes raw probs from any model and
     returns calibrated probs, then re-normalises per race to sum to 1.

Calibrators are fit on a HOLD-OUT slice that is chronologically *after*
train and *before* test — never the test slice itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class _Calibrator(Protocol):
    def predict(self, p: np.ndarray) -> np.ndarray: ...


@dataclass
class IsotonicCalibrator:
    iso: IsotonicRegression

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(p, y.astype(np.float64))
        return cls(iso)

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.iso.predict(p)


@dataclass
class PlattCalibrator:
    lr: LogisticRegression

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        # Use logit(p) as the single feature so Platt scaling is well-posed.
        eps = 1e-6
        p_clip = np.clip(p, eps, 1 - eps)
        x = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
        lr = LogisticRegression()
        lr.fit(x, y.astype(int))
        return cls(lr)

    def predict(self, p: np.ndarray) -> np.ndarray:
        eps = 1e-6
        p_clip = np.clip(p, eps, 1 - eps)
        x = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1]


def renormalise_by_group(probs: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """After per-row calibration, force per-race probs to sum to 1.

    BRIEF § A3.4 acceptance: |sum_r p_r - 1| < 1e-6 per race.
    """
    unique, inverse = np.unique(group_ids, return_inverse=True)
    denom = np.zeros(unique.size)
    np.add.at(denom, inverse, probs)
    return probs / denom[inverse]


def reliability_diagram(
    probs: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = 10,
    out_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Compute reliability data and optionally save a PNG."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    pred_mean = np.zeros(bins)
    obs_mean = np.zeros(bins)
    counts = np.zeros(bins, dtype=int)
    for b in range(bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            pred_mean[b] = float(probs[mask].mean())
            obs_mean[b] = float(y[mask].mean())

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.scatter(pred_mean, obs_mean, s=20 + 4 * np.sqrt(counts), label="bins")
        ax.set_xlabel("Predicted prob (bin mean)")
        ax.set_ylabel("Observed win rate")
        ax.set_title("Reliability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    return {"pred_mean": pred_mean, "obs_mean": obs_mean, "counts": counts}


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))


def remove_overround(market_prices: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Convert BSP decimal prices to a de-overrounded probability per race."""
    raw = 1.0 / market_prices
    return renormalise_by_group(raw, group_ids)
