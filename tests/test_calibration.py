from __future__ import annotations

import numpy as np

from greyhound.models.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    remove_overround,
    renormalise_by_group,
)


def test_renormalise_sums_to_one() -> None:
    probs = np.array([0.3, 0.2, 0.1, 0.5, 0.5, 0.5, 0.5])
    group = np.array(["A", "A", "A", "B", "B", "B", "B"])
    out = renormalise_by_group(probs, group)
    for g in np.unique(group):
        s = float(out[group == g].sum())
        assert abs(s - 1.0) < 1e-9


def test_isotonic_monotone() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000)
    # True prob is sqrt(p) — uncalibrated. Outcomes drawn from it.
    y = (rng.uniform(size=2000) < np.sqrt(p)).astype(int)
    cal = IsotonicCalibrator.fit(p, y)
    grid = np.linspace(0, 1, 50)
    out = cal.predict(grid)
    # Monotone non-decreasing
    assert np.all(np.diff(out) >= -1e-9)


def test_platt_runs() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    cal = PlattCalibrator.fit(p, y)
    out = cal.predict(p)
    assert out.shape == p.shape
    assert np.all((out >= 0) & (out <= 1))


def test_brier_perfect_zero() -> None:
    y = np.array([0, 1, 0, 1])
    assert brier_score(y.astype(float), y) == 0.0


def test_remove_overround_sums_to_one() -> None:
    prices = np.array([2.0, 4.0, 5.0, 10.0,    3.0, 3.0, 4.0, 6.0])
    group  = np.array(["A", "A", "A", "A",    "B", "B", "B", "B"])
    out = remove_overround(prices, group)
    for g in np.unique(group):
        s = float(out[group == g].sum())
        assert abs(s - 1.0) < 1e-9
