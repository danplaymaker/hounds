from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from greyhound.betting.edge import compute_edge, select_bets
from greyhound.betting.staking import StakingPolicy


def test_compute_edge_positive_when_underpriced() -> None:
    edges = compute_edge(np.array([0.40]), np.array([3.0]))
    assert edges[0] == pytest.approx(0.20)


def test_compute_edge_zero_at_fair_price() -> None:
    edges = compute_edge(np.array([0.50]), np.array([2.0]))
    assert abs(edges[0]) < 1e-9


def test_select_bets_threshold() -> None:
    df = pl.DataFrame({
        "model_prob": [0.4, 0.3, 0.2],
        "bsp":        [3.0, 4.0, 6.0],
    })
    # Edges: 0.20, 0.20, 0.20
    out = select_bets(df, edge_threshold=0.15)
    assert out.height == 3
    out2 = select_bets(df, edge_threshold=0.25)
    assert out2.height == 0


def test_staking_quarter_kelly() -> None:
    pol = StakingPolicy(bankroll=1000.0, max_stake_pct=1.0, min_stake_gbp=0.0, max_stake_gbp=1e9)
    # p=0.5, price=3.0: b=2, q=0.5. f=(0.5*2-0.5)/2 = 0.25. 0.25*0.25 = 0.0625 → £62.50
    s = pol.stake(0.5, 3.0)
    assert s == pytest.approx(62.5)


def test_staking_negative_kelly_returns_zero() -> None:
    pol = StakingPolicy(bankroll=1000.0)
    # p=0.1, price=2.0: b=1, q=0.9. f=(0.1-0.9)/1 = -0.8 → 0 stake.
    assert pol.stake(0.1, 2.0) == 0.0


def test_staking_min_stake_floor() -> None:
    pol = StakingPolicy(bankroll=1000.0, min_stake_gbp=10.0, max_stake_pct=0.001)
    # max_stake_pct caps at £1, below min → 0.
    assert pol.stake(0.5, 3.0) == 0.0


def test_staking_max_stake_cap() -> None:
    pol = StakingPolicy(bankroll=100_000.0, max_stake_pct=1.0, max_stake_gbp=100.0)
    assert pol.stake(0.5, 3.0) == 100.0
