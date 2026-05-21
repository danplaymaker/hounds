"""Staking (A4.2).

Two policies:
  - Quarter-Kelly with hard caps.
  - Flat: a fixed fraction of the *starting* bankroll, regardless of edge.
    Useful when the model is uncalibrated enough that Kelly amplifies the
    wrong bets, and when you want a predictable budget per bet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StakingPolicy:
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.02      # max % of bankroll per bet
    min_stake_gbp: float = 2.0       # Betfair minimum
    max_stake_gbp: float = 50.0      # absolute cap

    def stake(self, prob: float, price: float) -> float:
        b = price - 1.0
        q = 1.0 - prob
        if b <= 0 or prob <= 0:
            return 0.0
        f = (prob * b - q) / b
        if f <= 0:
            return 0.0
        raw = self.bankroll * min(f * self.kelly_fraction, self.max_stake_pct)
        capped = min(raw, self.max_stake_gbp)
        if capped < self.min_stake_gbp:
            return 0.0
        return float(capped)


@dataclass
class FlatStakingPolicy:
    """Flat fraction of the *starting* bankroll. Stake doesn't change with
    bankroll drift inside a fold — every bet is the same nominal size."""
    starting_bankroll: float = 100.0
    flat_pct: float = 0.02            # 2% of starting bankroll = 2 units of 100
    min_stake_gbp: float = 2.0
    max_stake_gbp: float = 1e9

    def stake(self, prob: float, price: float) -> float:
        raw = self.starting_bankroll * self.flat_pct
        if raw < self.min_stake_gbp:
            return 0.0
        return float(min(raw, self.max_stake_gbp))
