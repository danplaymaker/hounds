"""Staking (A4.2).

Fractional Kelly with hard caps. Negative-Kelly results return 0 (no bet).
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
