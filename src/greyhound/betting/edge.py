"""Edge calculation (A4.1).

edge = model_prob * price - 1

A positive edge means the model thinks the price is too generous. We
bet only when edge > threshold (default 10%, calibrated on val slice).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def compute_edge(model_prob: pl.Series | np.ndarray, price: pl.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(model_prob, dtype=np.float64)
    o = np.asarray(price, dtype=np.float64)
    return p * o - 1.0


def select_bets(
    df: pl.DataFrame,
    *,
    prob_col: str = "model_prob",
    price_col: str = "bsp",
    edge_threshold: float = 0.10,
) -> pl.DataFrame:
    """Return rows with positive edge above threshold."""
    edge = compute_edge(df[prob_col], df[price_col])
    return df.with_columns(pl.Series("edge", edge)).filter(pl.col("edge") > edge_threshold)
