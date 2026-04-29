"""Vol-targeted buy-and-hold benchmark.

Position size each bar = vol_target / trailing_realized_vol, capped at max_leverage.
The position is rebalanced bar-by-bar (no transaction cost modeled — this is a
benchmark, not a real strategy). Use this to compare strategies against a
'smart passive' baseline rather than naive 100% buy-and-hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_equity(
    close: pd.Series,
    vol_target: float = 0.30,
    vol_window_bars: int = 63,
    bars_per_year: float = 365,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Vol-targeted long-only buy-and-hold equity curve, normalized to start at 1.0."""
    rets = close.pct_change().fillna(0)
    realized_vol = rets.rolling(vol_window_bars).std() * np.sqrt(bars_per_year)
    pos = (vol_target / realized_vol).shift(1)
    pos = pos.fillna(0).clip(upper=max_leverage, lower=0)
    pf_rets = pos * rets
    eq = (1 + pf_rets).cumprod()
    return eq.rename("vol_target_bh_equity")


def equal_weight_basket_vol_target(
    closes: dict[str, pd.Series],
    vol_target: float = 0.30,
    vol_window_bars: int = 63,
    bars_per_year: float = 365,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Equal-weight basket of assets, each sized to vol_target/N. Aligns on common index."""
    df = pd.DataFrame(closes)
    df = df.dropna(how="all")
    n_assets = df.shape[1]
    rets = df.pct_change().fillna(0)
    per_asset_target = vol_target / np.sqrt(n_assets)  # so portfolio vol ≈ vol_target if uncorrelated
    realized_vols = rets.rolling(vol_window_bars).std() * np.sqrt(bars_per_year)
    pos = (per_asset_target / realized_vols).shift(1).fillna(0).clip(upper=max_leverage, lower=0)
    pf_rets = (pos * rets).sum(axis=1)
    eq = (1 + pf_rets).cumprod()
    return eq.rename("vol_target_basket_equity")


def summary_metrics(equity: pd.Series, bars_per_year: float) -> dict:
    rets = equity.pct_change().fillna(0)
    n = len(equity)
    years = n / bars_per_year
    final = float(equity.iloc[-1])
    cagr = final ** (1 / years) - 1 if final > 0 and years > 0 else float("nan")
    sharpe = (rets.mean() / rets.std()) * np.sqrt(bars_per_year) if rets.std() > 0 else 0.0
    rmax = equity.cummax()
    max_dd = float(((equity - rmax) / rmax).min())
    vol_ann = float(rets.std() * np.sqrt(bars_per_year))
    return {"sharpe": float(sharpe), "cagr": float(cagr) if not np.isnan(cagr) else 0.0,
            "max_dd": max_dd, "vol_ann": vol_ann, "final_eq": final, "n_bars": n}
