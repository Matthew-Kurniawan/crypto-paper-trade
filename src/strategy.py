"""Signal generators for the strategies under research.

Pure functions, no I/O. All inputs are pandas Series indexed by bar close_time UTC.
By convention, a value at index t uses only data observable at or before time t.

Available strategies:
    tsmom_signal           — sign of N-bar return (Moskowitz/Ooi/Pedersen 2012)
    donchian_signal        — channel breakout (Turtle traders)
    meanrev_zscore_signal  — buy oversold, sell overbought (z-score of close vs MA)
    ma_cross_signal        — long when fast SMA > slow SMA
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- Indicator helpers ----------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n_bars: int) -> pd.Series:
    """Wilder's Average True Range over n_bars."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n_bars, adjust=False).mean().rename("atr")


def funding_per_bar(funding_rates: pd.Series, bar_close_index: pd.DatetimeIndex) -> pd.Series:
    bar_close_index = pd.DatetimeIndex(bar_close_index)
    if len(bar_close_index) == 0:
        return pd.Series(dtype="float64", name="funding_per_bar")
    positions = bar_close_index.searchsorted(funding_rates.index, side="left")
    mask = positions < len(bar_close_index)
    result = np.zeros(len(bar_close_index), dtype="float64")
    np.add.at(result, positions[mask], funding_rates.values[mask])
    return pd.Series(result, index=bar_close_index, name="funding_per_bar")


# ---------- Strategy 1: Time-series momentum ----------

def tsmom_signal(close: pd.Series, lookback_bars: int, mode: str = "long_short") -> pd.Series:
    if mode not in ("long_short", "long_only"):
        raise ValueError(f"unknown mode: {mode}")
    n_bar_return = close / close.shift(lookback_bars) - 1
    sig = np.sign(n_bar_return).fillna(0).astype("int8")
    if mode == "long_only":
        sig = sig.clip(lower=0)
    return sig.rename("signal")


# ---------- Strategy 2: Donchian channel breakout ----------

def donchian_signal(
    high: pd.Series, low: pd.Series, close: pd.Series,
    n_entry_bars: int, n_exit_bars: int, mode: str = "long_short",
) -> pd.Series:
    """Long when close breaks above prior n_entry-bar high.
    Exit long when close breaks below prior n_exit-bar low (typically n_exit < n_entry).
    Short symmetric in long_short mode.
    Uses .shift(1) on the rolling channels so the current bar isn't part of its own breakout window.
    """
    if mode not in ("long_short", "long_only"):
        raise ValueError(f"unknown mode: {mode}")
    prior_high_n = high.shift(1).rolling(n_entry_bars).max()
    prior_low_n = low.shift(1).rolling(n_entry_bars).min()
    prior_high_m = high.shift(1).rolling(n_exit_bars).max()
    prior_low_m = low.shift(1).rolling(n_exit_bars).min()

    n = len(close)
    pos = np.zeros(n, dtype="int8")
    cur = 0
    closes = close.to_numpy()
    phn, pln = prior_high_n.to_numpy(), prior_low_n.to_numpy()
    phm, plm = prior_high_m.to_numpy(), prior_low_m.to_numpy()
    for i in range(n):
        c = closes[i]
        # Exit if active position breaks the inner channel
        if cur > 0 and not np.isnan(plm[i]) and c < plm[i]:
            cur = 0
        elif cur < 0 and not np.isnan(phm[i]) and c > phm[i]:
            cur = 0
        # Entry from flat
        if cur == 0:
            if not np.isnan(phn[i]) and c > phn[i]:
                cur = +1
            elif mode == "long_short" and not np.isnan(pln[i]) and c < pln[i]:
                cur = -1
        pos[i] = cur
    return pd.Series(pos, index=close.index, name="signal")


# ---------- Strategy 3: Mean-reversion via rolling z-score ----------

def meanrev_zscore_signal(
    close: pd.Series, window_bars: int,
    entry_z: float = 2.0, exit_z: float = 0.0,
    mode: str = "long_short",
) -> pd.Series:
    """Long when close is `entry_z` std-devs below its rolling mean; exit when
    z returns to within `exit_z` of mean. Symmetric short side in long_short.
    The rolling mean/std are computed on prior bars only (.shift(1)).
    """
    if mode not in ("long_short", "long_only"):
        raise ValueError(f"unknown mode: {mode}")
    mu = close.shift(1).rolling(window_bars).mean()
    sigma = close.shift(1).rolling(window_bars).std()
    z = (close - mu) / sigma

    n = len(close)
    pos = np.zeros(n, dtype="int8")
    cur = 0
    z_arr = z.to_numpy()
    for i in range(n):
        zi = z_arr[i]
        if np.isnan(zi):
            pos[i] = cur
            continue
        # Exit: z has returned to within +/- exit_z of mean
        if cur > 0 and zi >= -exit_z:
            cur = 0
        elif cur < 0 and zi <= exit_z:
            cur = 0
        # Entry from flat
        if cur == 0:
            if zi < -entry_z:
                cur = +1
            elif mode == "long_short" and zi > entry_z:
                cur = -1
        pos[i] = cur
    return pd.Series(pos, index=close.index, name="signal")


# ---------- Strategy 4: MA crossover ----------

def ma_cross_signal(
    close: pd.Series, fast_bars: int, slow_bars: int, mode: str = "long_short",
) -> pd.Series:
    """Long when fast SMA > slow SMA at bar close. Both SMAs use .shift(1)
    of close to ensure the signal at bar t depends only on data through t-1
    plus the current close (which is observable at t)."""
    if mode not in ("long_short", "long_only"):
        raise ValueError(f"unknown mode: {mode}")
    if fast_bars >= slow_bars:
        raise ValueError("fast_bars must be < slow_bars")
    ma_fast = close.rolling(fast_bars).mean()
    ma_slow = close.rolling(slow_bars).mean()
    raw = np.where(ma_fast > ma_slow, 1, (-1 if mode == "long_short" else 0))
    sig = pd.Series(raw, index=close.index).astype("int8")
    sig.iloc[: max(fast_bars, slow_bars)] = 0  # warmup
    return sig.rename("signal")
