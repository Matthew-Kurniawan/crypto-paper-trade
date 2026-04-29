"""Single-config backtester for TSMOM with stops, fees, slippage, funding.

Conventions:
- Bar-close execution: signal at bar t triggers position change at close[t].
- Stops are checked against bar's intra-bar high/low *before* mark-to-market.
- After a stop-out in direction d, re-entry in direction d is blocked until
  the signal moves off d (i.e. requires a fresh trend signal).
- Funding is applied to the position held during a bar, at the rate(s) that
  fired within the bar window (see strategy.funding_per_bar).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestParams:
    risk_pct: float = 0.02            # fraction of equity risked per trade
    stop_atr_mult: float = 2.0        # stop = entry +/- N * ATR
    leverage_cap: float = 3.0         # cap on |notional| / equity
    fee_per_side: float = 0.0008      # 0.08% taker (Binance USDM, conservative)
    slippage_per_side: float = 0.0002 # 2 bps each side
    initial_equity: float = 1.0       # normalized starting equity


def backtest(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    signal: pd.Series,
    atr_series: pd.Series,
    funding_per_bar: pd.Series,
    params: BacktestParams,
) -> dict:
    n = len(close)
    closes = close.to_numpy()
    highs = high.to_numpy()
    lows = low.to_numpy()
    sigs = signal.to_numpy()
    atrs = atr_series.to_numpy()
    funds = funding_per_bar.to_numpy()

    eq_arr = np.zeros(n)
    pos_arr = np.zeros(n)
    trade_log = []

    cur_eq = params.initial_equity
    cur_pos = 0.0           # BTC units, signed
    cur_entry = np.nan
    cur_stop = np.nan
    blocked_dir = 0         # direction blocked from re-entry after a stop
    cost_per_side = params.fee_per_side + params.slippage_per_side
    open_trade: dict | None = None

    for i in range(n):
        # --- 1. Check stop on this bar's intra-bar range ---
        stop_hit = False
        if i > 0 and cur_pos != 0 and not np.isnan(cur_stop):
            if cur_pos > 0 and lows[i] <= cur_stop:
                exit_px = cur_stop
                cur_eq += cur_pos * (exit_px - closes[i - 1])
                cur_eq -= abs(cur_pos) * exit_px * cost_per_side
                blocked_dir = +1
                if open_trade is not None:
                    open_trade.update(
                        exit_idx=i, exit_price=exit_px, exit_reason="stop",
                        pnl=cur_eq - open_trade["equity_before_entry"],
                    )
                    trade_log.append(open_trade)
                    open_trade = None
                cur_pos = 0.0
                cur_stop = np.nan
                stop_hit = True
            elif cur_pos < 0 and highs[i] >= cur_stop:
                exit_px = cur_stop
                cur_eq += cur_pos * (exit_px - closes[i - 1])
                cur_eq -= abs(cur_pos) * exit_px * cost_per_side
                blocked_dir = -1
                if open_trade is not None:
                    open_trade.update(
                        exit_idx=i, exit_price=exit_px, exit_reason="stop",
                        pnl=cur_eq - open_trade["equity_before_entry"],
                    )
                    trade_log.append(open_trade)
                    open_trade = None
                cur_pos = 0.0
                cur_stop = np.nan
                stop_hit = True

        # --- 2. Mark to market on close-to-close move (if not stopped this bar) ---
        if not stop_hit and i > 0 and cur_pos != 0:
            cur_eq += cur_pos * (closes[i] - closes[i - 1])

        # --- 3. Apply funding for this bar (long pays positive funding) ---
        if cur_pos != 0 and funds[i] != 0.0:
            cur_eq -= cur_pos * closes[i] * funds[i]

        # --- 4. Signal at this bar's close: possibly flip ---
        target = int(sigs[i])
        cur_dir = int(np.sign(cur_pos)) if cur_pos != 0 else 0

        # Clear the re-entry block once signal moves off the blocked direction
        if blocked_dir != 0 and target != blocked_dir:
            blocked_dir = 0

        if target != cur_dir and target != blocked_dir:
            # Close existing position if any
            if cur_pos != 0:
                cur_eq -= abs(cur_pos) * closes[i] * cost_per_side
                if open_trade is not None:
                    open_trade.update(
                        exit_idx=i, exit_price=closes[i], exit_reason="signal_flip",
                        pnl=cur_eq - open_trade["equity_before_entry"],
                    )
                    trade_log.append(open_trade)
                    open_trade = None
                cur_pos = 0.0
                cur_stop = np.nan

            # Open new position if signaled and ATR available and equity positive
            if target != 0 and not np.isnan(atrs[i]) and atrs[i] > 0 and cur_eq > 0:
                stop_distance = params.stop_atr_mult * atrs[i]
                size_btc_by_risk = (params.risk_pct * cur_eq) / stop_distance
                size_btc_by_lev = (params.leverage_cap * cur_eq) / closes[i]
                size_btc = min(size_btc_by_risk, size_btc_by_lev)

                equity_before_entry = cur_eq
                cur_pos = target * size_btc
                cur_entry = closes[i]
                cur_stop = cur_entry - target * stop_distance
                cur_eq -= abs(cur_pos) * cur_entry * cost_per_side

                open_trade = {
                    "entry_idx": i,
                    "entry_time": close.index[i],
                    "entry_price": cur_entry,
                    "direction": target,
                    "size_btc": size_btc,
                    "stop_price": cur_stop,
                    "equity_before_entry": equity_before_entry,
                    "size_limited_by": (
                        "leverage_cap" if size_btc_by_lev < size_btc_by_risk else "risk"
                    ),
                }

        eq_arr[i] = cur_eq
        pos_arr[i] = cur_pos

    # Close any open position at end of data for clean accounting
    if cur_pos != 0:
        cur_eq -= abs(cur_pos) * closes[-1] * cost_per_side
        if open_trade is not None:
            open_trade.update(
                exit_idx=n - 1, exit_price=closes[-1], exit_reason="end_of_data",
                pnl=cur_eq - open_trade["equity_before_entry"],
            )
            trade_log.append(open_trade)
        eq_arr[-1] = cur_eq

    equity = pd.Series(eq_arr, index=close.index, name="equity")
    position = pd.Series(pos_arr, index=close.index, name="position")
    trades = pd.DataFrame(trade_log)
    if len(trades):
        trades["exit_time"] = trades["exit_idx"].map(lambda j: close.index[j])

    return {"equity": equity, "position": position, "trades": trades}


def summary_stats(
    equity: pd.Series, position: pd.Series, trades: pd.DataFrame, bars_per_year: float
) -> dict:
    rets = equity.pct_change().fillna(0)
    n_bars = len(equity)
    years = n_bars / bars_per_year
    final_eq = float(equity.iloc[-1])

    cagr = final_eq ** (1 / years) - 1 if final_eq > 0 and years > 0 else float("nan")
    sharpe = (
        (rets.mean() / rets.std()) * np.sqrt(bars_per_year) if rets.std() > 0 else 0.0
    )
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())
    in_market = float((position != 0).mean())
    vol_ann = float(rets.std() * np.sqrt(bars_per_year))

    n_trades = int(len(trades))
    if n_trades > 0:
        wins = int((trades["pnl"] > 0).sum())
        win_rate = wins / n_trades
        avg_pnl = float(trades["pnl"].mean())
        n_stops = int((trades["exit_reason"] == "stop").sum())
        n_long = int((trades["direction"] == 1).sum())
        n_short = int((trades["direction"] == -1).sum())
    else:
        win_rate = 0.0
        avg_pnl = 0.0
        n_stops = 0
        n_long = 0
        n_short = 0

    return {
        "final_equity": final_eq,
        "cagr": float(cagr) if not np.isnan(cagr) else 0.0,
        "sharpe": float(sharpe),
        "vol_ann": vol_ann,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "n_long": n_long,
        "n_short": n_short,
        "win_rate": float(win_rate),
        "avg_trade_pnl": avg_pnl,
        "n_stops": n_stops,
        "in_market_pct": in_market,
        "n_bars": n_bars,
    }
