"""Multi-strategy sweep across assets, timeframes, time windows, and params.

Spot, long-only by default (cleanest signal-alpha test). Use --include-perp
to also sweep the long-short capable strategies on perp with funding modeled.

Outputs to data/processed/sweep/:
- sweep_results.csv          — flat row per (asset, timeframe, window, strategy, params)
- equity/<config_id>.parquet — equity curve per config (selectively saved)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import BacktestParams, backtest, summary_stats
from src.strategy import (
    atr,
    donchian_signal,
    funding_per_bar,
    ma_cross_signal,
    meanrev_zscore_signal,
    tsmom_signal,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SWEEP_OUT = PROJECT_ROOT / "data" / "processed" / "sweep"

# ----- Sweep matrix configuration -----

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["1h", "4h", "1d"]
BARS_PER_HOUR = {"1h": 1, "4h": 0.25, "1d": 1.0 / 24}
BARS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}

WINDOWS = {
    # Window name -> (start, end) inclusive of start, exclusive of end
    "2020_2024": ("2020-01-01", "2025-01-01"),
    "2022_2024": ("2022-01-01", "2025-01-01"),
}

# Strategy parameter sweeps. Lookbacks specified in HOURS, converted per timeframe.
# Each timeframe gets a list of param dicts. We enumerate the cross product (strategy, params, mode).
STRATEGY_GRID = {
    "tsmom": {
        "params_h": [
            {"lookback_h": 24}, {"lookback_h": 48}, {"lookback_h": 96},
            {"lookback_h": 168}, {"lookback_h": 336}, {"lookback_h": 720},
        ],
        # On 1d bars, 24h lookback = 1 bar (degenerate). Skip lookbacks < 2 bars.
        "min_bars": 2,
    },
    "donchian": {
        "params_h": [
            {"n_entry_h": 24, "n_exit_h": 12},
            {"n_entry_h": 48, "n_exit_h": 24},
            {"n_entry_h": 96, "n_exit_h": 48},
            {"n_entry_h": 168, "n_exit_h": 96},
            {"n_entry_h": 336, "n_exit_h": 168},
            {"n_entry_h": 720, "n_exit_h": 336},
        ],
        "min_bars": 3,
    },
    "meanrev": {
        "params_h": [
            {"window_h": 48,  "entry_z": 2.0, "exit_z": 0.0},
            {"window_h": 96,  "entry_z": 2.0, "exit_z": 0.0},
            {"window_h": 168, "entry_z": 2.0, "exit_z": 0.0},
            {"window_h": 336, "entry_z": 2.0, "exit_z": 0.0},
            {"window_h": 96,  "entry_z": 2.5, "exit_z": 0.0},
            {"window_h": 168, "entry_z": 2.5, "exit_z": 0.5},
        ],
        "min_bars": 5,
    },
    "macross": {
        "params_h": [
            {"fast_h": 24,  "slow_h": 96},
            {"fast_h": 48,  "slow_h": 168},
            {"fast_h": 96,  "slow_h": 336},
            {"fast_h": 168, "slow_h": 720},
        ],
        "min_bars": 5,
    },
}

# Per-strategy backtest config (stop multiplier — momentum strategies use tight stops,
# mean-reversion uses wide stops since the position is *betting against* a recent move).
BACKTEST_CFG = {
    "tsmom":    {"stop_atr_mult": 2.0},
    "donchian": {"stop_atr_mult": 3.0},
    "meanrev":  {"stop_atr_mult": 4.0},
    "macross":  {"stop_atr_mult": 2.0},
}


def make_bt_params(venue: str, strat: str) -> BacktestParams:
    """Per-(venue, strategy) BacktestParams."""
    if venue == "spot":
        return BacktestParams(
            risk_pct=0.02, stop_atr_mult=BACKTEST_CFG[strat]["stop_atr_mult"],
            leverage_cap=1.0, fee_per_side=0.0010, slippage_per_side=0.0003,
        )
    # perp
    return BacktestParams(
        risk_pct=0.02, stop_atr_mult=BACKTEST_CFG[strat]["stop_atr_mult"],
        leverage_cap=3.0, fee_per_side=0.0008, slippage_per_side=0.0002,
    )


def hours_to_bars(hours: float, tf: str) -> int:
    return max(1, int(round(hours * BARS_PER_HOUR[tf])))


def compute_signal(strat: str, params: dict, klines: pd.DataFrame, tf: str, mode: str) -> pd.Series | None:
    """Build the {-1,0,+1} signal for a strategy + param spec, or None if degenerate at this tf."""
    grid = STRATEGY_GRID[strat]
    min_bars = grid["min_bars"]
    if strat == "tsmom":
        lb = hours_to_bars(params["lookback_h"], tf)
        if lb < min_bars:
            return None
        return tsmom_signal(klines["close"], lb, mode=mode)
    if strat == "donchian":
        n_in = hours_to_bars(params["n_entry_h"], tf)
        n_out = hours_to_bars(params["n_exit_h"], tf)
        if n_in < min_bars or n_out < min_bars or n_out >= n_in:
            return None
        return donchian_signal(klines["high"], klines["low"], klines["close"],
                               n_in, n_out, mode=mode)
    if strat == "meanrev":
        w = hours_to_bars(params["window_h"], tf)
        if w < min_bars:
            return None
        return meanrev_zscore_signal(
            klines["close"], w,
            entry_z=params["entry_z"], exit_z=params["exit_z"], mode=mode,
        )
    if strat == "macross":
        fast = hours_to_bars(params["fast_h"], tf)
        slow = hours_to_bars(params["slow_h"], tf)
        if fast < min_bars or slow <= fast:
            return None
        return ma_cross_signal(klines["close"], fast, slow, mode=mode)
    raise ValueError(strat)


def param_label(strat: str, params: dict) -> str:
    if strat == "tsmom":
        return f"lb{params['lookback_h']}h"
    if strat == "donchian":
        return f"in{params['n_entry_h']}h_out{params['n_exit_h']}h"
    if strat == "meanrev":
        return f"w{params['window_h']}h_z{params['entry_z']}_x{params['exit_z']}"
    if strat == "macross":
        return f"f{params['fast_h']}h_s{params['slow_h']}h"
    raise ValueError(strat)


def load(asset: str, tf: str, venue: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    klines = pd.read_parquet(DATA_DIR / f"{asset.lower()}_{venue}_{tf}.parquet")
    fund = None
    if venue == "perp":
        fpath = DATA_DIR / f"{asset.lower()}_perp_funding.parquet"
        if fpath.exists():
            fund = pd.read_parquet(fpath)
    return klines, fund


def buy_and_hold_metrics(close: pd.Series, bars_per_year: float) -> dict:
    eq = close / close.iloc[0]
    rets = eq.pct_change().fillna(0)
    cagr = eq.iloc[-1] ** (bars_per_year / len(eq)) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(bars_per_year) if rets.std() > 0 else 0.0
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    return {"sharpe": float(sharpe), "cagr": float(cagr), "max_dd": float(dd),
            "vol_ann": float(rets.std() * np.sqrt(bars_per_year)),
            "final_eq": float(eq.iloc[-1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-perp", action="store_true",
                        help="Also sweep long_short on perp (with funding)")
    parser.add_argument("--save-equity-top", type=int, default=20,
                        help="How many top configs (by Sharpe) to save equity curves for")
    args = parser.parse_args()

    SWEEP_OUT.mkdir(parents=True, exist_ok=True)
    (SWEEP_OUT / "equity").mkdir(parents=True, exist_ok=True)

    rows = []
    bh_rows = []
    equity_curves = {}  # config_id -> equity Series, kept in memory for top-K save
    t0 = time.time()
    n_done = 0

    venues_modes = [("spot", "long_only")]
    if args.include_perp:
        venues_modes.append(("perp", "long_short"))

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            for venue, mode in venues_modes:
                try:
                    klines_full, fund_full = load(asset, tf, venue)
                except FileNotFoundError:
                    print(f"  [skip] no {venue} {asset} {tf}")
                    continue
                # Pre-compute ATR(24h) once (used by backtester for stops)
                atr_window = max(2, int(round(24 * BARS_PER_HOUR[tf])))
                atr_full = atr(klines_full["high"], klines_full["low"],
                               klines_full["close"], atr_window)

                for window_name, (w_start, w_end) in WINDOWS.items():
                    s = pd.Timestamp(w_start, tz="UTC")
                    e = pd.Timestamp(w_end, tz="UTC")
                    klines = klines_full.loc[(klines_full.index >= s) & (klines_full.index < e)]
                    if len(klines) < 200:
                        continue
                    atr_s = atr_full.loc[klines.index]
                    if venue == "perp" and fund_full is not None:
                        fpb = funding_per_bar(
                            fund_full.loc[(fund_full.index >= s) & (fund_full.index < e)]["fundingRate"],
                            klines.index,
                        )
                    else:
                        fpb = pd.Series(0.0, index=klines.index)

                    # Buy and hold benchmark for this slice
                    bh = buy_and_hold_metrics(klines["close"], BARS_PER_YEAR[tf])
                    bh_rows.append({
                        "asset": asset, "timeframe": tf, "venue": venue,
                        "window": window_name, **bh,
                        "n_bars": len(klines),
                    })

                    for strat, grid in STRATEGY_GRID.items():
                        bt_params = make_bt_params(venue, strat)
                        for ps in grid["params_h"]:
                            params = {**ps, "mode": mode}
                            sig = compute_signal(strat, params, klines, tf, mode)
                            if sig is None:
                                continue
                            try:
                                result = backtest(
                                    klines["high"], klines["low"], klines["close"],
                                    sig, atr_s, fpb, bt_params,
                                )
                                stats = summary_stats(
                                    result["equity"], result["position"],
                                    result["trades"], BARS_PER_YEAR[tf],
                                )
                            except Exception as ex:
                                print(f"  ERROR {asset} {tf} {venue} {window_name} {strat} {params}: {ex}")
                                continue
                            cfg_id = (
                                f"{venue}_{asset}_{tf}_{window_name}_{strat}_{param_label(strat, ps)}_{mode}"
                            )
                            row = {
                                "config_id": cfg_id,
                                "venue": venue, "asset": asset, "timeframe": tf,
                                "window": window_name, "strategy": strat,
                                "param_label": param_label(strat, ps),
                                "mode": mode,
                                **{f"p_{k}": v for k, v in ps.items()},
                                **stats,
                                "bh_sharpe": bh["sharpe"], "bh_cagr": bh["cagr"], "bh_max_dd": bh["max_dd"],
                                "alpha_sharpe": stats["sharpe"] - bh["sharpe"],
                                "alpha_cagr": stats["cagr"] - bh["cagr"],
                            }
                            rows.append(row)
                            equity_curves[cfg_id] = result["equity"]
                            n_done += 1

    elapsed = time.time() - t0
    print(f"\nRan {n_done:,} backtests in {elapsed:.1f}s ({n_done/elapsed:.1f} /s)")

    df = pd.DataFrame(rows)
    df_bh = pd.DataFrame(bh_rows)

    # Save top-K equity curves (by Sharpe)
    df_sorted = df.sort_values("sharpe", ascending=False)
    keep = set(df_sorted.head(args.save_equity_top)["config_id"].tolist())
    # Always save best of each (asset, strategy) combo too
    for _, g in df.groupby(["asset", "strategy"]):
        keep.add(g.sort_values("sharpe", ascending=False).iloc[0]["config_id"])
    for cfg_id in keep:
        if cfg_id in equity_curves:
            equity_curves[cfg_id].to_frame().to_parquet(SWEEP_OUT / "equity" / f"{cfg_id}.parquet")

    df.to_csv(SWEEP_OUT / "sweep_results.csv", index=False)
    df_bh.to_csv(SWEEP_OUT / "buy_and_hold.csv", index=False)

    print(f"\nSaved sweep results -> {(SWEEP_OUT / 'sweep_results.csv').relative_to(PROJECT_ROOT)}")
    print(f"Saved buy-and-hold benchmarks -> {(SWEEP_OUT / 'buy_and_hold.csv').relative_to(PROJECT_ROOT)}")
    print(f"Saved {len(keep)} equity curves -> {(SWEEP_OUT / 'equity').relative_to(PROJECT_ROOT)}")

    # Quick top-10 print
    print("\n=== Top 10 by Sharpe (raw, before walk-forward) ===")
    cols = ["asset", "timeframe", "window", "strategy", "param_label", "mode",
            "sharpe", "cagr", "max_dd", "n_trades", "alpha_sharpe", "alpha_cagr"]
    pretty = df_sorted.head(10)[cols].copy()
    for c in ("cagr", "max_dd", "alpha_cagr"):
        pretty[c] = (pretty[c] * 100).round(1).astype(str) + "%"
    pretty["sharpe"] = pretty["sharpe"].round(2)
    pretty["alpha_sharpe"] = pretty["alpha_sharpe"].round(2)
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
