"""Walk-forward validation: rolling 1-year train -> 3-month test, with param selection on train.

For each (asset, timeframe, venue, mode, strategy) combo, picks the best-Sharpe param
on the trailing 365-day window, evaluates that param on the next 90 days, then rolls.
The concatenated test pieces are the walk-forward equity curve.

This removes the in-sample param-selection bias from the headline sweep.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.backtest import backtest, summary_stats
from src.run_strategy_sweep import (
    ASSETS, BARS_PER_HOUR, BARS_PER_YEAR, STRATEGY_GRID, TIMEFRAMES,
    compute_signal, load, make_bt_params, param_label,
)
from src.strategy import atr, funding_per_bar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_OUT = PROJECT_ROOT / "data" / "processed" / "sweep"
WF_OUT = PROJECT_ROOT / "data" / "processed" / "walkforward"

TRAIN_DAYS = 365
TEST_DAYS = 90
STEP_DAYS = 90

WF_START = pd.Timestamp("2020-01-01", tz="UTC")
WF_END = pd.Timestamp("2025-01-01", tz="UTC")


def walk_forward_strategy(
    klines: pd.DataFrame,
    atr_s: pd.Series,
    fpb: pd.Series,
    strat: str,
    tf: str,
    venue: str,
    mode: str,
) -> dict:
    """Run walk-forward for one (strategy, asset, tf, venue, mode) combo."""
    bt_params = make_bt_params(venue, strat)
    grid = STRATEGY_GRID[strat]["params_h"]
    bpy = BARS_PER_YEAR[tf]

    cursor = WF_START + pd.Timedelta(days=TRAIN_DAYS)
    test_pieces = []
    decisions = []

    while cursor + pd.Timedelta(days=TEST_DAYS) <= min(WF_END, klines.index.max()):
        train_start = cursor - pd.Timedelta(days=TRAIN_DAYS)
        train_end = cursor
        test_start = cursor
        test_end = cursor + pd.Timedelta(days=TEST_DAYS)

        train_kl = klines.loc[(klines.index >= train_start) & (klines.index < train_end)]
        train_atr = atr_s.loc[train_kl.index]
        train_fpb = fpb.loc[train_kl.index]
        if len(train_kl) < 200:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue

        # Param selection on train
        best_ps = None
        best_sharpe = -np.inf
        for ps in grid:
            params = {**ps, "mode": mode}
            sig = compute_signal(strat, params, train_kl, tf, mode)
            if sig is None:
                continue
            try:
                r = backtest(train_kl["high"], train_kl["low"], train_kl["close"],
                             sig, train_atr, train_fpb, bt_params)
                s = summary_stats(r["equity"], r["position"], r["trades"], bpy)
            except Exception:
                continue
            if s["n_trades"] < 2:
                continue
            if s["sharpe"] > best_sharpe:
                best_sharpe = s["sharpe"]
                best_ps = ps

        if best_ps is None:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue

        # Evaluate on test
        # Use a slightly wider slice so the signal has lookback context, but only score on test window
        ctx_start = test_start - pd.Timedelta(days=120)  # warmup buffer for indicators
        ctx_kl = klines.loc[(klines.index >= ctx_start) & (klines.index < test_end)]
        ctx_atr = atr_s.loc[ctx_kl.index]
        ctx_fpb = fpb.loc[ctx_kl.index]
        if len(ctx_kl) < 50:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue
        sig_ctx = compute_signal(strat, {**best_ps, "mode": mode}, ctx_kl, tf, mode)
        if sig_ctx is None:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue
        try:
            r_ctx = backtest(ctx_kl["high"], ctx_kl["low"], ctx_kl["close"],
                             sig_ctx, ctx_atr, ctx_fpb, bt_params)
        except Exception:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue
        # Slice to test window only
        test_eq_raw = r_ctx["equity"].loc[r_ctx["equity"].index >= test_start]
        if len(test_eq_raw) < 5:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue
        # Re-normalize so each test piece starts at 1.0 then we'll chain
        test_eq = test_eq_raw / test_eq_raw.iloc[0]
        test_pieces.append(test_eq)
        decisions.append({
            "test_start": test_start, "test_end": test_end,
            "train_sharpe": best_sharpe,
            "param_label": param_label(strat, best_ps),
            "test_final_eq": float(test_eq.iloc[-1]),
            "test_return_pct": float((test_eq.iloc[-1] - 1) * 100),
        })

        cursor += pd.Timedelta(days=STEP_DAYS)

    if not test_pieces:
        return {"strategy": strat, "asset": None, "wf_sharpe": np.nan,
                "wf_cagr": np.nan, "wf_max_dd": np.nan, "n_test_periods": 0,
                "decisions": pd.DataFrame()}

    # Chain test pieces: each starts at 1.0; multiply by running equity
    chained = []
    cur = 1.0
    for piece in test_pieces:
        scaled = piece * cur
        chained.append(scaled)
        cur = float(scaled.iloc[-1])
    full_eq = pd.concat(chained)
    full_eq = full_eq[~full_eq.index.duplicated(keep="last")].sort_index()

    rets = full_eq.pct_change().fillna(0)
    n = len(full_eq)
    years = n / bpy
    final = float(full_eq.iloc[-1])
    cagr = final ** (1 / years) - 1 if final > 0 and years > 0 else np.nan
    sharpe = (rets.mean() / rets.std()) * np.sqrt(bpy) if rets.std() > 0 else 0.0
    rmax = full_eq.cummax()
    max_dd = float(((full_eq - rmax) / rmax).min())

    return {
        "wf_sharpe": float(sharpe),
        "wf_cagr": float(cagr) if not np.isnan(cagr) else np.nan,
        "wf_max_dd": max_dd,
        "wf_final_eq": final,
        "wf_n_bars": n,
        "n_test_periods": len(test_pieces),
        "decisions": pd.DataFrame(decisions),
        "equity": full_eq,
    }


def main():
    WF_OUT.mkdir(parents=True, exist_ok=True)
    (WF_OUT / "equity").mkdir(parents=True, exist_ok=True)

    venues_modes = [("spot", "long_only"), ("perp", "long_short")]

    rows = []
    t0 = time.time()
    n_combos = 0

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            for venue, mode in venues_modes:
                try:
                    klines, fund = load(asset, tf, venue)
                except FileNotFoundError:
                    continue
                # constrain to walk-forward window
                klines = klines.loc[(klines.index >= WF_START - pd.Timedelta(days=30)) & (klines.index < WF_END)]
                if len(klines) < TRAIN_DAYS + TEST_DAYS:
                    continue

                atr_window = max(2, int(round(24 * BARS_PER_HOUR[tf])))
                atr_s = atr(klines["high"], klines["low"], klines["close"], atr_window)
                if venue == "perp" and fund is not None:
                    fpb = funding_per_bar(fund["fundingRate"], klines.index)
                else:
                    fpb = pd.Series(0.0, index=klines.index)

                for strat in STRATEGY_GRID:
                    n_combos += 1
                    res = walk_forward_strategy(klines, atr_s, fpb, strat, tf, venue, mode)
                    cfg_id = f"wf_{venue}_{asset}_{tf}_{strat}_{mode}"
                    print(f"  {cfg_id:<55}  WF Sharpe={res['wf_sharpe']:>5.2f}  "
                          f"CAGR={(res['wf_cagr'] or 0)*100:>6.1f}%  "
                          f"DD={res['wf_max_dd']*100:>6.1f}%  "
                          f"periods={res['n_test_periods']}")
                    rows.append({
                        "asset": asset, "timeframe": tf, "venue": venue,
                        "mode": mode, "strategy": strat,
                        "wf_sharpe": res["wf_sharpe"], "wf_cagr": res["wf_cagr"],
                        "wf_max_dd": res["wf_max_dd"],
                        "wf_final_eq": res.get("wf_final_eq"),
                        "n_test_periods": res["n_test_periods"],
                    })
                    if res["n_test_periods"] > 0 and not res["decisions"].empty:
                        res["decisions"].to_csv(WF_OUT / f"{cfg_id}_decisions.csv", index=False)
                        if "equity" in res:
                            res["equity"].to_frame("equity").to_parquet(WF_OUT / "equity" / f"{cfg_id}.parquet")

    elapsed = time.time() - t0
    print(f"\nWalk-forward across {n_combos} combos in {elapsed:.1f}s")

    df = pd.DataFrame(rows).sort_values("wf_sharpe", ascending=False)
    df.to_csv(WF_OUT / "wf_results.csv", index=False)

    print(f"\nSaved -> {(WF_OUT / 'wf_results.csv').relative_to(PROJECT_ROOT)}")
    print()
    print("=== Top 15 by walk-forward Sharpe ===")
    pretty = df.head(15).copy()
    for c in ("wf_cagr", "wf_max_dd"):
        pretty[c] = (pretty[c] * 100).round(1).astype(str) + "%"
    pretty["wf_sharpe"] = pretty["wf_sharpe"].round(2)
    print(pretty[["asset", "timeframe", "venue", "mode", "strategy",
                  "wf_sharpe", "wf_cagr", "wf_max_dd", "n_test_periods"]].to_string(index=False))


if __name__ == "__main__":
    main()
