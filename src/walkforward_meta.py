"""Walk-forward with meta-selection across strategy classes.

For each rolling train window, picks BOTH the strategy class AND the params
that maximize train Sharpe. Applies the chosen (class, params) on the next
test window. Concatenates test pieces for an end-to-end out-of-sample equity
curve where every choice was made without seeing the future.

This is one rigor cut beyond `run_walkforward.py`, which fixed the strategy
class and only walk-forwarded the params within it.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.backtest import backtest, summary_stats
from src.run_strategy_sweep import (
    BARS_PER_HOUR, BARS_PER_YEAR, STRATEGY_GRID,
    compute_signal, load, make_bt_params, param_label,
)
from src.strategy import atr, funding_per_bar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WF_META_OUT = PROJECT_ROOT / "data" / "processed" / "walkforward_meta"

TRAIN_DAYS = 365
TEST_DAYS = 90
STEP_DAYS = 90
MIN_TRADES_FOR_PARAM = 2

WF_START = pd.Timestamp("2020-01-01", tz="UTC")
WF_END = pd.Timestamp("2025-01-01", tz="UTC")


def select_best(klines, atr_s, fpb, tf, venue, mode):
    """Pick the single (strategy, params) with the highest train Sharpe."""
    best = (None, None, -np.inf)  # (strategy, params, sharpe)
    bpy = BARS_PER_YEAR[tf]
    for strat, grid in STRATEGY_GRID.items():
        bt_params = make_bt_params(venue, strat)
        for ps in grid["params_h"]:
            params = {**ps, "mode": mode}
            sig = compute_signal(strat, params, klines, tf, mode)
            if sig is None:
                continue
            try:
                r = backtest(klines["high"], klines["low"], klines["close"],
                             sig, atr_s, fpb, bt_params)
                s = summary_stats(r["equity"], r["position"], r["trades"], bpy)
            except Exception:
                continue
            if s["n_trades"] < MIN_TRADES_FOR_PARAM:
                continue
            if s["sharpe"] > best[2]:
                best = (strat, ps, s["sharpe"])
    return best


def walk_forward_meta(klines, atr_s, fpb, tf, venue, mode):
    """Roll through (train, test) windows; meta-pick (class, params) on each train."""
    bpy = BARS_PER_YEAR[tf]
    test_pieces = []
    decisions = []

    cursor = WF_START + pd.Timedelta(days=TRAIN_DAYS)
    while cursor + pd.Timedelta(days=TEST_DAYS) <= min(WF_END, klines.index.max()):
        train_start = cursor - pd.Timedelta(days=TRAIN_DAYS)
        train_end = cursor
        test_start = cursor
        test_end = cursor + pd.Timedelta(days=TEST_DAYS)

        train_kl = klines.loc[(klines.index >= train_start) & (klines.index < train_end)]
        train_atr = atr_s.loc[train_kl.index]
        train_fpb = fpb.loc[train_kl.index] if fpb is not None else pd.Series(0.0, index=train_kl.index)
        if len(train_kl) < 200:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue

        best_strat, best_ps, train_sharpe = select_best(train_kl, train_atr, train_fpb, tf, venue, mode)
        if best_strat is None:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue

        # Test on next test_days using the chosen (strategy, params)
        # Use a 120-day warmup buffer for indicators
        ctx_start = test_start - pd.Timedelta(days=120)
        ctx_kl = klines.loc[(klines.index >= ctx_start) & (klines.index < test_end)]
        ctx_atr = atr_s.loc[ctx_kl.index]
        ctx_fpb = fpb.loc[ctx_kl.index] if fpb is not None else pd.Series(0.0, index=ctx_kl.index)
        if len(ctx_kl) < 50:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue
        sig_ctx = compute_signal(best_strat, {**best_ps, "mode": mode}, ctx_kl, tf, mode)
        if sig_ctx is None:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue
        bt_params = make_bt_params(venue, best_strat)
        try:
            r_ctx = backtest(ctx_kl["high"], ctx_kl["low"], ctx_kl["close"],
                             sig_ctx, ctx_atr, ctx_fpb, bt_params)
        except Exception:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue
        test_eq_raw = r_ctx["equity"].loc[r_ctx["equity"].index >= test_start]
        if len(test_eq_raw) < 5:
            cursor += pd.Timedelta(days=STEP_DAYS)
            continue
        test_eq = test_eq_raw / test_eq_raw.iloc[0]
        test_pieces.append(test_eq)
        decisions.append({
            "test_start": test_start, "test_end": test_end,
            "strategy": best_strat, "param_label": param_label(best_strat, best_ps),
            "train_sharpe": float(train_sharpe),
            "test_return_pct": float((test_eq.iloc[-1] - 1) * 100),
        })
        cursor += pd.Timedelta(days=STEP_DAYS)

    if not test_pieces:
        return {"wf_sharpe": np.nan, "wf_cagr": np.nan, "wf_max_dd": np.nan,
                "wf_n_bars": 0, "n_test_periods": 0,
                "decisions": pd.DataFrame(), "equity": pd.Series(dtype="float64")}

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
        "wf_max_dd": max_dd, "wf_final_eq": final, "wf_n_bars": n,
        "n_test_periods": len(test_pieces),
        "decisions": pd.DataFrame(decisions),
        "equity": full_eq,
    }


def main():
    WF_META_OUT.mkdir(parents=True, exist_ok=True)
    (WF_META_OUT / "equity").mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    # Focus on configurations that meet user's "deployable manually" criterion.
    # Daily timeframe is realistic for SGT-based manual execution.
    targets = [
        ("BTCUSDT", "1d", "spot", "long_only"),
        ("ETHUSDT", "1d", "spot", "long_only"),
        ("SOLUSDT", "1d", "spot", "long_only"),
        # 4H as a higher-frequency option (more decisions/day but still doable)
        ("BTCUSDT", "4h", "spot", "long_only"),
        ("ETHUSDT", "4h", "spot", "long_only"),
        ("SOLUSDT", "4h", "spot", "long_only"),
    ]

    for asset, tf, venue, mode in targets:
        try:
            klines, fund = load(asset, tf, venue)
        except FileNotFoundError:
            print(f"[skip] no {venue} {asset} {tf}")
            continue
        klines = klines.loc[(klines.index >= WF_START - pd.Timedelta(days=30)) & (klines.index < WF_END)]
        if len(klines) < TRAIN_DAYS + TEST_DAYS:
            continue
        atr_window = max(2, int(round(24 * BARS_PER_HOUR[tf])))
        atr_s = atr(klines["high"], klines["low"], klines["close"], atr_window)
        if venue == "perp" and fund is not None:
            fpb = funding_per_bar(fund["fundingRate"], klines.index)
        else:
            fpb = pd.Series(0.0, index=klines.index)

        res = walk_forward_meta(klines, atr_s, fpb, tf, venue, mode)
        cfg = f"wfm_{venue}_{asset}_{tf}_{mode}"
        rows.append({"asset": asset, "timeframe": tf, "venue": venue, "mode": mode,
                     **{k: v for k, v in res.items() if k not in ("decisions", "equity")}})
        print(f"  {cfg:<40}  WF_meta Sharpe={res['wf_sharpe']:>5.2f}  "
              f"CAGR={(res['wf_cagr'] or 0)*100:>6.1f}%  DD={res['wf_max_dd']*100:>6.1f}%  "
              f"periods={res['n_test_periods']}")
        if not res["decisions"].empty:
            res["decisions"].to_csv(WF_META_OUT / f"{cfg}_decisions.csv", index=False)
            res["equity"].to_frame("equity").to_parquet(WF_META_OUT / "equity" / f"{cfg}.parquet")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows).sort_values("wf_sharpe", ascending=False)
    df.to_csv(WF_META_OUT / "wfm_results.csv", index=False)
    print(f"\nWF-meta on {len(rows)} combos in {elapsed:.1f}s")
    print(f"Saved -> {(WF_META_OUT / 'wfm_results.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
