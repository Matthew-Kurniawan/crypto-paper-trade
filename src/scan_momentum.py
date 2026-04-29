"""Scan a universe of crypto symbols for current momentum signals.

For each symbol:
1. Fetch / cache daily klines (1d).
2. Retrain on the trailing 365 days — pick best (strategy class, params) by Sharpe.
3. Compute the current signal at the most recent bar.
4. Apply the same safety gate as live_signal.py (min train Sharpe).
5. Compute trailing 30/60-day return as an independent momentum proxy.
6. Rank: 'recommendable' = passes safety AND currently LONG.

Output: data/processed/scan/momentum_scan_<date>.csv + a printed table.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.fetch_data import update_klines
from src.live_signal import MIN_TRAIN_SHARPE, TRAIN_DAYS, select_best_on_train
from src.run_strategy_sweep import BARS_PER_HOUR, compute_signal, param_label
from src.strategy import atr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCAN_OUT = PROJECT_ROOT / "data" / "processed" / "scan"

# Curated universe — mix of large, mid, and newer/smaller. Some may not exist
# on Binance spot or have insufficient history; the scan handles that gracefully.
UNIVERSE = [
    # Large cap
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    # Mid-large
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "DOGEUSDT",
    "TRXUSDT", "LTCUSDT", "BCHUSDT",
    # Mid cap
    "NEARUSDT", "ATOMUSDT", "FILUSDT", "ETCUSDT", "ICPUSDT",
    "ALGOUSDT", "HBARUSDT", "AAVEUSDT", "UNIUSDT",
    # Newer / smaller (post-2022)
    "INJUSDT", "SUIUSDT", "TIAUSDT", "SEIUSDT", "JUPUSDT",
    "OPUSDT", "ARBUSDT", "TAOUSDT", "RUNEUSDT",
    # Memes (with appropriate skepticism)
    "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "FLOKIUSDT",
    # Other
    "RNDRUSDT", "FETUSDT",
]

TF = "1d"
VENUE = "spot"
MODE = "long_only"


def scan_one(symbol: str) -> dict | None:
    """Return scan result for one symbol, or None if data unavailable."""
    try:
        df = update_klines(symbol, TF, venue=VENUE)
    except Exception as e:
        return {"symbol": symbol, "ok": False, "reason": f"fetch_failed: {e}"}
    if df is None or df.empty:
        return {"symbol": symbol, "ok": False, "reason": "no_data"}
    if len(df) < 400:
        return {"symbol": symbol, "ok": False, "reason": f"insufficient_history ({len(df)} bars)"}

    last_close = df.index.max()
    train_start = last_close - pd.Timedelta(days=TRAIN_DAYS)
    train = df.loc[(df.index >= train_start) & (df.index <= last_close)]
    if len(train) < 200:
        return {"symbol": symbol, "ok": False, "reason": "insufficient_train_window"}

    atr_window = max(2, int(round(24 * BARS_PER_HOUR[TF])))
    atr_s = atr(train["high"], train["low"], train["close"], atr_window)
    fpb = pd.Series(0.0, index=train.index)

    strat, ps, train_sharpe, stats = select_best_on_train(train, atr_s, fpb, TF, VENUE, MODE)
    if strat is None:
        return {"symbol": symbol, "ok": False, "reason": "no_param_passed_filter"}

    sig = compute_signal(strat, {**ps, "mode": MODE}, train, TF, MODE)
    cur_signal = int(sig.iloc[-1])
    prior_signal = int(sig.iloc[-2]) if len(sig) >= 2 else 0

    cur_close = float(train["close"].iloc[-1])

    def trail_ret(n_bars):
        if len(train) <= n_bars:
            return None
        return float(train["close"].iloc[-1] / train["close"].iloc[-1 - n_bars] - 1)

    rets_60d = train["close"].pct_change().tail(60).dropna()
    realized_vol_60d = float(rets_60d.std() * np.sqrt(365)) if len(rets_60d) > 5 else float("nan")

    return {
        "symbol": symbol,
        "ok": True,
        "as_of": last_close.isoformat(),
        "cur_close": cur_close,
        "strategy": strat,
        "param_label": param_label(strat, ps),
        "train_sharpe": float(train_sharpe),
        "train_n_trades": int(stats["n_trades"]),
        "train_in_market_pct": float(stats["in_market_pct"]),
        "current_signal": cur_signal,
        "prior_signal": prior_signal,
        "is_flip": cur_signal != prior_signal,
        "ret_30d": trail_ret(30),
        "ret_60d": trail_ret(60),
        "ret_90d": trail_ret(90),
        "realized_vol_60d": realized_vol_60d,
        "passes_safety": train_sharpe >= MIN_TRAIN_SHARPE,
        "recommendable": (train_sharpe >= MIN_TRAIN_SHARPE) and (cur_signal == 1),
    }


def main():
    SCAN_OUT.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rows = []
    for sym in UNIVERSE:
        print(f"  scanning {sym} ...", end="", flush=True)
        r = scan_one(sym)
        rows.append(r)
        if r and r.get("ok"):
            sig_str = {1: "LONG", 0: "FLAT", -1: "SHORT"}[r["current_signal"]]
            flag = "[REC]" if r["recommendable"] else ("[OK ]" if r["passes_safety"] else "[GATE]")
            print(f" {flag} {r['strategy']:<8} {r['param_label']:<14} "
                  f"Sharpe={r['train_sharpe']:>5.2f} sig={sig_str:<5} "
                  f"60d={(r['ret_60d'] or 0)*100:>+6.1f}%")
        else:
            print(f"   skipped: {r.get('reason') if r else 'none'}")

    df = pd.DataFrame(rows)
    df_ok = df[df["ok"] == True].copy()
    df_failed = df[df["ok"] != True]

    if not df_ok.empty:
        df_ok = df_ok.sort_values(
            by=["recommendable", "passes_safety", "train_sharpe"],
            ascending=[False, False, False],
        )

    df_ok.to_csv(SCAN_OUT / f"momentum_scan_{today}.csv", index=False)

    print()
    print("=" * 75)
    print(f"Scan summary ({len(df_ok)} symbols with data, {len(df_failed)} skipped)")
    print("=" * 75)

    if not df_ok.empty:
        cols = ["symbol", "strategy", "param_label", "train_sharpe", "current_signal",
                "ret_30d", "ret_60d", "ret_90d", "realized_vol_60d",
                "passes_safety", "recommendable"]
        pretty = df_ok[cols].copy()
        for c in ("ret_30d", "ret_60d", "ret_90d", "realized_vol_60d"):
            pretty[c] = pretty[c].apply(lambda x: f"{x*100:+6.1f}%" if pd.notna(x) else "  n/a")
        pretty["train_sharpe"] = pretty["train_sharpe"].round(2)
        pretty["current_signal"] = pretty["current_signal"].map({1: "LONG", 0: "FLAT", -1: "SHORT"})
        print(pretty.to_string(index=False))

    recs = df_ok[df_ok["recommendable"] == True]
    print()
    if len(recs):
        print(f">>> {len(recs)} recommendable candidates (passed safety AND currently LONG):")
        print(", ".join(recs["symbol"].tolist()))
    else:
        print(">>> 0 recommendable candidates today.")
        print("    All symbols either failed the trailing-Sharpe safety gate or are FLAT/SHORT.")
        print("    Possible interpretations:")
        print("      1. Crypto market is in a regime where trend-following has been weak.")
        print("      2. Most assets are in correction or consolidation.")
        print("      3. Wait — when momentum returns, the safety gate will release automatically.")

    print(f"\nFull scan saved -> data/processed/scan/momentum_scan_{today}.csv")


if __name__ == "__main__":
    main()
