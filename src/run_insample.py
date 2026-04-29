"""In-sample sweep across (timeframe, lookback, mode) configs.

Window: 2020-01-01 -> 2024-12-31 (strictly before holdout cutoff).

Usage:
    python -m src.run_insample --venue perp     # 16 configs (long_short + long_only, with funding)
    python -m src.run_insample --venue spot     # 8 configs (long_only only, no funding, 1x lev)

Outputs to data/processed/ with venue suffix on filenames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import BacktestParams, backtest, summary_stats
from src.strategy import atr, funding_per_bar, tsmom_signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

IN_SAMPLE_END = pd.Timestamp("2025-01-01", tz="UTC")
LOOKBACK_HOURS = [24, 48, 96, 168]
TIMEFRAMES = ["1h", "4h"]
BARS_PER_HOUR = {"1h": 1, "4h": 0.25}
BARS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365}


def venue_config(venue: str) -> dict:
    """Per-venue params and modes."""
    if venue == "perp":
        return {
            "params": BacktestParams(
                risk_pct=0.02, stop_atr_mult=2.0, leverage_cap=3.0,
                fee_per_side=0.0008, slippage_per_side=0.0002,
            ),
            "modes": ["long_short", "long_only"],
            "use_funding": True,
        }
    if venue == "spot":
        return {
            "params": BacktestParams(
                risk_pct=0.02, stop_atr_mult=2.0, leverage_cap=1.0,    # spot: no leverage
                fee_per_side=0.0010, slippage_per_side=0.0003,         # spot: 10 bps + 3 bps
            ),
            "modes": ["long_only"],            # spot: can't short cleanly
            "use_funding": False,
        }
    raise ValueError(f"unknown venue: {venue}")


def load(tf: str, venue: str):
    klines = pd.read_parquet(DATA_DIR / f"btcusdt_{venue}_{tf}.parquet")
    klines = klines[klines.index < IN_SAMPLE_END]
    if venue == "perp":
        fund = pd.read_parquet(DATA_DIR / "btcusdt_perp_funding.parquet")
        fund = fund[fund.index < IN_SAMPLE_END]
    else:
        fund = None
    return klines, fund


def buy_and_hold(close: pd.Series, bars_per_year: float) -> dict:
    eq = close / close.iloc[0]
    rets = eq.pct_change().fillna(0)
    cagr = eq.iloc[-1] ** (bars_per_year / len(eq)) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(bars_per_year)
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    return {"equity": eq, "cagr": float(cagr), "sharpe": float(sharpe), "max_dd": float(dd)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", choices=["perp", "spot"], default="perp")
    args = parser.parse_args()
    venue = args.venue

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = venue_config(venue)
    params = cfg["params"]
    modes = cfg["modes"]

    print(f"Venue: {venue.upper()}")
    print(f"Params: risk={params.risk_pct:.1%}/trade  "
          f"stop={params.stop_atr_mult}*ATR  "
          f"lev_cap={params.leverage_cap}x  "
          f"fee={params.fee_per_side*1e4:.0f}bps/side  "
          f"slip={params.slippage_per_side*1e4:.0f}bps/side  "
          f"funding={'on' if cfg['use_funding'] else 'off'}")
    print(f"Window: 2020-01-01 -> {IN_SAMPLE_END.date()} (strict in-sample)")
    print()

    rows = []
    equity_curves: dict[str, pd.Series] = {}

    for tf in TIMEFRAMES:
        klines, fund = load(tf, venue)
        print(f"[{tf}] loaded {len(klines):,} bars  "
              f"({klines.index.min().date()} -> {klines.index.max().date()})")

        atr_window = int(24 * BARS_PER_HOUR[tf])
        atr_s = atr(klines["high"], klines["low"], klines["close"], atr_window)
        if cfg["use_funding"]:
            fpb = funding_per_bar(fund["fundingRate"], klines.index)
        else:
            fpb = pd.Series(0.0, index=klines.index)

        for lb_hours in LOOKBACK_HOURS:
            lb_bars = int(lb_hours * BARS_PER_HOUR[tf])
            for mode in modes:
                sig = tsmom_signal(klines["close"], lb_bars, mode)
                result = backtest(
                    klines["high"], klines["low"], klines["close"],
                    sig, atr_s, fpb, params,
                )
                stats = summary_stats(
                    result["equity"], result["position"],
                    result["trades"], BARS_PER_YEAR[tf],
                )
                tag = f"{venue}_{tf}_lb{lb_hours}h_{mode}"
                stats_row = {"venue": venue, "timeframe": tf, "lookback_h": lb_hours, "mode": mode, **stats}
                rows.append(stats_row)
                equity_curves[tag] = result["equity"]

                result["equity"].to_frame().to_parquet(OUT_DIR / f"equity_{tag}.parquet")
                if len(result["trades"]):
                    result["trades"].to_parquet(OUT_DIR / f"trades_{tag}.parquet")

                print(f"  {tf} N={lb_hours:>3}h {mode:<11}  "
                      f"Sharpe={stats['sharpe']:>6.2f}  "
                      f"CAGR={stats['cagr']*100:>7.1f}%  "
                      f"DD={stats['max_dd']*100:>7.1f}%  "
                      f"trades={stats['n_trades']:>4}  "
                      f"in_mkt={stats['in_market_pct']*100:>4.0f}%")
        print()

    # Buy-and-hold benchmark
    print("Buy-and-hold benchmark (no leverage, no fees, in-sample window):")
    bh_results = {}
    for tf in TIMEFRAMES:
        klines, _ = load(tf, venue)
        bh = buy_and_hold(klines["close"], BARS_PER_YEAR[tf])
        bh_results[tf] = bh
        print(f"  {venue} {tf}: Sharpe={bh['sharpe']:>5.2f}  "
              f"CAGR={bh['cagr']*100:>6.1f}%  DD={bh['max_dd']*100:>6.1f}%")
    print()

    # Save table
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    csv_path = OUT_DIR / f"insample_results_{venue}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved sweep summary -> {csv_path.relative_to(PROJECT_ROOT)}")
    print()
    print(f"=== {venue.upper()} configs sorted by Sharpe ===")
    cols = ["timeframe", "lookback_h", "mode", "sharpe", "cagr", "max_dd",
            "vol_ann", "n_trades", "win_rate", "in_market_pct"]
    pretty = df[cols].copy()
    for c in ("cagr", "max_dd", "vol_ann", "win_rate", "in_market_pct"):
        pretty[c] = (pretty[c] * 100).round(1).astype(str) + "%"
    pretty["sharpe"] = pretty["sharpe"].round(2)
    print(pretty.to_string(index=False))

    # Plot top 3 + benchmark
    top3 = df.head(3)
    fig, ax = plt.subplots(figsize=(12, 6))
    for _, r in top3.iterrows():
        tag = f"{venue}_{r['timeframe']}_lb{int(r['lookback_h'])}h_{r['mode']}"
        ax.plot(equity_curves[tag].index, equity_curves[tag].values,
                label=f"{tag} (Sharpe={r['sharpe']:.2f}, CAGR={r['cagr']*100:.0f}%)",
                linewidth=1.2)
    top_tf = top3.iloc[0]["timeframe"]
    bh_eq = bh_results[top_tf]["equity"]
    ax.plot(bh_eq.index, bh_eq.values,
            label=f"Buy & hold {venue} {top_tf} "
                  f"(Sharpe={bh_results[top_tf]['sharpe']:.2f}, "
                  f"CAGR={bh_results[top_tf]['cagr']*100:.0f}%)",
            linewidth=1.0, color="grey", linestyle="--")
    ax.set_yscale("log")
    ax.set_title(f"In-sample equity curves on {venue.upper()}: top 3 TSMOM vs buy-and-hold")
    ax.set_ylabel("Equity (log scale, normalized to 1.0)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plot_path = OUT_DIR / f"equity_top_{venue}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)
    print()
    print(f"Saved equity plot -> {plot_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
