"""Multi-asset risk-parity portfolio from walk-forward meta-strategy tracks.

Inputs: WF-meta equity curves per (asset, timeframe) from `walkforward_meta.py`.

Method:
1. For each timeframe (1d, 4h), align asset tracks on a common index.
2. Compute rolling realized vol per track (default: 63-bar window, annualized).
3. Inverse-vol weights, re-normalized to sum to 1.0 each bar.
4. Portfolio return = sum(prior_weight_i * track_return_i).
5. Portfolio equity = cumprod of portfolio returns.

Benchmark: vol-targeted equal-weight basket of (BTC, ETH, SOL) at the same
   timeframe and over the same date range. This is the 'smart passive' control.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.vol_target_bh import equal_weight_basket_vol_target, summary_metrics, vol_target_equity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WF_META = PROJECT_ROOT / "data" / "processed" / "walkforward_meta"
DATA = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "processed" / "portfolio"

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["1d", "4h"]
BARS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}
VOL_WINDOW_BARS = {"1d": 63, "4h": 63 * 6, "1h": 63 * 24}  # ~3 months
VOL_TARGET = 0.30  # 30% annualized portfolio target
MAX_LEVERAGE = 3.0


def load_wf_track(asset: str, tf: str) -> pd.Series | None:
    p = WF_META / "equity" / f"wfm_spot_{asset}_{tf}_long_only.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)["equity"].rename(asset)


def load_close(asset: str, tf: str) -> pd.Series:
    return pd.read_parquet(DATA / f"{asset.lower()}_spot_{tf}.parquet")["close"].rename(asset)


def inv_vol_portfolio(track_equities: dict[str, pd.Series], vol_window: int, bpy: float) -> tuple[pd.Series, pd.DataFrame]:
    """Combine asset tracks into a single portfolio via rolling inverse-vol weights."""
    df = pd.DataFrame(track_equities).dropna(how="any")
    if df.empty:
        return pd.Series(dtype="float64"), pd.DataFrame()
    track_rets = df.pct_change().fillna(0)
    rolling_vols = track_rets.rolling(vol_window).std() * np.sqrt(bpy)
    inv_vols = 1.0 / rolling_vols.replace(0, np.nan)
    weights = inv_vols.div(inv_vols.sum(axis=1), axis=0)
    # Before vol window has matured, equal-weight
    n_assets = df.shape[1]
    weights = weights.fillna(1.0 / n_assets)
    pf_rets = (weights.shift(1).fillna(1.0 / n_assets) * track_rets).sum(axis=1)
    pf_eq = (1 + pf_rets).cumprod().rename("portfolio_equity")
    return pf_eq, weights


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for tf in TIMEFRAMES:
        # Load WF-meta tracks per asset
        tracks: dict[str, pd.Series] = {}
        for asset in ASSETS:
            t = load_wf_track(asset, tf)
            if t is not None and not t.empty:
                tracks[asset] = t
        if not tracks:
            continue

        # Risk-parity portfolio
        pf_eq, weights = inv_vol_portfolio(tracks, VOL_WINDOW_BARS[tf], BARS_PER_YEAR[tf])
        if pf_eq.empty:
            continue

        # Save
        pf_eq.to_frame("equity").to_parquet(OUT / f"portfolio_{tf}.parquet")
        weights.to_parquet(OUT / f"weights_{tf}.parquet")

        # Vol-targeted equal-weight basket benchmark over the same range
        common_index = pf_eq.index
        closes = {a: load_close(a, tf).reindex(common_index).ffill() for a in tracks}
        basket_eq = equal_weight_basket_vol_target(
            closes, vol_target=VOL_TARGET, vol_window_bars=VOL_WINDOW_BARS[tf],
            bars_per_year=BARS_PER_YEAR[tf], max_leverage=MAX_LEVERAGE,
        )
        basket_eq = basket_eq.reindex(common_index).ffill()

        # Naive equal-weight buy-and-hold benchmark (no leverage scaling)
        eq_basket_naive = sum(closes[a] / closes[a].iloc[0] for a in closes) / len(closes)
        eq_basket_naive = eq_basket_naive.reindex(common_index).ffill().rename("naive_basket")

        # Per-asset vol-targeted B&H for comparison
        per_asset_vt = {}
        for a in tracks:
            per_asset_vt[a] = vol_target_equity(
                closes[a], vol_target=VOL_TARGET, vol_window_bars=VOL_WINDOW_BARS[tf],
                bars_per_year=BARS_PER_YEAR[tf], max_leverage=MAX_LEVERAGE,
            ).reindex(common_index).ffill()

        # Metrics
        def m(eq, label):
            s = summary_metrics(eq, BARS_PER_YEAR[tf])
            return {"timeframe": tf, "label": label, **s}

        summary_rows.append(m(pf_eq, f"portfolio_riskparity_{tf}"))
        summary_rows.append(m(basket_eq, f"vol_target_basket_{tf}"))
        summary_rows.append(m(eq_basket_naive, f"naive_eq_basket_{tf}"))
        for a in tracks:
            summary_rows.append(m(tracks[a].reindex(common_index).ffill(), f"track_{a}_{tf}"))
            summary_rows.append(m(per_asset_vt[a], f"voltarget_{a}_{tf}"))

        # Plot
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.plot(pf_eq.index, pf_eq.values, label="Risk-parity portfolio (WF-meta tracks)", linewidth=1.6, color="C0")
        ax.plot(basket_eq.index, basket_eq.values, label=f"Vol-target B&H basket (target {VOL_TARGET:.0%})", linewidth=1.2, color="C2", linestyle="--")
        ax.plot(eq_basket_naive.index, eq_basket_naive.values, label="Naive equal-weight B&H", linewidth=1.0, color="grey", linestyle=":")
        for i, a in enumerate(tracks):
            ax.plot(tracks[a].reindex(common_index).ffill().index,
                    tracks[a].reindex(common_index).ffill().values,
                    label=f"WF-meta {a}", linewidth=0.8, alpha=0.7, color=f"C{3+i}")
        ax.set_yscale("log")
        ax.set_title(f"Multi-asset risk-parity portfolio vs benchmarks ({tf})")
        ax.set_ylabel("Equity (log scale, normalized to 1.0)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"portfolio_{tf}.png", dpi=110)
        plt.close(fig)
        print(f"Saved -> {(OUT / f'portfolio_{tf}.png').relative_to(PROJECT_ROOT)}")

        # Plot weights over time
        fig, ax = plt.subplots(figsize=(13, 4))
        weights.plot.area(ax=ax, alpha=0.7)
        ax.set_title(f"Risk-parity weights over time ({tf})")
        ax.set_ylabel("Weight")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(OUT / f"weights_{tf}.png", dpi=110)
        plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / "portfolio_summary.csv", index=False)

    print("\n=== Portfolio + benchmarks summary ===")
    pretty = summary_df.copy()
    for c in ("cagr", "max_dd", "vol_ann"):
        pretty[c] = (pretty[c] * 100).round(1).astype(str) + "%"
    pretty["sharpe"] = pretty["sharpe"].round(2)
    pretty["final_eq"] = pretty["final_eq"].round(2)
    print(pretty[["timeframe", "label", "sharpe", "cagr", "max_dd", "vol_ann", "final_eq", "n_bars"]].to_string(index=False))


if __name__ == "__main__":
    main()
