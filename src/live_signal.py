"""Live signal generator — produces today's target position for the deployable strategy.

What this does (in order):
1. Refreshes the on-disk klines cache (incremental — re-runs are cheap).
2. For each track in the live portfolio config:
   - Checks data freshness (skip if last bar is stale > MAX_STALE_DAYS old).
   - Picks the (strategy, params) with the best Sharpe on the trailing TRAIN_DAYS
     window (one walk-forward retrain at the current moment).
   - Computes the strategy's signal at the most recent bar.
   - Applies the safety gate: if best train Sharpe < MIN_TRAIN_SHARPE, override to FLAT.
3. Combines per-track signals into a portfolio target via inverse-vol weights
   computed from each asset's trailing 60-day realized volatility.
4. Compares each track's recommended target to the prior run's recorded target
   (state.json) to detect *real* action changes (not artifacts of the safety gate).
5. Writes a daily report to data/processed/live/YYYY-MM-DD.md.
6. Updates state.json with today's targets so tomorrow's run can compare.

Designed to be run on a daily schedule. The report is the deliverable.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import BacktestParams, backtest, summary_stats
from src.fetch_data import update_klines
from src.run_strategy_sweep import (
    BARS_PER_HOUR, BARS_PER_YEAR, STRATEGY_GRID, compute_signal,
    make_bt_params, param_label,
)
from src.strategy import atr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "raw"
LIVE_OUT = PROJECT_ROOT / "data" / "processed" / "live"

# Live portfolio config — what we actually paper trade.
# Updated 2026-04-29 after running src/scan_momentum.py: BTC/ETH/SOL/BNB all
# fail the trailing-Sharpe safety gate as of today; only TRX and RNDR pass.
# We include the larger caps anyway because they may release the gate later
# and we want consistent reporting; the per-track safety gate handles them.
TRACKS = [
    {"asset": "BTCUSDT",  "timeframe": "1d", "venue": "spot", "mode": "long_only"},
    {"asset": "ETHUSDT",  "timeframe": "1d", "venue": "spot", "mode": "long_only"},
    {"asset": "SOLUSDT",  "timeframe": "1d", "venue": "spot", "mode": "long_only"},
    {"asset": "TRXUSDT",  "timeframe": "1d", "venue": "spot", "mode": "long_only"},
    # RNDR was rebranded to RENDER (ticker change). Try the new ticker — if Binance
    # doesn't list it, the freshness gate will skip it and the report will say so.
    {"asset": "RENDERUSDT", "timeframe": "1d", "venue": "spot", "mode": "long_only"},
]

TRAIN_DAYS = 365              # trailing window used to pick (strategy, params)
VOL_WINDOW_BARS_DAILY = 63    # ~3 months for inverse-vol weights
PORTFOLIO_NAV = 1000.0        # SGD nominal for sizing display (paper trading)
MIN_TRAIN_SHARPE = 0.30       # Safety: if best trailing-year Sharpe < this, stand down (no trades)
MAX_STALE_DAYS = 3            # Safety: skip tracks whose last bar is older than this


def select_best_on_train(klines, atr_s, fpb, tf, venue, mode) -> tuple[str, dict, float, dict]:
    """Pick (strategy, params) with best train Sharpe."""
    best = (None, None, -np.inf, {})
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
            if s["n_trades"] < 2:
                continue
            if s["sharpe"] > best[2]:
                best = (strat, ps, s["sharpe"], s)
    return best


def run_track(track: dict) -> dict:
    asset, tf, venue, mode = track["asset"], track["timeframe"], track["venue"], track["mode"]
    print(f"\n[{asset} {tf} {venue} {mode}]")

    # 1. Refresh data
    df = update_klines(asset, tf, venue=venue)
    if df is None or df.empty:
        return {"track": track, "ok": False, "reason": "no_data"}

    # 1a. Freshness check — fail loud if data is stale (e.g., delisted/rebranded)
    last_close_time = df.index.max()
    now_utc = pd.Timestamp.now(tz="UTC")
    age = now_utc - last_close_time
    if age > pd.Timedelta(days=MAX_STALE_DAYS):
        return {"track": track, "ok": False,
                "reason": f"stale_data: last bar {last_close_time}, "
                          f"{age.days} days old (> {MAX_STALE_DAYS} day threshold)"}

    # 2. Build trailing train window ending at most recent bar
    train_end = last_close_time
    train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)
    train = df.loc[(df.index >= train_start) & (df.index <= train_end)]
    if len(train) < 100:
        return {"track": track, "ok": False, "reason": "insufficient_history"}

    atr_window = max(2, int(round(24 * BARS_PER_HOUR[tf])))
    atr_s = atr(train["high"], train["low"], train["close"], atr_window)
    fpb = pd.Series(0.0, index=train.index)  # spot (no funding)

    # 3. Pick best (strategy, params) on train
    strat, ps, train_sharpe, stats = select_best_on_train(train, atr_s, fpb, tf, venue, mode)
    if strat is None:
        return {"track": track, "ok": False, "reason": "no_param_passed_filter"}
    print(f"  trailing-{TRAIN_DAYS}d best: {strat} {param_label(strat, ps)}  "
          f"train Sharpe={train_sharpe:.2f}  trades={stats['n_trades']}  "
          f"in_market={stats['in_market_pct']*100:.0f}%")

    # 4. Compute current signal at the latest bar
    sig = compute_signal(strat, {**ps, "mode": mode}, train, tf, mode)
    current_signal = int(sig.iloc[-1])

    # 5. Determine prior signal (1 bar back) to flag if today is a flip
    prior_signal = int(sig.iloc[-2]) if len(sig) >= 2 else 0

    # 5b. Safety: if best trailing Sharpe is too weak, override to FLAT.
    #     The signal *says* what the strategy would do; the gate decides whether
    #     to trust the strategy at all this period.
    stand_down = train_sharpe < MIN_TRAIN_SHARPE
    if stand_down:
        original_signal = current_signal
        current_signal = 0
        prior_signal_for_flip = 0  # treat as flat-to-flat unless it's a true flip
    else:
        original_signal = current_signal
        prior_signal_for_flip = prior_signal

    # 6. Position-sizing suggestion using current ATR and the same risk rules as backtest
    cur_atr = float(atr_s.iloc[-1])
    cur_close = float(train["close"].iloc[-1])
    bt = make_bt_params(venue, strat)
    if current_signal != 0 and cur_atr > 0:
        risk_dollars = bt.risk_pct * PORTFOLIO_NAV
        stop_distance = bt.stop_atr_mult * cur_atr
        size_btc_by_risk = risk_dollars / stop_distance
        size_btc_by_lev = bt.leverage_cap * PORTFOLIO_NAV / cur_close
        size_units = min(size_btc_by_risk, size_btc_by_lev)
        notional = size_units * cur_close
        stop_price = cur_close - current_signal * stop_distance
    else:
        size_units = 0.0
        notional = 0.0
        stop_price = float("nan")

    return {
        "track": track,
        "ok": True,
        "as_of": last_close_time.isoformat(),
        "current_close": cur_close,
        "current_bar_high": float(train["high"].iloc[-1]),
        "current_bar_low": float(train["low"].iloc[-1]),
        "current_atr_24h": cur_atr,
        "selected_strategy": strat,
        "selected_params": ps,
        "param_label": param_label(strat, ps),
        "train_sharpe": float(train_sharpe),
        "train_n_trades": int(stats["n_trades"]),
        "train_in_market_pct": float(stats["in_market_pct"]),
        "stand_down": bool(stand_down),             # safety gate triggered?
        "raw_strategy_signal": int(original_signal),
        "current_signal": current_signal,           # -1, 0, +1 (post-safety)
        "prior_signal": prior_signal,
        "is_flip": current_signal != prior_signal,
        "target_units": float(size_units),          # base-asset units
        "target_notional_sgd": float(notional),
        "stop_price": float(stop_price) if not np.isnan(stop_price) else None,
        "stop_distance_pct": float(bt.stop_atr_mult * cur_atr / cur_close * 100) if cur_atr > 0 else 0.0,
    }


STATE_PATH_DEFAULT = None  # set in main() so tests can override


def load_prior_state(path: Path) -> dict[str, dict]:
    """Load prior recommended positions from state file. Empty dict if first run."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(path: Path, track_results: list[dict]) -> None:
    """Persist today's recommended actions for tomorrow's flip detection."""
    state = {}
    for r in track_results:
        if not r.get("ok"):
            continue
        asset = r["track"]["asset"]
        state[asset] = {
            "recommended_signal": int(r["current_signal"]),
            "as_of": r["as_of"],
            "stand_down": bool(r.get("stand_down", False)),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_decisions_log(path: Path, track_results: list[dict]) -> None:
    """Append today's decisions to a long-running CSV log for the dashboard."""
    rows = []
    run_ts = datetime.now(timezone.utc).isoformat()
    for r in track_results:
        if not r.get("ok"):
            continue
        t = r["track"]
        rows.append({
            "run_ts": run_ts,
            "as_of": r["as_of"],
            "asset": t["asset"],
            "timeframe": t["timeframe"],
            "current_close": r["current_close"],
            "selected_strategy": r["selected_strategy"],
            "param_label": r["param_label"],
            "train_sharpe": r["train_sharpe"],
            "stand_down": int(bool(r.get("stand_down", False))),
            "raw_signal": r["raw_strategy_signal"],
            "current_signal": r["current_signal"],
            "is_flip": int(bool(r.get("is_flip", False))),
            "weight": r.get("weight", 0.0),
            "target_units": r.get("target_units", 0.0),
            "target_notional_sgd": r.get("target_notional_sgd", 0.0),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def annotate_with_prior_state(track_results: list[dict], prior_state: dict[str, dict]) -> None:
    """Override `is_flip` to compare today's recommendation against the prior persisted state,
    not the strategy's raw prior-bar signal. This is the action-level flip detection."""
    for r in track_results:
        if not r.get("ok"):
            continue
        asset = r["track"]["asset"]
        prior = prior_state.get(asset)
        if prior is None:
            r["prior_recommended_signal"] = None  # first run for this asset
            r["is_flip"] = (r["current_signal"] != 0)  # initial position is a flip from no-position
            r["first_run"] = True
        else:
            prior_sig = int(prior.get("recommended_signal", 0))
            r["prior_recommended_signal"] = prior_sig
            r["is_flip"] = (r["current_signal"] != prior_sig)
            r["first_run"] = False


def compute_inverse_vol_weights(track_results: list[dict], window_bars: int = 60) -> dict[str, float]:
    """Inverse-vol weights based on each asset's trailing realized daily vol.
    SOL (high vol) gets a smaller weight than TRX (low vol), etc.
    """
    raw = {}
    for r in track_results:
        if not r.get("ok"):
            continue
        t = r["track"]
        asset, tf, venue = t["asset"], t["timeframe"], t["venue"]
        try:
            df = pd.read_parquet(DATA / f"{asset.lower()}_{venue}_{tf}.parquet")
        except FileNotFoundError:
            continue
        rets = df["close"].pct_change().dropna().tail(window_bars)
        if len(rets) < 5:
            continue
        vol = float(rets.std() * np.sqrt(BARS_PER_YEAR[tf]))
        raw[asset] = (1.0 / vol) if vol > 0 else 0.0
        r["asset_realized_vol_60d"] = vol
    total = sum(raw.values())
    if total <= 0:
        return {k: 0.0 for k in raw}
    return {k: v / total for k, v in raw.items()}


def apply_weights_to_sizing(track_results: list[dict], weights: dict[str, float], total_nav: float) -> None:
    """Mutates track_results in place: each track's allocated NAV = total * weight,
    then re-derives target units/notional/stop using the same risk rules as the backtester."""
    for r in track_results:
        if not r.get("ok"):
            continue
        asset = r["track"]["asset"]
        w = float(weights.get(asset, 0.0))
        r["weight"] = w
        r["allocated_nav"] = float(total_nav * w)
        if r["current_signal"] != 0 and r["current_atr_24h"] > 0 and w > 0:
            bt = make_bt_params(r["track"]["venue"], r["selected_strategy"])
            allocated = total_nav * w
            risk_dollars = bt.risk_pct * allocated
            stop_distance = bt.stop_atr_mult * r["current_atr_24h"]
            size_units_by_risk = risk_dollars / stop_distance
            size_units_by_lev = bt.leverage_cap * allocated / r["current_close"]
            size_units = min(size_units_by_risk, size_units_by_lev)
            r["target_units"] = float(size_units)
            r["target_notional_sgd"] = float(size_units * r["current_close"])
            r["stop_price"] = float(r["current_close"] - r["current_signal"] * stop_distance)
        else:
            r["target_units"] = 0.0
            r["target_notional_sgd"] = 0.0


def write_report(track_results: list[dict], weights: dict[str, float] | None = None) -> Path:
    LIVE_OUT.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = LIVE_OUT / f"{today}.md"

    direction_word = {1: "LONG", 0: "FLAT", -1: "SHORT"}

    lines = []
    lines.append(f"# Live Signal — {today}\n")
    lines.append(f"_Auto-generated by `src/live_signal.py`. Portfolio NAV assumed: S${PORTFOLIO_NAV:,.0f}._\n")

    flips = [r for r in track_results if r.get("ok") and r.get("is_flip")]
    if flips:
        lines.append("## Action required\n")
        for r in flips:
            t = r["track"]
            prior = r.get("prior_recommended_signal")
            prior_word = "(no prior)" if prior is None else direction_word[prior]
            tag = " (first run)" if r.get("first_run") else ""
            lines.append(f"- **{t['asset']} {t['timeframe']}**{tag}: target changes "
                         f"**{prior_word}** → **{direction_word[r['current_signal']]}**")
        lines.append("")
    else:
        lines.append("## No position changes today\n")
        lines.append("- All tracks held the same recommended target as the previous run.\n")

    # Portfolio-level weight summary
    if weights is not None and weights:
        lines.append("## Portfolio allocation (inverse-vol)\n")
        lines.append("| Asset | Weight | Allocated NAV | Realized vol (60d) |")
        lines.append("|---|---|---|---|")
        for r in track_results:
            if r.get("ok"):
                t = r["track"]
                w = r.get("weight", 0.0)
                alloc = r.get("allocated_nav", 0.0)
                vol = r.get("asset_realized_vol_60d", float("nan"))
                lines.append(f"| {t['asset']} | {w*100:.1f}% | S${alloc:,.0f} | {vol*100:.1f}% |")
        lines.append("")

    def fmt_price(p: float) -> str:
        if p == 0 or p is None:
            return "0"
        if abs(p) < 0.01:
            return f"{p:.6f}"
        if abs(p) < 1:
            return f"{p:.4f}"
        return f"{p:,.2f}"

    lines.append("## Per-track detail\n")
    for r in track_results:
        if not r.get("ok"):
            t = r["track"]
            lines.append(f"### {t['asset']} {t['timeframe']} — _skipped_: {r.get('reason')}\n")
            continue
        t = r["track"]
        lines.append(f"### {t['asset']} {t['timeframe']} ({t['venue']} {t['mode']})\n")
        lines.append(f"- As of bar close: **{r['as_of']}**, last close = **${fmt_price(r['current_close'])}**")
        lines.append(f"- Selected strategy on trailing {TRAIN_DAYS}d: **{r['selected_strategy']}** "
                     f"({r['param_label']}); train Sharpe **{r['train_sharpe']:.2f}**, "
                     f"{r['train_n_trades']} trades, in-market {r['train_in_market_pct']*100:.0f}%")
        if r.get("stand_down"):
            lines.append(f"- ⚠️  **Safety gate: STAND DOWN** "
                         f"(train Sharpe {r['train_sharpe']:.2f} < threshold {MIN_TRAIN_SHARPE}). "
                         f"Override signal to **FLAT** regardless of strategy output.")
            lines.append(f"  - Strategy raw signal would be: {direction_word[r['raw_strategy_signal']]}")
        prior_rec = r.get("prior_recommended_signal")
        prior_word = "(no prior)" if prior_rec is None else direction_word[prior_rec]
        lines.append(f"- **Current target: {direction_word[r['current_signal']]}** "
                     f"(prior recommendation: {prior_word})")
        if r["current_signal"] != 0:
            alloc = r.get("allocated_nav", PORTFOLIO_NAV)
            lines.append(f"- Target units: **{r['target_units']:.5f}** {t['asset'].replace('USDT','')}")
            lines.append(f"- Target notional: **S${r['target_notional_sgd']:,.0f}** "
                         f"(of S${alloc:,.0f} allocated, weight {r.get('weight',1.0)*100:.1f}%)")
            lines.append(f"- Stop: ${fmt_price(r['stop_price'])} "
                         f"(distance: {r['stop_distance_pct']:.2f}% from close)")
        else:
            lines.append("- Target: hold no position")
        lines.append(f"- Current ATR(24h): ${fmt_price(r['current_atr_24h'])}\n")

    lines.append("---\n")
    lines.append("## Reading this report\n")
    lines.append("- **Action required** section is the only thing that needs your attention day-to-day.")
    lines.append("- The strategy retrains the (class, params) choice **every run** on the trailing year,")
    lines.append("  so the selected strategy may change over time.")
    lines.append("- Stops are *advisory* — at this stage you're paper-trading. Log every trade you'd take")
    lines.append("  with timestamp, entry, exit, P&L, and whether you followed the rules.")
    lines.append("- This is **paper trading only**. Do not deploy real capital until forward results match")
    lines.append("  the in-sample backtest expectation per the original brief.\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Also save a JSON snapshot for downstream tools
    json_path = LIVE_OUT / f"{today}.json"
    json_path.write_text(
        json.dumps([{**r, "track": r.get("track")} for r in track_results],
                   indent=2, default=str),
        encoding="utf-8",
    )
    return report_path


def main():
    print(f"Live signal run @ {datetime.now(timezone.utc).isoformat()}")

    LIVE_OUT.mkdir(parents=True, exist_ok=True)
    state_path = LIVE_OUT / "state.json"
    prior_state = load_prior_state(state_path)
    if prior_state:
        print(f"  Loaded prior state for {len(prior_state)} assets")
    else:
        print("  No prior state — first run (every active signal is a flip)")

    results = [run_track(t) for t in TRACKS]

    # Detect real action changes against prior recorded recommendations
    annotate_with_prior_state(results, prior_state)

    # Inverse-vol weights and per-track sizing
    weights = compute_inverse_vol_weights(results, window_bars=VOL_WINDOW_BARS_DAILY)
    apply_weights_to_sizing(results, weights, PORTFOLIO_NAV)

    if weights:
        print("\nInverse-vol weights:")
        for asset, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            print(f"  {asset:<10}  {w*100:>5.1f}%")

    report_path = write_report(results, weights)
    save_state(state_path, results)
    append_decisions_log(LIVE_OUT / "decisions_log.csv", results)

    # Reconcile auto-journal — handles new opens, closes, stop-outs, reversals.
    # On first run (journal empty) we replay decisions_log to bootstrap.
    from src.journal import reconcile as reconcile_journal, bootstrap_from_decisions_log
    journal_path = LIVE_OUT / "paper_trade_journal.csv"
    bootstrapped = bootstrap_from_decisions_log(LIVE_OUT / "decisions_log.csv", journal_path)
    if bootstrapped > 0:
        print(f"  Journal bootstrap: replayed {bootstrapped} entries from decisions_log")
    journal_summary = reconcile_journal(results, journal_path)
    print(f"\nJournal: opened={journal_summary['opened']}, "
          f"closed={journal_summary['closed']}, stopped={journal_summary['stopped']}, "
          f"rows={journal_summary['total_rows']}")

    print(f"\nReport saved -> {report_path.relative_to(PROJECT_ROOT)}")
    print(f"State updated -> {state_path.relative_to(PROJECT_ROOT)}")
    print(f"Decisions logged -> {(LIVE_OUT / 'decisions_log.csv').relative_to(PROJECT_ROOT)}")
    print(f"Journal -> {journal_path.relative_to(PROJECT_ROOT)}")

    # Generate the visual dashboard for easy daily review
    try:
        from src.build_dashboard import build_dashboard
        dash_path = build_dashboard()
        print(f"Dashboard -> {dash_path.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"  (dashboard generation skipped: {e})")

    # Print actionable summary to stdout for log scrubbing
    flips = [r for r in results if r.get("ok") and r.get("is_flip")]
    if flips:
        print("\n=== ACTION REQUIRED ===")
        for r in flips:
            t = r["track"]
            prior = r.get("prior_recommended_signal")
            prior_str = "none" if prior is None else f"{prior:+d}"
            tag = " (first run)" if r.get("first_run") else ""
            print(f"  {t['asset']} {t['timeframe']}{tag}: {prior_str} -> {r['current_signal']:+d}")
    else:
        print("\nNo position changes — hold steady.")


if __name__ == "__main__":
    main()
