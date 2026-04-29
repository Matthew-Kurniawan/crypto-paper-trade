"""Generate dashboard.html — a single-file visual summary of the paper-trade system.

Reads:
- data/processed/live/state.json            (current recommended targets)
- data/processed/live/<latest>.json         (most recent run snapshot)
- data/processed/live/decisions_log.csv     (all historical decisions)
- data/processed/walkforward_meta/equity/   (backtested equity curves)
- data/raw/<asset>_spot_1d.parquet          (price history for charts)

Writes:
- data/processed/live/dashboard.html        (open in browser)

Run via `python -m src.build_dashboard` after a live_signal run, or it will be
called automatically by live_signal.py at the end of each run.
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE = PROJECT_ROOT / "data" / "processed" / "live"
WF_META = PROJECT_ROOT / "data" / "processed" / "walkforward_meta"
DATA = PROJECT_ROOT / "data" / "raw"

DIRECTION_WORD = {1: "LONG", 0: "FLAT", -1: "SHORT"}


def _latest_snapshot_path() -> Path | None:
    snaps = sorted(LIVE.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))
    return snaps[-1] if snaps else None


def _fmt_price(p: float | None) -> str:
    if p is None or p == 0:
        return "—"
    if abs(p) < 0.01:
        return f"${p:.6f}"
    if abs(p) < 1:
        return f"${p:.4f}"
    return f"${p:,.2f}"


def _fmt_pct(p: float | None, decimals: int = 1) -> str:
    if p is None:
        return "—"
    return f"{p*100:+.{decimals}f}%"


def _embed_png(fig, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="chart" />'


def _action_banner(results: list[dict]) -> str:
    """The big top banner: HOLD / ACT / STAND DOWN / NO DATA."""
    ok_results = [r for r in results if r.get("ok")]
    if not ok_results:
        return ('<div class="banner banner-error">No tracks have valid data. '
                'Check the per-track table below for details.</div>')

    flips = [r for r in ok_results if r.get("is_flip")]
    if flips:
        items = []
        for r in flips:
            t = r["track"]
            cur = r["current_signal"]
            prior = r.get("prior_recommended_signal")
            prior_word = "no prior" if prior is None else DIRECTION_WORD[prior]
            cur_word = DIRECTION_WORD[cur]
            if cur == 0:
                action = f"<strong>EXIT</strong> {t['asset']}"
            elif prior is None or prior == 0:
                units = r.get("target_units", 0.0)
                px = _fmt_price(r["current_close"])
                stop = _fmt_price(r.get("stop_price"))
                action = (f"<strong>OPEN {cur_word}</strong> "
                          f"{units:,.4f} {t['asset'].replace('USDT','')} "
                          f"@ {px} (stop {stop})")
            else:
                action = f"Flip {t['asset']} from {prior_word} → {cur_word}"
            items.append(f"<li>{action}</li>")
        return ('<div class="banner banner-act">'
                '<div class="banner-title">⚠️ ACTION REQUIRED</div>'
                f'<ul>{"".join(items)}</ul></div>')

    all_stand_down = all(r.get("stand_down") for r in ok_results)
    if all_stand_down:
        return ('<div class="banner banner-down">'
                '<div class="banner-title">🛑 ALL TRACKS STAND DOWN</div>'
                '<p>Trailing-year Sharpe is below threshold for every track. '
                'Hold no positions. The gate releases automatically when momentum returns.</p>'
                '</div>')

    active_holdings = [r for r in ok_results if r["current_signal"] != 0]
    if active_holdings:
        items = []
        for r in active_holdings:
            t = r["track"]
            units = r.get("target_units", 0.0)
            px = _fmt_price(r["current_close"])
            stop = _fmt_price(r.get("stop_price"))
            items.append(f"<li><strong>{t['asset']}</strong>: hold "
                         f"{units:,.4f} units @ {px}, stop {stop}</li>")
        return ('<div class="banner banner-hold">'
                '<div class="banner-title">✓ HOLD STEADY</div>'
                f'<p>No position changes today. Currently holding:</p>'
                f'<ul>{"".join(items)}</ul></div>')

    return ('<div class="banner banner-flat">'
            '<div class="banner-title">— ALL FLAT</div>'
            '<p>No active positions, no action required.</p></div>')


def _portfolio_table(results: list[dict], total_nav: float) -> str:
    rows = ['<tr><th>Asset</th><th>Status</th><th>Weight</th><th>Allocated</th>'
            '<th>Strategy</th><th>Train Sharpe</th><th>60d return</th>'
            '<th>Last close</th></tr>']
    for r in results:
        t = r["track"]
        if not r.get("ok"):
            rows.append(f'<tr><td>{t["asset"]}</td>'
                        f'<td colspan="7" class="skipped">SKIPPED: {r.get("reason","")}</td></tr>')
            continue
        sig = r["current_signal"]
        if r.get("stand_down"):
            status = '<span class="status-down">STAND DOWN</span>'
        elif sig == 1:
            status = '<span class="status-active">LONG</span>'
        elif sig == -1:
            status = '<span class="status-active">SHORT</span>'
        else:
            status = '<span class="status-flat">FLAT</span>'
        weight = r.get("weight", 0.0)
        alloc = r.get("allocated_nav", 0.0)
        ret60 = r.get("ret_60d_pct")
        rows.append(
            f'<tr><td><strong>{t["asset"]}</strong></td>'
            f'<td>{status}</td>'
            f'<td>{weight*100:.1f}%</td>'
            f'<td>S${alloc:,.0f}</td>'
            f'<td>{r["selected_strategy"]} ({r["param_label"]})</td>'
            f'<td>{r["train_sharpe"]:.2f}</td>'
            f'<td>{ret60 if ret60 else "—"}</td>'
            f'<td>{_fmt_price(r["current_close"])}</td></tr>'
        )
    return f'<table class="portfolio">{"".join(rows)}</table>'


def _attach_60d_returns(results: list[dict]) -> None:
    """Compute trailing 60d return per asset for display in the portfolio table."""
    for r in results:
        if not r.get("ok"):
            continue
        t = r["track"]
        try:
            df = pd.read_parquet(DATA / f"{t['asset'].lower()}_{t['venue']}_{t['timeframe']}.parquet")
        except FileNotFoundError:
            r["ret_60d_pct"] = None
            continue
        if len(df) < 61:
            r["ret_60d_pct"] = None
            continue
        ret = float(df["close"].iloc[-1] / df["close"].iloc[-61] - 1)
        r["ret_60d_pct"] = _fmt_pct(ret)


def _equity_chart(results: list[dict]) -> str:
    """Stack of small equity-curve panels — one per track that has WF-meta data."""
    fig, axes = plt.subplots(1, len([r for r in results if r.get("ok")]) or 1,
                             figsize=(13, 3.2), sharey=False)
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return ""
    if len(ok) == 1:
        axes = [axes]

    for ax, r in zip(axes, ok):
        t = r["track"]
        wf_path = WF_META / "equity" / f"wfm_spot_{t['asset']}_{t['timeframe']}_{t['mode']}.parquet"
        kl_path = DATA / f"{t['asset'].lower()}_{t['venue']}_{t['timeframe']}.parquet"
        title = t["asset"]
        try:
            kl = pd.read_parquet(kl_path)
            kl = kl.tail(365 * 2)  # last 2 years for context
            if not kl.empty:
                bh = kl["close"] / kl["close"].iloc[0]
                ax.plot(bh.index, bh.values, label="B&H", linewidth=1.0,
                        color="grey", linestyle="--", alpha=0.7)
        except FileNotFoundError:
            pass

        if wf_path.exists():
            wf = pd.read_parquet(wf_path)["equity"]
            ax.plot(wf.index, wf.values, label="WF strat", linewidth=1.4, color="C0")
            title += " (WF + B&H)"
        else:
            title += " (B&H only — no WF backtest)"

        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_fontsize(7)
        for label in ax.get_yticklabels():
            label.set_fontsize(7)
    fig.suptitle("Walk-forward backtest equity vs buy-and-hold (last ~2y)", fontsize=11)
    fig.tight_layout()
    return _embed_png(fig)


def _decisions_table(days: int = 60) -> str:
    log_path = LIVE / "decisions_log.csv"
    if not log_path.exists():
        return '<p class="caption">No decisions logged yet — this is the first run.</p>'
    df = pd.read_csv(log_path)
    if df.empty:
        return '<p class="caption">No decisions logged yet.</p>'
    df["as_of"] = pd.to_datetime(df["as_of"])
    cutoff = df["as_of"].max() - pd.Timedelta(days=days)
    df = df[df["as_of"] >= cutoff]
    # Show flips and stand-down changes, plus most recent run for each asset
    flips = df[df["is_flip"] == 1].copy()
    if flips.empty:
        return ('<p class="caption">No flips in the last '
                f'{days} days. The system has been steady.</p>')
    flips = flips.sort_values("as_of", ascending=False).head(40)
    rows = ['<tr><th>Date</th><th>Asset</th><th>Action</th>'
            '<th>Train Sharpe</th><th>Strategy</th></tr>']
    for _, row in flips.iterrows():
        cur = int(row["current_signal"])
        action = DIRECTION_WORD[cur]
        if row["stand_down"] == 1 and cur == 0:
            action = "STAND DOWN"
        date_str = row["as_of"].strftime("%Y-%m-%d") if isinstance(row["as_of"], pd.Timestamp) else str(row["as_of"])[:10]
        rows.append(
            f'<tr><td>{date_str}</td>'
            f'<td>{row["asset"]}</td>'
            f'<td>{action}</td>'
            f'<td>{row["train_sharpe"]:.2f}</td>'
            f'<td>{row["selected_strategy"]} ({row["param_label"]})</td></tr>'
        )
    return f'<table class="decisions">{"".join(rows)}</table>'


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Paper Trade Dashboard — {date}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 1.5em auto; padding: 0 1.5em; color: #1d1d1f; line-height: 1.5; }}
  h1 {{ font-size: 1.8em; margin-bottom: 0.1em; }}
  h2 {{ font-size: 1.2em; margin-top: 2em; padding-bottom: 0.3em; border-bottom: 1px solid #e5e5e5; }}
  .timestamp {{ color: #6e6e73; font-size: 0.9em; }}
  .banner {{ padding: 1.4em 1.6em; border-radius: 10px; margin: 1.6em 0; }}
  .banner-title {{ font-size: 1.25em; font-weight: 600; margin-bottom: 0.4em; }}
  .banner-act {{ background: #fff4e0; border-left: 6px solid #ff9500; }}
  .banner-act ul {{ margin: 0.3em 0 0 0.6em; }}
  .banner-act li {{ font-size: 1.05em; padding: 0.2em 0; }}
  .banner-down {{ background: #fbe9e9; border-left: 6px solid #d32f2f; }}
  .banner-down p {{ margin: 0; }}
  .banner-hold {{ background: #e8f5e9; border-left: 6px solid #34c759; }}
  .banner-hold p {{ margin: 0.2em 0; }}
  .banner-flat {{ background: #f5f5f7; border-left: 6px solid #8e8e93; }}
  .banner-error {{ background: #fbe9e9; border-left: 6px solid #d32f2f; padding: 1em 1.4em; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.6em 0; font-size: 0.94em; }}
  th, td {{ padding: 0.5em 0.7em; text-align: left; border-bottom: 1px solid #ececec; }}
  th {{ background: #f7f7f7; font-weight: 600; font-size: 0.88em; text-transform: uppercase; letter-spacing: 0.02em; color: #333; }}
  table.portfolio td {{ vertical-align: middle; }}
  .status-active {{ color: #2e7d32; font-weight: 700; }}
  .status-flat {{ color: #8e8e93; }}
  .status-down {{ color: #c62828; font-weight: 600; }}
  .skipped {{ color: #8e8e93; font-style: italic; }}
  .caption {{ color: #6e6e73; font-size: 0.9em; }}
  img {{ max-width: 100%; height: auto; }}
  .footer {{ color: #8e8e93; font-size: 0.85em; margin-top: 3em; border-top: 1px solid #ececec; padding-top: 1em; }}
  code {{ background: #f5f5f7; padding: 0.1em 0.4em; border-radius: 4px; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Paper Trade Dashboard</h1>
<p class="timestamp">Updated {now} • Portfolio NAV: <strong>S${nav}</strong></p>

{action_banner}

<h2>Portfolio status</h2>
{portfolio_table}

<h2>Walk-forward backtest reference</h2>
{equity_chart}
<p class="caption">Each panel shows the per-asset walk-forward backtest equity (when available) vs naive buy-and-hold over the last ~2 years. Tracks added recently (e.g. TRX, RENDER) may show buy-and-hold only — they are out-of-sample additions.</p>

<h2>Recent decisions ({history_days} days)</h2>
{decisions_table}

<div class="footer">
  Strategy: trend-following with safety gate (min train Sharpe = 0.30, freshness gate = 3 days).<br>
  Re-runs each weekday via <code>python -m src.live_signal</code>.<br>
  This is paper trading — log your simulated trades in your own journal.
  Source code: <code>src/live_signal.py</code>, <code>src/build_dashboard.py</code>.
</div>
</body>
</html>
"""


def build_dashboard(total_nav: float = 1000.0, history_days: int = 60) -> Path:
    LIVE.mkdir(parents=True, exist_ok=True)
    snap = _latest_snapshot_path()
    if snap is None:
        raise FileNotFoundError("No live snapshot found — run live_signal.py first.")
    results = json.loads(snap.read_text(encoding="utf-8"))
    _attach_60d_returns(results)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = HTML.format(
        date=snap.stem,
        now=now,
        nav=f"{total_nav:,.0f}",
        action_banner=_action_banner(results),
        portfolio_table=_portfolio_table(results, total_nav),
        equity_chart=_equity_chart(results),
        decisions_table=_decisions_table(days=history_days),
        history_days=history_days,
    )
    out = LIVE / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build_dashboard()
    print(f"Wrote {p}")
