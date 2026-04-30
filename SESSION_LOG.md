# Session log

Append new sessions at the **top**. Each entry < ~150 words.
Older than ~6 weeks: prune unless still load-bearing.

---

## 2026-04-30 — Auto-journaling, GitHub Actions cloud schedule, token-economy setup

**Made the system fully autonomous and session-resumable.**

- Built `src/journal.py` with `reconcile()` and `bootstrap_from_decisions_log()`. Wired into `live_signal.py` end-of-run.
- Journal schema: `paper_trade_journal.csv` with auto-filled trade lifecycle + two human-fillable columns (`followed_rules`, `notes`).
- First closed paper trade: **TRX +1.24% / +S$4.99** (LONG @ $0.3217 → close @ $0.3257).
- Un-ignored `data/processed/live/*` so dashboard, journal, decisions_log, state are tracked in git for session continuity.
- Created `CLAUDE.md` (read-first orientation) and `SESSION_LOG.md` (this file) to cap token use across sessions.
- Added `.github/workflows/daily_live_signal.yml`: cron 01:00 UTC, runs `live_signal.py`, commits state, opens an Issue on flips (so user gets email notification).
- Files changed: `src/journal.py` (new), `src/live_signal.py`, `.gitignore`, `CLAUDE.md` (new), `SESSION_LOG.md` (this), `.github/workflows/daily_live_signal.yml` (new).
- Outstanding: user to verify GitHub Action runs cleanly tomorrow; manually trigger via Actions tab if curious.

---

## 2026-04-29 — Multi-asset deployable + dashboard + scan + GitHub init

**Made the strategy deployable, visualizable, and version-controlled.**

- Built `src/scan_momentum.py` (36-symbol universe scan). Only 2 symbols passed safety + LONG: **TRX** and **RNDR** (RNDR data was stale — found via the new freshness gate; replaced with RENDERUSDT).
- Made `live_signal.py` multi-track. Added inverse-vol weights: TRX 40%, BTC 19%, SOL 15%, ETH 14%, RENDER 12%. SOL gets a smaller slice automatically (high vol).
- Added safety gate (min train Sharpe = 0.30) and freshness gate (skip if last bar > 3 days old).
- Added state tracking (`state.json`) so flips reflect real action changes, not raw signal flip-flop.
- Built `src/build_dashboard.py`: self-contained HTML with action banner, portfolio table, embedded backtest charts, decisions log.
- `git init` + initial commit + push to https://github.com/Matthew-Kurniawan/crypto-paper-trade (private).
- Today's recommended action: LONG TRX (1247 units @ $0.3217, stop $0.3169).
- Outstanding: user to set up Task Scheduler and/or GitHub Actions for auto-runs.

---

## 2026-04-29 — Phase-3 walk-forward meta + risk-parity portfolio

**Added one more rigor cut and a multi-asset preview.**

- Built `src/walkforward_meta.py`: WF where the strategy class itself is selected per train window. Trend-following gets picked ~85% of the time across all (asset, tf) combos.
- WF-meta Sharpe: BTC/1d 1.18, ETH/1d 1.07, SOL/1d 1.35. ~0.1–0.3 below per-strategy WF (rigor cut works as expected — adds noise but doesn't break signal).
- Built `src/build_portfolio.py`: 3-asset risk-parity portfolio + vol-targeted B&H benchmark.
- **Honest finding**: strategies have *lower* Sharpe than vol-targeted B&H (0.81 vs 0.93) but *much* lower drawdown (-11% vs -40%). The value is drawdown control, not raw Sharpe.

---

## 2026-04-28 — Overnight phase-2 multi-strategy research

**Tested 4 strategy classes × 3 assets × 3 timeframes × 2 windows.**

- 672 in-sample backtests + 72 walk-forward validations. Spot, long-only.
- Trend-following on spot long-only beats B&H ~78% of (asset, tf) combos in WF.
- Long_short on perp got destroyed by whipsaws at every lookback tested.
- Recommended single-asset deployable: BTC/1d MA cross (168/720). Param picked all 11 quarters → most stable config.
- Full results in `RESEARCH_REPORT.md`.

---

## 2026-04-28 — Project scoping + phase-1 in-sample TSMOM

**Established methodology, ran first rigorous backtest, found honest negative result.**

- Decided: BTC/USDT, perp futures, long/short TSMOM, 4H bars, 2% risk, 3x leverage cap, in-sample 2020-2024 / holdout 2025+.
- Phase-1 result: TSMOM with our committed param grid did not beat buy-and-hold on perp or spot. Funding cost on long-held perp positions ~12% annualized.
- Per brief discipline ("if strategy fails, pick a different class"), moved to phase-2 multi-strategy research.
