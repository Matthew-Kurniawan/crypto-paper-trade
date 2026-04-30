# Read-first orientation for future Claude sessions

**This file is your token-economy entry point.** When a fresh session starts on
this repo, read in this order and stop when you have enough to answer:

1. **`SESSION_LOG.md`** — most recent 3–5 sessions. ~80% of the time this is
   all the context you need.
2. **`data/processed/live/dashboard.html`** (or the corresponding `.md` for
   the latest date) — current trading state in one screen.
3. **`data/processed/live/paper_trade_journal.csv`** — every paper trade
   so far, with realized P&L.
4. **`data/processed/live/state.json`** — current target position per asset.
5. **`README.md`** — setup instructions if anything looks broken.
6. **`RESEARCH_REPORT.md`** — only if the user asks about methodology,
   strategy choice, or backtest results. This is the long doc; avoid loading
   it for routine daily questions.
7. **`btc_momentum_paper_trade_brief.md`** — original project brief; only
   needed for first-principles disagreements.

## What this project is, in one paragraph

Single-asset → small multi-asset crypto paper-trading research. We
walk-forward-tested 4 strategy classes (TSMOM, Donchian, mean-rev z-score, MA
crossover) across BTC/ETH/SOL × multiple timeframes × multiple windows.
Trend-following on spot long-only beats buy-and-hold ~78% of (asset, tf)
combos in walk-forward. Phase-3 deployable system runs daily on 1d bars,
auto-retrains on trailing 365 days, picks the best (strategy, params) per
asset, applies a safety gate (min train Sharpe = 0.30) and a freshness gate
(skip stale tickers), risk-parity weights across tracks, writes a daily
report + journal + dashboard. User is paper-trading from S$1,000 NAV in
Singapore (SGT). Goal is *learning*, not return-chasing.

## Day-to-day commands

```bash
python -m src.live_signal          # daily: refresh data, retrain, recommend
python -m src.scan_momentum        # weekly: re-scan altcoin universe for new candidates
python -m src.build_dashboard      # if you want to regen dashboard without re-running signal
```

The GitHub Action `.github/workflows/daily_live_signal.yml` runs daily at
01:00 UTC. If you see a recent commit by `github-actions[bot]`, that's it.

## How to end a session (token economy)

When the user signals "done for today" / "save and exit" / similar, append
a digest to `SESSION_LOG.md` using this template:

```markdown
## YYYY-MM-DD (most recent at top)

**Topic in one line.**

- Key decision 1
- Key decision 2
- Files changed: `path/to/file.py`, ...
- Outstanding: anything the next session needs to know

```

Keep each digest under ~150 words. Drop digests older than ~6 weeks if the
log gets long. The point is rolling context, not an audit trail.

## Things to NOT do without checking with the user first

- Lower the safety-gate threshold (currently 0.30) to make the system trade
  more. The user explicitly chose strict over loose.
- Add new strategy classes or new lookback values. Param-tweaking after
  seeing results is what the brief warns against.
- Switch from spot to perp. We tested perp; funding cost was a structural
  drag on long-held positions.
- Increase leverage above 3x. The user picked the "aggressive but
  survivable" tier deliberately.
- Invoke real-money trading APIs. This is paper-only until the user says so.

## Things you can do without checking

- Add new tracks to the live portfolio (after running `scan_momentum` to
  justify them).
- Update the dashboard layout / formatting.
- Fix bugs in any module.
- Improve the report / journal schema (additive only — don't delete columns).
- Run `git commit` with descriptive messages. Don't push without asking.

## User profile reminder

ML engineering background. Speak technically. They handle the math; what
they want from you is rigor, honest pushback, and a clean trail of
decisions. Don't pad responses.
