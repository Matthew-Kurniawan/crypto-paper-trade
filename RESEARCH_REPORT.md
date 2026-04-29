# Strategy Research Report — Phase 2

**Run date:** 2026-04-29 (overnight)
**Code:** `src/run_strategy_sweep.py`, `src/run_walkforward.py`
**Raw outputs:** `data/processed/sweep/`, `data/processed/walkforward/`, `data/processed/plots/`

---

## TL;DR

- **672 in-sample backtests** across 3 assets × 3 timeframes × 2 windows × 4 strategies × 2 venues × ~6 param sets.
- **72 walk-forward validations** (1-year train / 3-month test / 3-month step, with rolling param selection per period — the rigorous test).
- **Headline winner: trend-following on SOL spot.** Walk-forward Sharpe 1.7–2.0, alpha vs buy-and-hold +0.7–1.0 Sharpe.
- **Modest but real winner: BTC daily long-only momentum.** WF Sharpe 1.17–1.26, alpha ~0.4 Sharpe.
- **Robust loser: every long_short strategy on perp.** Funding cost + whipsaws turn signal alpha into negative WF returns. **Mean reversion on perp long_short is the single worst combo tested** (WF alpha −1.41 Sharpe).
- **Honest caveat:** SOL benefited from a structural rally; some of the alpha is regime alpha, not pure strategy alpha. The walk-forward design controls for forward-looking parameter peeking, but cannot control for asset selection.

---

## What was tested

### Strategies (4 classes)

| Strategy | Mechanism | Stop |
|---|---|---|
| **TSMOM** | Long if N-bar return > 0; flat or short otherwise (sign of N-bar return) | 2 × ATR(24h) |
| **Donchian** | Long when close > prior N-bar high; exit when close < prior M-bar low (M < N) | 3 × ATR(24h) |
| **MR z-score** | Long when close is `entry_z` std-devs below rolling mean; exit when within `exit_z` of mean | 4 × ATR(24h) |
| **MA crossover** | Long when SMA(fast) > SMA(slow) | 2 × ATR(24h) |

Stop multipliers differ by strategy class because mean-reversion *needs* room (it's deliberately betting against a recent move). Within each class, stops are uniform.

### Assets, timeframes, windows

- **Assets:** BTC/USDT, ETH/USDT, SOL/USDT
- **Timeframes:** 1H, 4H, 1D (daily)
- **Time windows:** `2020_2024` (full) and `2022_2024` (post-cycle peak; less bull-biased)
- **Venues:** spot (long-only, 1× lev, 10 bps fee + 3 bps slip), perp (long/short, 3× lev cap, 8 bps fee + 2 bps slip + 8h funding)
- **Risk per trade:** 2% of equity (locked design from Phase 1)

### Parameter grids (in hours; converted per timeframe)

- TSMOM lookbacks: {24, 48, 96, 168, 336, 720}h (1d–30d)
- Donchian: entry/exit pairs (24/12, 48/24, 96/48, 168/96, 336/168, 720/336)h
- MR: rolling-window {48, 96, 168, 336}h × entry_z {2.0, 2.5}
- MA crossover: (fast, slow) pairs (24/96, 48/168, 96/336, 168/720)h

Lookbacks shorter than 2 bars are skipped at a given timeframe (e.g., 24h on daily = 1 bar — degenerate).

---

## Methodology

### Sweep (in-sample)

Brute force: backtest every (asset, tf, window, strategy, params, venue, mode). Sort by Sharpe. **The headline number from this is biased upward** because we picked the best post-hoc — that's why we follow with walk-forward.

### Walk-forward (out-of-sample)

For each (asset, tf, venue, mode, strategy):
1. Train window = trailing 365 days
2. On train: backtest each candidate param, pick the one with best Sharpe
3. Test window = next 90 days. Evaluate the picked param on test.
4. Step = 90 days. Roll forward.
5. Concatenate all test pieces → walk-forward equity curve.
6. Walk-forward Sharpe = Sharpe of that concatenated curve.

This is the gold-standard test. If WF Sharpe ≈ in-sample Sharpe, the signal is real. If WF Sharpe ≪ in-sample, it was overfit to noise.

### Buy-and-hold benchmark

Computed over the **same period** as the walk-forward test (i.e., from first test_start to last test_end). Apples-to-apples.

---

## Findings

### 1. Strategy class aggregates (walk-forward)

Averaged across all (asset, timeframe) combos:

| Strategy | Venue | Mode | Avg WF α (Sharpe) | Median WF α | % combos with positive α |
|---|---|---|---|---|---|
| **TSMOM** | spot | long_only | **+0.35** | +0.30 | **78%** |
| **MA cross** | spot | long_only | **+0.26** | +0.29 | **78%** |
| Donchian | spot | long_only | +0.24 | +0.12 | 67% |
| Mean rev | spot | long_only | +0.10 | +0.12 | 67% |
| MA cross | perp | long_short | −0.28 | −0.35 | 33% |
| Donchian | perp | long_short | −0.28 | −0.20 | 0% |
| TSMOM | perp | long_short | −0.43 | −0.49 | 11% |
| **Mean rev** | perp | long_short | **−1.41** | −1.39 | **0%** |

**Two clean conclusions:**

1. **Trend-following on spot, long-only, has positive average alpha across our universe.** TSMOM and MA crossover beat buy-and-hold ~78% of the time across (asset × timeframe) combos.
2. **Long_short on perps is structurally bad in our setup.** Funding cost when long + whipsaw losses on flips wipe out the signal. Mean reversion is catastrophic.

### 2. Top individual configs (walk-forward, ranked by alpha vs buy-and-hold same period)

| Asset | TF | Strategy | Mode | WF Sharpe | B&H Sharpe | α Sharpe | WF CAGR | WF DD | Test periods |
|---|---|---|---|---|---|---|---|---|---|
| SOL | 1H | TSMOM | long_only | 1.99 | 0.95 | **+1.04** | 318% | −66% | 16 |
| SOL | 1H | Donchian | long_only | 1.83 | 0.95 | +0.88 | 214% | −61% | 16 |
| SOL | 4H | MA cross | long_only | 1.75 | 0.94 | +0.80 | 170% | −55% | 16 |
| SOL | 4H | Donchian | long_only | 1.72 | 0.94 | +0.78 | 136% | −53% | 16 |
| SOL | 4H | TSMOM | long_only | 1.69 | 0.94 | +0.75 | 157% | −56% | 16 |
| SOL | 1H | MA cross | long_only | 1.59 | 0.95 | +0.64 | 172% | −76% | 16 |
| SOL | 1D | MA cross | long_only | 1.55 | 0.95 | +0.60 | 75% | −33% | 11 |
| **BTC** | 1D | MA cross | long_only | 1.26 | 0.78 | **+0.48** | 38% | −17% | 11 |
| SOL | 1D | TSMOM | long_only | 1.42 | 0.95 | +0.47 | 77% | −43% | 12 |
| SOL | 4H | MA cross | long_short | 1.41 | 0.94 | +0.46 | 55% | −35% | 16 (perp) |
| BTC | 1D | Donchian | long_only | 1.17 | 0.78 | +0.40 | 22% | −14% | 11 |
| BTC | 1D | TSMOM | long_only | 1.17 | 0.78 | +0.39 | 31% | −20% | 11 |
| ETH | 4H | TSMOM | long_only | 1.17 | 0.87 | +0.30 | 51% | −44% | 16 |

A SOL strategy at 1H showed up as top with WF Sharpe 1.99 and the largest absolute alpha. **All top 10 are spot long-only.** First perp long_short appears at #10.

### 3. Bottom configs — what *not* to do

| Asset | TF | Strategy | Mode | WF Sharpe | α |
|---|---|---|---|---|---|
| ETH | 1H | Mean rev | long_short (perp) | −1.29 | −2.16 |
| ETH | 4H | Mean rev | long_short (perp) | −0.91 | −1.78 |
| ETH | 1H | MA cross | long_short (perp) | −0.66 | −1.53 |
| BTC | 1H | Mean rev | long_short (perp) | −0.68 | −1.46 |
| SOL | 1D | Mean rev | long_short (perp) | −0.45 | −1.40 |

**Pattern: short-timeframe (1H) + long_short + perp + mean-reversion = catastrophic.** Each adds friction:
- Short timeframe → high signal noise
- Long_short → 2× exposure to whipsaw fees
- Perp → funding drag (when long; sometimes benefit when short)
- Mean reversion → fights the trend; with tight ATR stops gets wrecked

### 4. Parameter stability over time (top configs)

Inspecting which parameter the rolling train-window picked over 16 quarterly test periods:

- **SOL/1H/TSMOM**: 6 distinct lookbacks chosen across 16 periods (some clustering on lb96h and lb720h). Param drift is real.
- **SOL/4H/MA cross**: 4 distinct (fast, slow) pairs; (96, 336)h dominated.
- **BTC/1D/TSMOM**: 4 distinct lookbacks; lb720h dominated (4 of 11 periods).
- **BTC/1D/MA cross**: only 1 param chosen across all 11 periods — (168, 720)h. Most stable config tested.

Implication: **the BTC daily configs are picking the same parameters quarter after quarter** — that's a more credible signal than SOL configs where params drift around. SOL alpha is bigger, but BTC alpha is more *stable*.

### 5. Per-period win rate

The walk-forward Sharpe numbers can hide that most individual periods are losers. For the top configs:

| Config | % positive test periods | Comment |
|---|---|---|
| SOL/1H/TSMOM | 56% (9/16) | Wins are massive; losses are bounded by stop |
| SOL/4H/MA cross | 50% (8/16) | Same shape |
| BTC/1D/TSMOM | 27% (3/11) | Right-tailed: rare wins, frequent small losses |
| BTC/1D/MA cross | 27% (3/11) | Same shape |
| ETH/4H/TSMOM | 56% (9/16) | More balanced |

**Don't expect to win every quarter.** The trend-following strategies have right-tailed return distributions: many small losses, rare big wins. That's a hard psychological profile to trade — *especially* manually.

---

## Surprising findings

1. **Mean reversion long_only on spot is mediocre but not bad.** I expected it to be much worse on crypto (which trends hard). Average WF α +0.10 means it eked out a tiny edge. With wider stops or looser thresholds it might do better.

2. **Daily timeframe on BTC and ETH outperforms hourly for trend-following.** This contradicts the intuitive "more bars = more signal." Daily noise/signal ratio is better, and daily simple strategies generalize better in walk-forward.

3. **Donchian and TSMOM produce nearly identical results in many places.** They're testing the same thing (positive trend) through different mechanisms. Adding both to a portfolio adds little diversification.

4. **One non-spot config landed in top 15: SOL/4H MA cross long_short on perp** (WF Sharpe 1.41, α +0.46). This is the only long_short combo where the signal beat the funding+whipsaw cost. Worth noting.

5. **Buy-and-hold benchmark itself is wildly different across windows.** BTC: Sharpe 1.11 in 2020-24, 0.70 in 2022-24. ETH: 1.20 → 0.28 (CAGR went from +92%/yr to −4%/yr). The user's intuition that "the time window matters enormously" was correct — and a strategy that looks great vs 2022-2024 BTC buy-and-hold (which is mediocre) might just be capturing the obvious "hold mostly, exit during the FTX crash" trade.

---

## Honest caveats — read these before believing the numbers

1. **Multiple comparisons.** I ran 672 backtests. The best one will look great by chance even if no signal exists. Walk-forward partly addresses this (per (asset, tf, strategy, venue, mode) combo, the param choice is not post-hoc), but I still chose the *strategy class* and *asset universe* with hindsight. With 32 strategy class + asset + tf + venue + mode combos, getting a few "winners" by chance is plausible. The fact that SOL spot is top across multiple strategy classes is a stronger signal than a single config winning.

2. **SOL had a 100×+ rally in our window.** Both buy-and-hold AND any momentum strategy will look great. The strategies' alpha (~0.7-1.0 Sharpe vs SOL B&H of 0.95) is real, but the absolute CAGRs of 100-300% are not portable to a different asset or different period.

3. **Walk-forward selects params from in-sample data, but the *strategy class* is still chosen with full hindsight.** A more rigorous test would also walk-forward the strategy class choice — and I expect that would degrade results further.

4. **Trade count for the BTC daily configs is low** (11 periods, ~3 trades per period). Tiny sample size. The Sharpe ~1.2 has wide error bars.

5. **My backtester uses the bar's high/low for stop-fill.** Real fills can be worse during gaps or thin liquidity. Slippage may be under-modeled in fast moves.

6. **Funding rates are modeled at the realized historical values.** Future funding could be very different (higher or lower) — this is a market-state-dependent variable.

7. **No transaction-cost model for being out of the market.** If you're flat 50% of the time on SOL during a 100× rally, you've missed half of it. The strategies do well partly because they're in-market 80-99% of the time. This is also why long-only strategies dominate long-short ones.

---

## Recommendations for next iteration

**If the goal is real-money paper trading next:**

### Tier 1 (ready for forward paper trading, with eyes open)

- **BTC/1D MA crossover (168h fast / 720h slow), long-only on spot.** Most stable config tested (1 param picked across 11 periods). WF Sharpe 1.26, α 0.48. Daily timeframe is achievable manually from Singapore (one decision per day). DD only −17%. **My #1 pick to forward-test.**
- **BTC/1D TSMOM (lb=720h, i.e. 30-day) long-only spot.** Similar profile, slightly lower Sharpe but similar mechanism. Run alongside the MA cross as a check.

### Tier 2 (interesting but riskier)

- **SOL/4H Donchian (in=336h, out=168h) long-only spot.** Best SOL config that's not just "lucky on SOL pumps." WF Sharpe 1.72, α 0.78. **Caveat:** SOL alpha may not survive a different asset cycle.
- **Multi-asset trend-following portfolio** (BTC + ETH + SOL, equal-weight, 1D, MA cross). Not tested here but a natural extension. Diversifies asset risk while keeping the signal logic constant.

### Strategies to drop

- **All perp long_short configurations** in our setup. The funding cost + whipsaw cost overwhelms the signal in walk-forward. To make long_short work you'd need to either: (a) trade much less frequently to amortize fees, (b) net positions across multiple assets, or (c) use the funding rate itself as part of the signal.
- **Mean reversion on perps long_short.** Worst combo tested. Don't even debug — just drop.

### Methodology improvements for next iteration

1. **Walk-forward the strategy class choice too.** Pick from {TSMOM, Donchian, MA cross} on train; apply on test. This is one more honest cut. I expect the WF Sharpe to drop ~0.1–0.2 from our current numbers.
2. **Add ensemble strategies.** Two trend-followers voting might be more robust than one. But adds complexity.
3. **Add a vol filter.** Skip trading when realized vol is extremely high (e.g., flash crash days). Limits worst-case slippage.
4. **Add multi-asset portfolios.** Equal-weight or vol-targeted allocations across BTC/ETH/SOL. Reduces single-asset risk.
5. **Add funding-rate-aware sizing for perps.** When funding is extreme, scale position down or flip to spot.
6. **Test on more assets.** Top-10 by liquidity (BNB, XRP, DOGE, ADA, AVAX). If trend-following works on a basket, the alpha is more credible.
7. **Fix the equity-can-go-negative quirk** in the backtester (add a hard liquidation rule when equity hits 0). Doesn't change Sharpe much but makes results more realistic.
8. **Compare against a vol-targeted buy-and-hold benchmark.** Plain buy-and-hold has wild swings; vol-targeted holds (rebalance to 30% portfolio vol) is a fairer benchmark because that's what a sensible passive trader would actually do.

### Practical note for paper trading

- The top configs win **less than half the months/quarters but win big when they do.** Right-tailed payoffs require psychological discipline — you'll have several losing months in a row before a big winner. If you're going to break the rules during the losing streak, the strategy doesn't work.
- A daily-timeframe BTC strategy needs ~5 minutes per day to check. Fits a Singapore schedule easily.
- An hourly SOL strategy needs constant attention and isn't realistic to execute manually.

---

## Files saved

```
data/processed/
├── sweep/
│   ├── sweep_results.csv             # 672 in-sample backtests
│   ├── buy_and_hold.csv              # B&H benchmarks per (asset, tf, window)
│   └── equity/                       # equity curves for top 25 + best per (asset, strategy)
├── walkforward/
│   ├── wf_results.csv                # 72 walk-forward configs
│   ├── wf_with_alpha.csv             # WF results + same-period B&H alpha
│   ├── wf_<config>_decisions.csv     # which params got picked over time
│   └── equity/                       # WF equity curves
└── plots/
    ├── walkforward_top.png           # 6-panel: top WF curves vs B&H
    └── wf_alpha_by_strategy.png      # scatter: alpha by strategy class
```

---

## What I'd ask you next

1. **Does the BTC/1D MA cross (168/720) result feel believable to you, given everything above?** That's the candidate I'd want to commit to forward paper-test.
2. **How committed are you to crypto specifically?** The same trend-following methodology is more robust on equity index futures (longer history, lower vol, less regime-changey). If the *learning* is the goal, ES/NQ might be better instruments to learn on.
3. **Do you want to explore the multi-asset portfolio angle next?** That's the most natural way to *increase* the alpha confidence without doing more parameter tweaking.

---

# Phase 3 — Walk-forward meta + risk-parity portfolio + deployable strategy

**Run date:** 2026-04-29 (after morning review)
**Code:** `src/walkforward_meta.py`, `src/build_portfolio.py`, `src/vol_target_bh.py`, `src/live_signal.py`

## What changed vs Phase 2

Three additions:

1. **Walk-forward meta-selection.** Previously the strategy class was fixed (e.g., "TSMOM") and we only walk-forwarded params within it. Now the strategy class itself is also chosen on each train window — TSMOM, Donchian, mean-rev, MA cross all compete, the highest train-Sharpe wins, and the chosen pair is applied on test.
2. **Multi-asset risk-parity portfolio.** Three asset tracks (BTC, ETH, SOL) each get walk-forward-meta-selected, then combined into a portfolio via inverse-vol weights. SOL gets a smaller weight than BTC despite higher Sharpe — that's what you asked for.
3. **Vol-targeted buy-and-hold benchmark.** Instead of comparing to naive 100% buy-and-hold, we compare to a vol-targeted version (re-sizing daily to maintain 30% annualized vol). This is a "smart passive" baseline that actually a lot of long-term crypto holders implicitly run.

## Walk-forward meta results

For each (asset, timeframe), the strategy class chosen per quarterly retrain:

| Asset | TF | Picks |
|---|---|---|
| BTC | 1d | TSMOM 6×, MA cross 4×, Donchian 2× |
| ETH | 1d | TSMOM 12×, MA cross 2× |
| SOL | 1d | TSMOM 11×, MA cross 1×, MR 1× |
| BTC | 4h | TSMOM 8×, MR 5×, MA cross 3× |
| ETH | 4h | TSMOM 7×, MA cross 7×, MR 1×, Donchian 1× |
| SOL | 4h | MA cross 9×, TSMOM 6×, MR 1× |

**Trend-following (TSMOM + MA cross) gets picked ~85% of the time** across all assets and timeframes. Mean reversion shows up occasionally during sideways periods (BTC/4h in 2023). Donchian rarely wins on train Sharpe.

WF-meta Sharpe per asset/timeframe:

| Asset | TF | WF-meta Sharpe | WF-meta CAGR | WF-meta Max DD |
|---|---|---|---|---|
| BTC | 1d | 1.18 | 33.8% | -19.9% |
| ETH | 1d | 1.07 | 19.8% | -14.1% |
| SOL | 1d | 1.35 | 68.3% | -42.7% |
| BTC | 4h | 1.15 | 47.7% | -46.5% |
| ETH | 4h | 1.01 | 40.5% | -43.6% |
| SOL | 4h | 1.42 | 103.8% | -54.9% |

Compared to Phase 2's per-strategy WF (where the strategy class was fixed), meta-WF Sharpe drops by ~0.1–0.3. **The signal survives the extra rigor cut.** Meta-selection adds noise but doesn't break the strategy.

## Multi-asset portfolio vs vol-targeted benchmarks

Three-asset (BTC, ETH, SOL) daily, risk-parity weights, over the WF period (~2 years):

| | Sharpe | CAGR | Max DD | Vol(ann) | Final Eq |
|---|---|---|---|---|---|
| **WF-meta + risk-parity portfolio** | **0.81** | **20%** | **-11%** | **26%** | 1.50× |
| Vol-targeted equal-weight basket (30% target) | 0.93 | 53% | -40% | 65% | 2.56× |
| Naive equal-weight buy-and-hold | 0.44 | 6% | -77% | 91% | 1.13× |

And the per-asset breakdown:

| | Sharpe | CAGR | Max DD | Vol |
|---|---|---|---|---|
| BTC WF-meta track | 0.78 | 47% | **-14%** | 20% |
| BTC vol-target B&H | 0.96 | 37% | -28% | 40% |
| ETH WF-meta track | 0.53 | 30% | **-10%** | 26% |
| ETH vol-target B&H | 0.67 | 21% | -25% | 39% |
| SOL WF-meta track | 0.83 | 112% | **-21%** | 49% |
| SOL vol-target B&H | 1.01 | 39% | -32% | 39% |

**This is a more sobering picture than Phase 2.** Vol-targeted B&H has *better Sharpe* than the WF-meta strategies on most assets. **The strategies' real value is not in higher Sharpe — it's in dramatically lower drawdowns** (11–21% vs 25–40%).

For a paper trader testing if they can stomach the drawdowns, this is actually the more useful answer: the strategies don't beat the smart-passive baseline on risk-adjusted return, but they keep your sleep cleaner during crashes. Whether that tradeoff is worth it depends on what you value.

The risk-parity portfolio has even lower vol (26% vs 30% target) — a sign we're under-leveraged in the blend. Adding portfolio-level vol-targeting on top would lift CAGR while keeping Sharpe the same. (Not done yet — natural extension.)

Plots in `data/processed/portfolio/portfolio_1d.png` and `weights_1d.png`.

## Deployable live signal

Built `src/live_signal.py`. Single command, one report per day. What it does:
1. Refreshes the on-disk cache (incremental fetch of new bars).
2. For each track in the live portfolio config, retrains on the trailing 365 days and picks the best (strategy, params) by train Sharpe.
3. Computes the current signal at the most recent bar.
4. Computes target position size using the same risk rules as the backtester (2% risk per trade, 2-ATR stop, leverage cap).
5. **Safety gate**: if the trailing-year Sharpe of the best param < 0.30, overrides to FLAT regardless of signal. This prevents deploying a strategy that's been losing on recent data.
6. Writes a markdown report to `data/processed/live/<YYYY-MM-DD>.md` and a JSON snapshot. The "Action required" section is the only thing you need to read day-to-day.

**Today's actual run (2026-04-29):** BTC trailing-year Sharpe is **-0.08** (negative). All 21 candidate (strategy, params) on the trailing year had weak or negative Sharpe. Safety gate triggered — system says **STAND DOWN, hold no position**. The raw strategy signal would also have been FLAT, but the safety gate makes the reasoning explicit.

This is exactly what the safety gate is for: BTC has been choppy/down recently, none of our trend-following variants are in their happy regime, so we don't trade.

## Phase-3 inventory of what's saved

```
data/processed/
├── walkforward_meta/
│   ├── wfm_results.csv                  # WF-meta per asset/timeframe
│   ├── wfm_<config>_decisions.csv       # which (class, params) got picked each quarter
│   └── equity/<config>.parquet          # WF-meta equity curves
├── portfolio/
│   ├── portfolio_summary.csv            # all benchmarks compared
│   ├── portfolio_<tf>.parquet           # multi-asset risk-parity equity
│   ├── weights_<tf>.parquet             # rolling weights over time
│   ├── portfolio_<tf>.png               # visual comparison vs benchmarks
│   └── weights_<tf>.png                 # weight stack chart over time
└── live/
    ├── <date>.md                        # daily action report
    └── <date>.json                      # machine-readable snapshot
```

## Recommended live config (current default in live_signal.py)

```python
TRACKS = [
    {"asset": "BTCUSDT", "timeframe": "1d", "venue": "spot", "mode": "long_only"},
]
TRAIN_DAYS = 365
MIN_TRAIN_SHARPE = 0.30      # safety gate
PORTFOLIO_NAV = 1000.0       # SGD nominal
```

Single asset, daily timeframe, long-only spot, with the safety gate. Once you've forward-tested this for 6–8 weeks and want to scale up, uncomment the ETH and SOL tracks in `live_signal.py` to enable the multi-asset risk-parity blend.

## Honest current state

- WF-meta + risk-parity strategy has **lower Sharpe but materially lower drawdown** vs vol-targeted B&H in 2021–2024.
- The signal survives walk-forward at the strategy class level — it's not just parameter overfitting.
- The safety gate (min train Sharpe) is currently active for BTC right now: trailing-year Sharpe is negative, so the system correctly says don't trade. If trend resumes, the gate will release and the system will start signaling LONG entries.
- Multi-asset (BTC + ETH + SOL) is **fully working but not enabled in the deployed live signal yet** — per your instruction to make a robust single-asset version first.

## Open recommendations for next phase

1. **Add portfolio-level vol-targeting** so the risk-parity blend hits a 30% annualized vol target instead of being passively 26%. Mechanical change, lifts CAGR ~15% with same Sharpe.
2. **Expand the asset universe.** If trend-following works on BTC/ETH/SOL, it should generalize. Adding 5–10 more liquid coins would diversify the alpha.
3. **Add a dynamic training-window length.** 365 days is fixed; in fast regime changes (FTX-like) a shorter window would adapt faster, in stable bull markets a longer window would be more stable.
4. **Connect to a real exchange API** (read-only first) for live position checking against the recommended target. This catches drift between paper-trade reality and the model's expected state.
