# BTC/USDT Hourly Momentum Paper Trade — Project Brief

**Purpose of this document:** Hand off context to a fresh Claude session so the user
doesn't have to re-explain everything. Read this in full before responding.

---

## TL;DR for the next Claude

The user wants to paper-trade a **single, well-defined momentum strategy on BTC/USDT
on the 1-hour timeframe** for ~6–8 weeks, with you as a thinking/coding partner. 
Your job is to help them define rules precisely, fetch and analyze historical data,
do walk-forward validation, and then track manual paper trades. In the future, help them
deploy autonomous execution, connect Claude to a live exchange API for trading
decisions.

---

## User profile (important context)

- Based in **Singapore**. US market hours are brutal for them, which is part of why
  crypto (24/7) is appealing. SGT.
- Has an **ML engineering background** from graduate work — comfortable with pandas,
  numpy, basic statsmodels, and standard ML evaluation concepts (train/test split,
  overfitting, etc.). You can speak technically.
- Currently between graduate work and starting a full-time job (~1 month away).
- **Has a solid existing portfolio** in diversified ETFs already. Understands long-term
  compounding. This is not their primary investment vehicle.
- **S$1,000 satellite allocation** for this experiment. Can genuinely afford to lose
  it. Not earmarked for any near-term goal.
---

## What we ARE doing (the actual project)

### Strategy

- **Asset:** BTC/USDT (deepest liquidity, cleanest data, 24/7)
- **Timeframe:** 1-hour bars
- **Class:** Momentum (user's choice from a constrained menu I offered)
- **Mode:** Paper trading only, manually executed by the user when conditions
  trigger. No live money for at least 6–8 weeks of forward results.

### Methodology (the part that matters)

We're doing this rigorously to actually learn something, not to confirm a bias.

1. **Strategy specification (Week 1)**
   - Define entry rule, exit rule, position sizing, stop loss, time stop. In writing,
     before any data is touched. The user has to commit to the rules before seeing
     out-of-sample results.
   - One strategy class, one parameter set initially. No "let's also try…" until
     we've finished evaluating the first.

2. **In-sample development**
   - Fetch BTC/USDT 1H OHLCV data from **Binance public API** (no auth required for
     historical klines — `https://api.binance.com/api/v3/klines`).
   - Use 2020-01-01 to 2024-12-31 as the development window.
   - Evaluate: total return, Sharpe (annualized, accounting for 24/7 trading →
     sqrt(24*365) for hourly), max drawdown, win rate, average trade, trade count.
   - Be honest about how many parameter variants are tried. Each one inflates
     false-positive risk.

3. **Walk-forward / out-of-sample**
   - Held-out window: 2025-01-01 onward. The user does not look at this data
     during step 2. Even ambient awareness of "what happened in markets recently"
     is a form of look-ahead — flag this honestly.
   - Walk-forward: rolling train window → test on next month → roll forward.
     Report metrics on aggregate test periods, not just the full holdout.
   - If the strategy fails here: **go back to step 1 and pick a different strategy
     class**, do not just tweak parameters on the same idea (that's overfitting).

4. **Forward paper trading (Onwards)**
   - User manually places paper trades when their rules trigger. TradingView free
     tier is fine for charting and alerts.
   - Log each trade: timestamp, entry price, exit price, P&L, reason exited
     (stop, target, time stop, rule violation).
   - **Crucially:** track rule adherence, not just P&L. If they overrode rules,
     log it. The strategy might work; the trader might not. Both are valuable
     lessons.
   - Weekly review with Claude: spot mistakes, review decisions, do NOT change
     rules mid-experiment based on recent losses (that's the same gut-feel
     trading they wanted to escape).

5. **Real-money decision (afterwards)**
   - If forward results meaningfully match backtest expectations: consider real
     money in *small* size (S$200 of the S$1,000, not the full amount). Real
     money creates psychological dynamics paper doesn't.
   - If forward results disagree with backtest: that's the most valuable lesson
     of all. The strategy didn't work, but the methodology did.

### Specific momentum rule starting points (to discuss with user)

Don't just pick one for them — discuss the trade-offs. Reasonable starting candidates:

- **Donchian breakout:** Long when close > N-period high (e.g. 24h or 48h),
  exit when close < shorter-period low (e.g. 12h). Classic, easy to backtest,
  well-documented in published research (Turtle traders, etc).
- **MA crossover momentum:** Long when fast MA > slow MA *and* price > slow MA,
  exit on cross-down. Pick reasonable lengths (e.g. 24/72 on hourly).
- **Time-series momentum:** Long when N-period return > 0 (e.g. past 7 days
  return positive), flat otherwise. Has the most academic support
  (Moskowitz/Ooi/Pedersen 2012).

help research the most proven method and suggest to the user

### Position sizing and risk

- Fixed fractional: risk a fixed % (e.g. 1–2%) of paper portfolio per trade,
  defined by stop distance. No martingale, no scaling in, no averaging down.
- Stop loss: ATR-based (e.g. 2× 24-hour ATR below entry) or fixed % (e.g. 3%).
  Pick one and stick with it.
- Time stop: optional but worth considering — exit if not at target after N
  bars, since momentum that hasn't continued is statistically less likely to.

---

## Tooling (keep it minimal)

- **Python + pandas + numpy** for backtest. The user has the chops.
- **`requests`** to hit Binance public API. No API key needed for historical
  klines. Endpoint: `https://api.binance.com/api/v3/klines` with params
  `symbol=BTCUSDT`, `interval=1h`, `startTime`, `endTime`, `limit=1000` (paginate).
- **`vectorbt` or `backtrader`** are optional. For a single strategy on hourly
  data, plain pandas is honestly enough and more transparent. Recommend pandas
  unless the user wants the practice with a backtesting library.
- **TradingView free tier** for charting and alerts during forward trading.

---

## Trading frictions to model honestly in the backtest

These are where most retail backtests lie:

- **Fees:** Binance taker fee ~0.1% per side → 0.2% round-trip. Model this.
- **Slippage:** Add a realistic estimate (1–5 bps for BTC/USDT, more in volatile
  conditions). For hourly bars on a liquid pair this is small but not zero.
- **Funding rates:** Only relevant if trading perpetual futures, not spot.
  Confirm with user which they're modeling. **Default assumption: spot.**
- **Bar-close execution:** Assume entries and exits happen at the bar close
  the rule fires on, not the open of the next bar. Be consistent. Document
  the choice.
- **No look-ahead:** All indicator values must use only data available at the
  bar's close. Easy to mess this up — e.g. a 24-period rolling max that
  includes the current bar leaks future info.

---

## What success looks like for this project

Not a dollar number. The user is here to learn. Success is:

1. A clean, reproducible backtest with honest evaluation.
2. Forward paper trading results that the user can compare to the backtest.
3. The user develops calibrated intuition about what works, what doesn't, and
   how easy it is to fool yourself.
4. The user does *not* deploy real money on something that hasn't survived
   forward testing.

If after 6–8 weeks the strategy performs and the user wants to deploy small
real-money size, fine. If it fails, that's a successful project too.

---

## Tone and stance for the next Claude

- Be technically rigorous. The user can handle it.
- Be honest about limitations of LLMs in trading. Don't oversell.
- Push back if the user tries to expand scope back into autonomous AI
  trading territory. Refer them to this document.
- Don't be preachy or repeat the same warnings every message. Once we're
  past the initial scoping (which we are), focus on doing the work well.
- Celebrate small wins (clean code, honest results) and bad results (a
  strategy that fails out-of-sample is the system working correctly).

---

## Suggested first message from the next Claude

Something like:

> "I've read the brief. Before we start coding, let me confirm three things
> with you so we lock the scope:
>
> 1. **Strategy choice:** help pick the best after your research
>
> 2. **Spot or perpetual futures?** Spot is simpler (no funding, no
>    liquidation risk). Perps let you go short cleanly but add complexity.
>    For a first project, I'd suggest spot, long-only.
>
> 3. **Python environment:** Do you already have a Python setup with
>    pandas/numpy ready, or do you want me to walk through setting one up?
>    (If you used pandas in your grad projects you're fine.)
>
> Once we have those three, I'll write the data-fetch script for Binance
> historical klines and we can have BTC/USDT 1H bars from 2020 onward in
> a DataFrame within the next message."

---

*End of brief.*
