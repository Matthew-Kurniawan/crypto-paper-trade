# BTC/USDT 4H Time-Series Momentum — Paper Trading Project

Phase 1 of a learn-by-doing quant project: a single, well-specified TSMOM strategy on BTC/USDT
perpetual futures, evaluated rigorously before any real money touches it. Full context lives
in [`btc_momentum_paper_trade_brief.md`](btc_momentum_paper_trade_brief.md).

## Locked design (phase 1)

| Item | Value |
|---|---|
| Strategy | Time-series momentum (long/short by signal sign) |
| Asset | BTC/USDT perpetual futures (Binance) |
| Bars | **1H and 4H both evaluated in-sample**, headline timeframe picked from data |
| Lookback search | {24h, 48h, 96h, 168h} → pick on in-sample |
| Position sizing | Fixed-fractional, **2% portfolio risk per trade** |
| Stop | 2 × 24h ATR |
| Leverage cap | **3x** |
| Frictions | 0.08% taker × 2 = 0.16% round-trip + 2 bps slippage + 8h funding |
| In-sample window | 2020-01-01 → 2024-12-31 |
| Holdout window | 2025-01-01 → present (do not peek during in-sample) |

## Setup (Windows, Python 3.12+)

```bash
# from project root
python -m venv .venv
.venv/Scripts/activate          # bash on Windows (Git Bash / MSYS)
# OR  .venv\Scripts\activate.bat   (cmd)
# OR  .venv\Scripts\Activate.ps1   (PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
```

Verify:

```bash
python -c "import pandas, numpy, requests, pyarrow; print('ok')"
```

## Project layout

```
.
├── btc_momentum_paper_trade_brief.md   # Source-of-truth project brief
├── README.md                           # This file
├── requirements.txt
├── src/                                # Reusable modules (data fetch, strategy, backtest)
├── data/
│   ├── raw/                            # Cached Binance klines (parquet, gitignored)
│   └── processed/                      # Resampled / derived series
├── notebooks/                          # Exploration, plots, analysis
└── tests/                              # Unit tests for indicator math, no-look-ahead checks
```

## Methodology guardrails

1. **Lock rules in writing before backtesting.** No tweaking parameters after seeing results.
2. **Holdout is sacred.** No peeking at 2025+ data during in-sample dev.
3. **Model frictions honestly.** Fees, slippage, funding, bar-close execution.
4. **Track rule adherence in forward trading**, not just P&L.
5. **If the strategy fails out-of-sample, do not retrofit.** Pick a different strategy class.
