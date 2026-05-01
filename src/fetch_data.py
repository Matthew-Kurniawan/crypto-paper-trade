"""Fetch BTC/USDT perpetual futures klines + funding rates from Binance.

USDM-margined futures endpoint (fapi.binance.com) — public, no auth.
Run as a module from the project root:

    python -m src.fetch_data

Outputs to data/raw/:
    btcusdt_perp_1h.parquet
    btcusdt_perp_4h.parquet
    btcusdt_perp_funding.parquet

Indexed by bar close_time in UTC so no-look-ahead is trivially enforced:
any signal computed at index t uses only data fully observable by time t.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PERP_BASE_URL = "https://fapi.binance.com"
PERP_KLINES_PATH = "/fapi/v1/klines"
FUNDING_PATH = "/fapi/v1/fundingRate"
# Use the public read-only mirror for spot. api.binance.com is geo-blocked
# from some regions (notably US, where GitHub Actions runners live);
# data-api.binance.vision is the same data, no restrictions, no auth.
SPOT_BASE_URL = "https://data-api.binance.vision"
SPOT_KLINES_PATH = "/api/v3/klines"
SYMBOL = "BTCUSDT"
DEFAULT_START = "2020-01-01T00:00:00Z"

INTERVAL_MS = {
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}
PERP_KLINE_LIMIT = 1500
SPOT_KLINE_LIMIT = 1000
FUNDING_LIMIT = 1000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _to_ms(ts) -> int:
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)


def _from_ms(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def _request(url: str, params: dict, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code == 418:
            wait = int(r.headers.get("Retry-After", 5))
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("max retries exceeded")


def fetch_klines(
    symbol: str, interval: str, start_ms: int, end_ms: int, venue: str = "perp"
) -> pd.DataFrame:
    if venue == "perp":
        url = PERP_BASE_URL + PERP_KLINES_PATH
        limit = PERP_KLINE_LIMIT
    elif venue == "spot":
        url = SPOT_BASE_URL + SPOT_KLINES_PATH
        limit = SPOT_KLINE_LIMIT
    else:
        raise ValueError(f"unknown venue: {venue}")
    interval_ms = INTERVAL_MS[interval]
    bars: list = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        chunk = _request(url, params)
        if not chunk:
            break
        bars.extend(chunk)
        last_open = chunk[-1][0]
        cursor = last_open + interval_ms
        if len(chunk) < limit:
            break
        time.sleep(0.1)

    if not bars:
        return _empty_klines()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(bars, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"):
        df[c] = df[c].astype("float64")
    df["trades"] = df["trades"].astype("int64")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop(columns=["ignore"])
    df = df.set_index("close_time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url = PERP_BASE_URL + FUNDING_PATH
    rows: list = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": FUNDING_LIMIT,
        }
        chunk = _request(url, params)
        if not chunk:
            break
        rows.extend(chunk)
        last_ts = int(chunk[-1]["fundingTime"])
        cursor = last_ts + 1
        if len(chunk) < FUNDING_LIMIT:
            break
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame(columns=["fundingRate"]).rename_axis("fundingTime")

    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype("float64")
    df = df[["fundingTime", "fundingRate"]].set_index("fundingTime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def validate_klines(df: pd.DataFrame, interval: str) -> None:
    if df.empty:
        raise ValueError("empty klines dataframe")

    bad_ohlc = df[
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ]
    if len(bad_ohlc):
        raise ValueError(f"OHLC inconsistency in {len(bad_ohlc)} rows")

    expected = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    deltas = df.index.to_series().diff().dropna()
    gaps = deltas[deltas > expected * 1.5]
    if len(gaps):
        print(f"  warning: {len(gaps)} timestamp gap(s) >1.5x bar size; "
              f"max = {gaps.max()}, first at {gaps.index[0]}")


def _empty_klines() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open_time", "open", "high", "low", "close", "volume",
                 "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
    ).rename_axis("close_time")


def update_klines(
    symbol: str, interval: str, start: str = DEFAULT_START, venue: str = "perp"
) -> pd.DataFrame:
    cache_path = DATA_DIR / f"{symbol.lower()}_{venue}_{interval}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        last_close = cached.index.max()
        start_ms = _to_ms(last_close) + 1
        print(f"  cache: {len(cached):,} bars through {last_close}, "
              f"updating from {_from_ms(start_ms)}")
    else:
        cached = None
        start_ms = _to_ms(start)
        print(f"  no cache, fetching from {_from_ms(start_ms)}")

    end_ms = int(time.time() * 1000)
    if start_ms >= end_ms:
        print("  up to date")
        return cached

    new = fetch_klines(symbol, interval, start_ms, end_ms, venue=venue)
    if new.empty:
        print("  no new bars")
        return cached if cached is not None else new

    print(f"  fetched {len(new):,} new bars")
    if cached is not None:
        combined = pd.concat([cached, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new

    validate_klines(combined, interval)
    combined.to_parquet(cache_path, compression="snappy")
    print(f"  saved {len(combined):,} bars to {cache_path.relative_to(PROJECT_ROOT)}")
    return combined


def update_funding(symbol: str, start: str = DEFAULT_START) -> pd.DataFrame:
    cache_path = DATA_DIR / f"{symbol.lower()}_perp_funding.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        start_ms = _to_ms(cached.index.max()) + 1
        print(f"  cache: {len(cached):,} entries through {cached.index.max()}")
    else:
        cached = None
        start_ms = _to_ms(start)
        print(f"  no cache, fetching from {_from_ms(start_ms)}")

    end_ms = int(time.time() * 1000)
    if start_ms >= end_ms:
        print("  up to date")
        return cached

    new = fetch_funding(symbol, start_ms, end_ms)
    if new.empty:
        print("  no new entries")
        return cached if cached is not None else new

    if cached is not None:
        combined = pd.concat([cached, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new

    print(f"  fetched {len(new):,} new entries, saving {len(combined):,} total")
    combined.to_parquet(cache_path, compression="snappy")
    return combined


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["1h", "4h", "1d"]


def main():
    print("=" * 60)
    print(f"Fetching multi-asset data -> {DATA_DIR}")
    print("=" * 60)

    summary = []
    for sym in SYMBOLS:
        for venue in ("spot", "perp"):
            for interval in INTERVALS:
                print(f"\n[{venue.upper()} {sym} {interval}]")
                try:
                    df = update_klines(sym, interval, venue=venue)
                    if df is not None and not df.empty:
                        summary.append((venue, sym, interval, len(df),
                                        df.index.min(), df.index.max()))
                except Exception as e:
                    print(f"  FAILED: {e}")

        # Funding only on perp (and only for symbols with active perps)
        print(f"\n[PERP {sym} funding]")
        try:
            fund = update_funding(sym)
            if fund is not None and not fund.empty:
                ann = fund["fundingRate"].mean() * 3 * 365 * 100
                print(f"  mean = {fund['fundingRate'].mean()*100:.5f}% per 8h "
                      f"(~{ann:.2f}% annualized, {len(fund):,} entries)")
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for venue, sym, interval, n, mn, mx in summary:
        print(f"{venue.upper()} {sym} {interval}: {n:>8,} bars   {mn}  ->{mx}")


if __name__ == "__main__":
    main()
