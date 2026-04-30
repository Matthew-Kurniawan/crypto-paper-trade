"""Append-only paper-trade journal, reconciled against the live state each run.

Rules:
- One row per trade. The row starts with `status='open'` and is filled in over time.
- When the live state shows an active position for an asset that has no open
  journal row, we create a new row (handles bootstrap and any missed events).
- When the live state shows FLAT for an asset that has an open journal row,
  we close that row.
- Before any of the above, we check whether the current bar's high/low has
  touched the stop on any open position; if so, we close at the stop price
  with exit_reason='stop'.

The auto-filled columns are everything the system can know. Two columns are
left for the human: `followed_rules` (Y/N — did you actually execute it?)
and `notes` (free text). Everything else is computed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

JOURNAL_COLUMNS = [
    "trade_id",
    "asset",
    "side",
    "opened_run_ts",
    "entry_bar_close_ts",
    "entry_price_recommended",
    "size_units",
    "stop_price",
    "allocated_nav_sgd",
    "selected_strategy",
    "param_label",
    "status",                   # open / closed / stopped
    "closed_run_ts",
    "exit_bar_close_ts",
    "exit_price_recommended",
    "exit_reason",              # signal_flip / stop / reversal / stand_down / manual
    "gross_pnl_sgd",
    "gross_pnl_pct",
    "followed_rules",           # human-fillable: Y / N / blank
    "notes",                    # human-fillable
]

DIRECTION_NAME = {1: "LONG", -1: "SHORT"}


def load_journal(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        # Make sure all columns exist (forward-compat with schema additions)
        for col in JOURNAL_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[JOURNAL_COLUMNS]
    return pd.DataFrame(columns=JOURNAL_COLUMNS)


def save_journal(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df[JOURNAL_COLUMNS]
    df.to_csv(path, index=False)


def _next_trade_id(journal: pd.DataFrame) -> int:
    if journal.empty or journal["trade_id"].isna().all():
        return 1
    return int(journal["trade_id"].max()) + 1


def _close_row(journal: pd.DataFrame, idx: int, *, run_ts: str, exit_bar_ts: str,
               exit_price: float, exit_reason: str, status: str = "closed") -> None:
    """Mutate journal in place to close one trade row."""
    journal.loc[idx, "status"] = status
    journal.loc[idx, "closed_run_ts"] = run_ts
    journal.loc[idx, "exit_bar_close_ts"] = exit_bar_ts
    journal.loc[idx, "exit_price_recommended"] = float(exit_price)
    journal.loc[idx, "exit_reason"] = exit_reason
    side = 1 if journal.loc[idx, "side"] == "LONG" else -1
    entry_px = float(journal.loc[idx, "entry_price_recommended"])
    size = float(journal.loc[idx, "size_units"] or 0)
    pnl = side * size * (float(exit_price) - entry_px)
    nav = float(journal.loc[idx, "allocated_nav_sgd"] or 0)
    journal.loc[idx, "gross_pnl_sgd"] = round(pnl, 4)
    journal.loc[idx, "gross_pnl_pct"] = round(pnl / nav, 6) if nav > 0 else None


def bootstrap_from_decisions_log(decisions_log_path: Path, journal_path: Path) -> int:
    """One-shot replay of decisions_log.csv into the journal.

    Only runs when the journal file does not exist or is empty. After the
    journal is populated, normal `reconcile()` calls take over.

    Returns number of rows written.
    """
    if journal_path.exists():
        existing = pd.read_csv(journal_path)
        if not existing.empty:
            return 0
    if not decisions_log_path.exists():
        return 0

    log = pd.read_csv(decisions_log_path)
    if log.empty:
        return 0
    log = log.sort_values("run_ts").reset_index(drop=True)

    journal = pd.DataFrame(columns=JOURNAL_COLUMNS)
    open_trades: dict[str, int] = {}
    next_id = 1

    for _, row in log.iterrows():
        asset = row["asset"]
        cur_sig = int(row["current_signal"])
        run_ts = row["run_ts"]
        as_of = row["as_of"]
        cur_px = float(row["current_close"])
        prior_open = open_trades.get(asset)

        if cur_sig != 0 and prior_open is None:
            new = {col: pd.NA for col in JOURNAL_COLUMNS}
            new.update({
                "trade_id": next_id,
                "asset": asset,
                "side": DIRECTION_NAME[cur_sig],
                "opened_run_ts": run_ts,
                "entry_bar_close_ts": as_of,
                "entry_price_recommended": cur_px,
                "size_units": float(row.get("target_units", 0)),
                "allocated_nav_sgd": float(row.get("target_notional_sgd", 0)),
                "selected_strategy": row.get("selected_strategy"),
                "param_label": row.get("param_label"),
                "status": "open",
            })
            journal = pd.concat([journal, pd.DataFrame([new])], ignore_index=True)
            open_trades[asset] = len(journal) - 1
            next_id += 1
        elif cur_sig == 0 and prior_open is not None:
            _close_row(journal, prior_open,
                       run_ts=run_ts, exit_bar_ts=as_of,
                       exit_price=cur_px, exit_reason="signal_flip")
            del open_trades[asset]
        elif cur_sig != 0 and prior_open is not None:
            existing_side = journal.loc[prior_open, "side"]
            target_side = DIRECTION_NAME[cur_sig]
            if existing_side != target_side:
                _close_row(journal, prior_open,
                           run_ts=run_ts, exit_bar_ts=as_of,
                           exit_price=cur_px, exit_reason="reversal")
                del open_trades[asset]
                new = {col: pd.NA for col in JOURNAL_COLUMNS}
                new.update({
                    "trade_id": next_id,
                    "asset": asset,
                    "side": target_side,
                    "opened_run_ts": run_ts,
                    "entry_bar_close_ts": as_of,
                    "entry_price_recommended": cur_px,
                    "size_units": float(row.get("target_units", 0)),
                    "allocated_nav_sgd": float(row.get("target_notional_sgd", 0)),
                    "selected_strategy": row.get("selected_strategy"),
                    "param_label": row.get("param_label"),
                    "status": "open",
                })
                journal = pd.concat([journal, pd.DataFrame([new])], ignore_index=True)
                open_trades[asset] = len(journal) - 1
                next_id += 1

    save_journal(journal, journal_path)
    return len(journal)


def reconcile(track_results: list[dict], journal_path: Path) -> dict:
    """Reconcile the journal against today's track_results.

    Returns a summary dict: counts of opens / closes / stops written.
    """
    run_ts = datetime.now(timezone.utc).isoformat()
    journal = load_journal(journal_path)

    open_idx_by_asset: dict[str, int] = {}
    if not journal.empty:
        opens = journal[journal["status"] == "open"]
        for idx, row in opens.iterrows():
            open_idx_by_asset[row["asset"]] = idx

    n_opened = 0
    n_closed = 0
    n_stopped = 0
    new_rows = []

    for r in track_results:
        if not r.get("ok"):
            continue
        t = r["track"]
        asset = t["asset"]
        cur_signal = int(r["current_signal"])
        cur_close = float(r["current_close"])
        as_of = str(r["as_of"])
        existing_idx = open_idx_by_asset.get(asset)

        # 1. Stop check on any existing open position for this asset.
        # We use the most recent bar's high/low as a proxy. (For paper trading on
        # daily bars, this catches a stop hit on the previous day's bar.)
        if existing_idx is not None:
            stop_price_raw = journal.loc[existing_idx, "stop_price"]
            try:
                stop_price = float(stop_price_raw) if pd.notna(stop_price_raw) else None
            except (TypeError, ValueError):
                stop_price = None
            if stop_price is not None and r.get("current_bar_high") is not None:
                bar_high = r["current_bar_high"]
                bar_low = r["current_bar_low"]
                side = journal.loc[existing_idx, "side"]
                if side == "LONG" and bar_low <= stop_price:
                    _close_row(journal, existing_idx,
                               run_ts=run_ts, exit_bar_ts=as_of,
                               exit_price=stop_price, exit_reason="stop",
                               status="stopped")
                    n_stopped += 1
                    open_idx_by_asset.pop(asset, None)
                    existing_idx = None
                elif side == "SHORT" and bar_high >= stop_price:
                    _close_row(journal, existing_idx,
                               run_ts=run_ts, exit_bar_ts=as_of,
                               exit_price=stop_price, exit_reason="stop",
                               status="stopped")
                    n_stopped += 1
                    open_idx_by_asset.pop(asset, None)
                    existing_idx = None

        # 2. Reconcile state ↔ journal.
        if cur_signal == 0 and existing_idx is not None:
            # Position should be closed.
            reason = "stand_down" if r.get("stand_down") else "signal_flip"
            _close_row(journal, existing_idx,
                       run_ts=run_ts, exit_bar_ts=as_of,
                       exit_price=cur_close, exit_reason=reason)
            n_closed += 1
            open_idx_by_asset.pop(asset, None)
        elif cur_signal != 0 and existing_idx is None:
            # New position to open.
            new_rows.append({
                "trade_id": _next_trade_id(journal) + len(new_rows),
                "asset": asset,
                "side": DIRECTION_NAME[cur_signal],
                "opened_run_ts": run_ts,
                "entry_bar_close_ts": as_of,
                "entry_price_recommended": cur_close,
                "size_units": float(r.get("target_units", 0) or 0),
                "stop_price": float(r["stop_price"]) if r.get("stop_price") else None,
                "allocated_nav_sgd": float(r.get("allocated_nav", 0) or 0),
                "selected_strategy": r["selected_strategy"],
                "param_label": r["param_label"],
                "status": "open",
                "closed_run_ts": pd.NA, "exit_bar_close_ts": pd.NA,
                "exit_price_recommended": pd.NA, "exit_reason": pd.NA,
                "gross_pnl_sgd": pd.NA, "gross_pnl_pct": pd.NA,
                "followed_rules": pd.NA, "notes": pd.NA,
            })
            n_opened += 1
        elif cur_signal != 0 and existing_idx is not None:
            existing_side = journal.loc[existing_idx, "side"]
            target_side = DIRECTION_NAME[cur_signal]
            if existing_side != target_side:
                # Reverse: close existing, open new.
                _close_row(journal, existing_idx,
                           run_ts=run_ts, exit_bar_ts=as_of,
                           exit_price=cur_close, exit_reason="reversal")
                n_closed += 1
                open_idx_by_asset.pop(asset, None)
                new_rows.append({
                    "trade_id": _next_trade_id(journal) + len(new_rows),
                    "asset": asset,
                    "side": target_side,
                    "opened_run_ts": run_ts,
                    "entry_bar_close_ts": as_of,
                    "entry_price_recommended": cur_close,
                    "size_units": float(r.get("target_units", 0) or 0),
                    "stop_price": float(r["stop_price"]) if r.get("stop_price") else None,
                    "allocated_nav_sgd": float(r.get("allocated_nav", 0) or 0),
                    "selected_strategy": r["selected_strategy"],
                    "param_label": r["param_label"],
                    "status": "open",
                    "closed_run_ts": pd.NA, "exit_bar_close_ts": pd.NA,
                    "exit_price_recommended": pd.NA, "exit_reason": pd.NA,
                    "gross_pnl_sgd": pd.NA, "gross_pnl_pct": pd.NA,
                    "followed_rules": pd.NA, "notes": pd.NA,
                })
                n_opened += 1
        # else: state and journal both flat OR both consistent open → nothing to do.

    if new_rows:
        journal = pd.concat([journal, pd.DataFrame(new_rows)], ignore_index=True)

    save_journal(journal, journal_path)
    return {"opened": n_opened, "closed": n_closed, "stopped": n_stopped,
            "total_rows": len(journal)}
