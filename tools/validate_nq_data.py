#!/usr/bin/env python3
"""
NQ Data Validation: contract rollover, cleanliness, spot-checks.

Validates the glbx-mdp3 OHLCV data used by analyses:
- Front-month selection trace (which contract per day)
- Rollover-day report (when symbol changes)
- Duplicate / overlap check
- Spot-check specific dates (6am H/L, 10:00-16:00 range, sweep detection)

Usage:
    python tools/validate_nq_data.py

Outputs:
    data/data_validation_front_month_trace.csv
    data/data_validation_rollover_report.txt
"""

import re
from pathlib import Path
from datetime import time

import pandas as pd
import pytz

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20260221.ohlcv-1m.csv")
TRACE_OUT = str(DATA_DIR / "data_validation_front_month_trace.csv")
ROLLOVER_OUT = str(DATA_DIR / "data_validation_rollover_report.txt")

CHUNKSIZE = 1_000_000
NY_OPEN = time(9, 30)
NQ_RE = re.compile(r"^NQ[A-Z]\d{1,2}$")
SPOT_CHECK_DATES = ["2025-12-04", "2026-01-26"]
# ==============================================


def _within(ts, start_t, end_t):
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)


def _front_month_with_vol(df):
    """Return (df filtered to front-month, trace rows: list of (et_date, symbol, vol))."""
    rth = df.loc[_within(df["ts_et"], NY_OPEN, time(9, 45))]
    if rth.empty:
        return df, []
    vol = (
        rth.groupby(["et_date", "symbol"])
        .agg(volume=("volume", "sum"))
        .reset_index()
    )
    best = vol.loc[
        vol.groupby("et_date")["volume"].idxmax(),
        ["et_date", "symbol", "volume"],
    ].rename(columns={"symbol": "front", "volume": "vol_9_30_9_45"})
    trace = best[["et_date", "front", "vol_9_30_9_45"]].rename(columns={"front": "symbol"})
    out = df.merge(best[["et_date", "front"]], on="et_date", how="left")
    out = out.loc[out["symbol"] == out["front"]].drop(columns=["front"])
    return out, trace.to_dict("records")


def run_validation():
    tz = pytz.timezone("US/Eastern")
    all_trace = []
    residual = pd.DataFrame()
    chunk_n = 0
    duplicate_ts_symbol_count = 0
    spread_count = 0
    outright_count = 0

    print("=" * 60)
    print("NQ Data Validation")
    print("=" * 60)
    print(f"Input: {CSV_FILE}")
    print()

    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    dtypes = {
        "ts_event": "string",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "int64",
        "symbol": "string",
    }

    for chunk in pd.read_csv(CSV_FILE, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE):
        chunk_n += 1
        df = pd.concat([residual, chunk], ignore_index=True)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz)
        df["et_date"] = df["ts_et"].dt.date
        df = df[df["ts_et"].dt.weekday <= 4]
        if df.empty:
            residual = pd.DataFrame()
            continue

        # Count symbols
        spreads = df[df["symbol"].astype(str).str.contains("-", na=False)]
        outrights = df[df["symbol"].astype(str).str.match(NQ_RE)]
        spread_count += len(spreads)
        outright_count += len(outrights)

        # Duplicate check: (ts_event, symbol) should be unique
        dup = df[df.duplicated(subset=["ts_event", "symbol"], keep=False)]
        if not dup.empty:
            duplicate_ts_symbol_count += len(dup)

        df = outrights.copy()
        if df.empty:
            residual = pd.DataFrame()
            continue

        max_date = df["et_date"].max()
        to_proc = df[df["et_date"] < max_date]
        residual = df[df["et_date"] == max_date]

        if to_proc.empty:
            continue

        to_proc, trace = _front_month_with_vol(to_proc)
        for row in trace:
            row["et_date"] = row["et_date"].isoformat() if hasattr(row["et_date"], "isoformat") else str(row["et_date"])
            all_trace.append(row)
        print(f"  Chunk {chunk_n}: {len(trace)} days traced")

    if not residual.empty:
        residual, trace = _front_month_with_vol(residual)
        for row in trace:
            row["et_date"] = row["et_date"].isoformat() if hasattr(row["et_date"], "isoformat") else str(row["et_date"])
            all_trace.append(row)
        print(f"  Residual: {len(trace)} days traced")

    # ---- 1. Front-month trace ----
    trace_df = pd.DataFrame(all_trace)
    if not trace_df.empty:
        trace_df = trace_df.drop_duplicates(subset=["et_date"], keep="last")
        trace_df.to_csv(TRACE_OUT, index=False)
        print(f"\n1. Front-month trace: {len(trace_df)} days -> {TRACE_OUT}")

    # ---- 2. Rollover report ----
    rollover_rows = []
    if len(trace_df) >= 2:
        trace_df = trace_df.sort_values("et_date").reset_index(drop=True)
        prev_sym = None
        for _, row in trace_df.iterrows():
            sym = row["symbol"]
            if prev_sym is not None and sym != prev_sym:
                rollover_rows.append({"date": row["et_date"], "from": prev_sym, "to": sym})
            prev_sym = sym

    with open(ROLLOVER_OUT, "w") as f:
        f.write("NQ Data Validation - Rollover Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total trading days: {len(trace_df)}\n")
        f.write(f"Rollover days (symbol changed): {len(rollover_rows)}\n\n")
        f.write("Rollover events (from -> to):\n")
        for r in rollover_rows[:50]:
            f.write(f"  {r['date']}: {r['from']} -> {r['to']}\n")
        if len(rollover_rows) > 50:
            f.write(f"  ... and {len(rollover_rows) - 50} more\n")
    print(f"2. Rollover report: {len(rollover_rows)} rollover days -> {ROLLOVER_OUT}")

    # ---- 3. Duplicate / overlap summary ----
    print("\n3. Data cleanliness:")
    print(f"   Outright rows (NQ[A-Z]dd): {outright_count:,}")
    print(f"   Spread rows (excluded): {spread_count:,}")
    if duplicate_ts_symbol_count > 0:
        print(f"   WARNING: {duplicate_ts_symbol_count} duplicate (ts_event, symbol) rows")
    else:
        print("   OK: No duplicate (ts_event, symbol) rows")

    # ---- 4. Spot-check specific dates ----
    print("\n4. Spot-check (6am H/L, 10:00-16:00 range, sweep):")
    target_dates = [pd.to_datetime(d).date() for d in SPOT_CHECK_DATES]
    spot_data = {d: [] for d in target_dates}

    for chunk in pd.read_csv(CSV_FILE, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE):
        chunk["ts_event"] = pd.to_datetime(chunk["ts_event"], utc=True, errors="coerce")
        chunk["ts_et"] = chunk["ts_event"].dt.tz_convert(tz)
        chunk["et_date"] = chunk["ts_et"].dt.date
        chunk = chunk[chunk["symbol"].astype(str).str.match(NQ_RE)]
        for d in target_dates:
            day_chunk = chunk[chunk["et_date"] == d]
            if not day_chunk.empty:
                spot_data[d].append(day_chunk)

    for target_date_str in SPOT_CHECK_DATES:
        target_date = pd.to_datetime(target_date_str).date()
        parts = spot_data.get(target_date, [])
        if not parts:
            print(f"   {target_date_str}: No data")
            continue
        day_data = pd.concat(parts, ignore_index=True)

        rth = day_data[_within(day_data["ts_et"], NY_OPEN, time(9, 45))]
        if rth.empty:
            sym = day_data["symbol"].iloc[0]
        else:
            vol = rth.groupby("symbol")["volume"].sum()
            sym = vol.idxmax()
        g = day_data[day_data["symbol"] == sym].sort_values("ts_et")

        candle_bars = g[_within(g["ts_et"], time(6, 0), time(10, 0))]
        sweep_bars = g[_within(g["ts_et"], time(10, 0), time(16, 0))]

        if candle_bars.empty:
            print(f"   {target_date_str}: No 6am candle bars")
            continue

        candle_high = candle_bars["high"].max()
        candle_low = candle_bars["low"].min()
        candle_close = candle_bars.iloc[-1]["close"]

        if sweep_bars.empty:
            max_high_10_16 = None
            min_low_10_16 = None
            high_swept = False
            low_swept = False
        else:
            max_high_10_16 = sweep_bars["high"].max()
            min_low_10_16 = sweep_bars["low"].min()
            high_swept = (sweep_bars["high"] > candle_high + 0.25).any()
            low_swept = (sweep_bars["low"] < candle_low - 0.25).any()

        print(f"   {target_date_str} ({sym}):")
        print(f"      6am candle: H={candle_high:.2f} L={candle_low:.2f} close={candle_close:.2f}")
        if max_high_10_16 is not None:
            print(f"      10:00-16:00: max_high={max_high_10_16:.2f} min_low={min_low_10_16:.2f}")
            print(f"      High swept (>{candle_high + 0.25}): {high_swept}")
            print(f"      Low swept (<{candle_low - 0.25}): {low_swept}")
        else:
            print("      10:00-16:00: No bars")

    print("\n" + "=" * 60)
    if duplicate_ts_symbol_count == 0:
        print("Data validation: OK (no duplicate ts_event+symbol)")
    else:
        print("Data validation: ISSUES (see above)")
    print("=" * 60)


if __name__ == "__main__":
    run_validation()
