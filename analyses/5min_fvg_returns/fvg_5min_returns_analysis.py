#!/usr/bin/env python3
"""
5-Minute Fair Value Gap Returns Analysis
----------------------------------------

Analyzes 5-minute fair value gaps (FVGs) created in the first 45 minutes
(9:30-10:15 ET) and determines the probability that price returns to
(touches) those gaps within the first 45 minutes.

Key Questions:
- If a 5-min FVG is created in the first 45 minutes, what are the chances
  that price will return to it within the first 45 minutes?
- Day-by-day tracking of FVG creation and return behavior
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import time, timedelta
import pytz
from tqdm import tqdm

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20250926.ohlcv-1m.csv")

RTH_START, RTH_END = time(9, 30), time(16, 0)
FIRST_45MIN_START = time(9, 30)
FIRST_45MIN_END = time(10, 15)
FVG_CREATION_CUTOFF = time(10, 10)  # Ignore FVGs created at 10:10 or later

CHUNKSIZE = 1_000_000
EPSILON = 0.1  # Price tolerance for gap touches

# =================== HELPER FUNCTIONS ===================

def _within(ts: pd.Series, start_t: time, end_t: time) -> pd.Series:
    """Check if timestamps fall within time range."""
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)


def _determine_front_month(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to front-month per date by max volume in first 15 min."""
    mask = _within(df["ts_et"], RTH_START, time(9, 45))
    rth = df.loc[mask]
    if rth.empty:
        return df
    agg = rth.groupby(["et_date", "symbol"]).agg(volume=("volume", "sum")).reset_index()
    idx = agg.groupby("et_date")["volume"].idxmax()
    fm = agg.loc[idx][["et_date", "symbol"]].rename(columns={"symbol": "front_symbol"})
    out = df.merge(fm, on="et_date", how="left")
    out = out.loc[out["symbol"] == out["front_symbol"]].drop(columns=["front_symbol"])
    return out


def resample_to_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute data to 5-minute candles."""
    df_1m = df_1m.sort_values("ts_et").reset_index(drop=True)
    
    # Use pandas resample (more robust)
    # Make sure ts_et is timezone-aware
    df_1m_indexed = df_1m.set_index("ts_et")
    agg = df_1m_indexed.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])
    
    # Reset index
    agg = agg.reset_index()
    
    # Ensure ts_et is timezone-aware before extracting time
    if agg["ts_et"].dt.tz is None:
        print("WARNING: ts_et has no timezone!")
    
    agg["time"] = agg["ts_et"].dt.time
    
    return agg


def detect_5min_fvgs(df_5m: pd.DataFrame, creation_start: time, creation_end: time, debug_counters: Optional[Dict] = None) -> List[Dict]:
    """
    Detect 5-minute Fair Value Gaps created within the specified time window.
    
    FVG definition:
    - Bullish FVG: gap up where curr["low"] > prev["high"]
    - Bearish FVG: gap down where curr["high"] < prev["low"]
    - Gap must not fill in the next 2 candles to be considered valid
    
    Returns list of FVG dictionaries with:
    - fvg_type: "bullish" or "bearish"
    - fvg_low: lower bound of gap
    - fvg_high: upper bound of gap
    - created_ts: timestamp when FVG was created
    - created_time: time when FVG was created
    - candle_idx: index of candle that created the FVG
    """
    fvgs = []
    
    if len(df_5m) < 2:
        return fvgs
    
    # Debug counters
    gaps_found = 0
    gaps_filled = 0
    
    if debug_counters is not None:
        debug_counters["gaps_found"] = debug_counters.get("gaps_found", 0)
        debug_counters["gaps_filled"] = debug_counters.get("gaps_filled", 0)
    
    # FVG is a 3-candle pattern
    # Iterate through candles, checking for FVG pattern
    # FVG requires 3 candles: candle[i-2], candle[i-1], candle[i]
    for i in range(2, len(df_5m)):
        candle1 = df_5m.iloc[i-2]  # First candle
        candle2 = df_5m.iloc[i-1]  # Middle candle  
        candle3 = df_5m.iloc[i]    # Third candle (creates the FVG)
        
        # Check if third candle is within creation window
        candle3_time = candle3["time"]
        if not (creation_start <= candle3_time < creation_end):
            continue
        
        # Check creation time cutoff - ignore FVGs created at 10:10 or later
        if candle3_time >= FVG_CREATION_CUTOFF:
            continue
        
        # Bullish FVG: candle3's low > candle1's high
        if candle3["low"] > candle1["high"]:
            gaps_found += 1
            gap_low = candle1["high"]
            gap_high = candle3["low"]
            
            # Check if gap fills in next 2 candles
            filled = False
            for j in range(i+1, min(i+3, len(df_5m))):
                if df_5m.iloc[j]["low"] <= gap_low:
                    filled = True
                    gaps_filled += 1
                    break
            
            if not filled:
                fvgs.append({
                    "fvg_type": "bullish",
                    "fvg_low": gap_low,
                    "fvg_high": gap_high,
                    "created_ts": candle3["ts_et"],
                    "created_time": candle3_time,
                    "candle_idx": i,
                    "gap_size": gap_high - gap_low,
                })
        
        # Bearish FVG: candle3's high < candle1's low
        elif candle3["high"] < candle1["low"]:
            gaps_found += 1
            gap_low = candle3["high"]
            gap_high = candle1["low"]
            
            # Check if gap fills in next 2 candles
            filled = False
            for j in range(i+1, min(i+3, len(df_5m))):
                if df_5m.iloc[j]["high"] >= gap_high:
                    filled = True
                    gaps_filled += 1
                    break
            
            if not filled:
                fvgs.append({
                    "fvg_type": "bearish",
                    "fvg_low": gap_low,
                    "fvg_high": gap_high,
                    "created_ts": candle3["ts_et"],
                    "created_time": candle3_time,
                    "candle_idx": i,
                    "gap_size": gap_high - gap_low,
                })
    
    # Update debug counters
    if debug_counters is not None:
        debug_counters["gaps_found"] += gaps_found
        debug_counters["gaps_filled"] += gaps_filled
    
    return fvgs


def check_fvg_return(df_1m: pd.DataFrame, fvg: Dict, check_start: time, check_end: time) -> Dict:
    """
    Check if price returns to (touches) the FVG within the specified time window.
    
    Returns dictionary with:
    - returned: bool
    - first_touch_ts: timestamp of first touch (if returned)
    - first_touch_time: time of first touch (if returned)
    - minutes_after_creation: minutes between creation and first touch
    """
    result = {
        "returned": False,
        "first_touch_ts": None,
        "first_touch_time": None,
        "minutes_after_creation": None,
    }
    
    # Filter to check window
    check_mask = _within(df_1m["ts_et"], check_start, check_end)
    check_data = df_1m[check_mask].copy()
    
    if check_data.empty:
        return result
    
    # Only check after FVG creation
    fvg_created_ts = fvg["created_ts"]
    check_data = check_data[check_data["ts_et"] > fvg_created_ts]
    
    if check_data.empty:
        return result
    
    fvg_low = fvg["fvg_low"]
    fvg_high = fvg["fvg_high"]
    
    # Check if price touches the gap zone
    # Touch means: low <= fvg_high AND high >= fvg_low
    touch_mask = (check_data["low"] <= fvg_high + EPSILON) & (check_data["high"] >= fvg_low - EPSILON)
    
    if touch_mask.any():
        result["returned"] = True
        first_touch_idx = check_data[touch_mask].index[0]
        first_touch_ts = check_data.loc[first_touch_idx, "ts_et"]
        result["first_touch_ts"] = first_touch_ts
        result["first_touch_time"] = first_touch_ts.time()
        
        # Calculate minutes after creation
        time_diff = first_touch_ts - fvg_created_ts
        result["minutes_after_creation"] = time_diff.total_seconds() / 60.0
    
    return result


def process_day(g: pd.DataFrame, debug_counters: Optional[Dict] = None) -> Optional[Dict]:
    """Process a single trading day."""
    # Filter to RTH
    rth_mask = _within(g["ts_et"], RTH_START, RTH_END)
    rth_1m = g.loc[rth_mask].copy()
    
    if rth_1m.empty:
        return None
    
    # Resample to 5-minute candles
    df_5m = resample_to_5min(rth_1m)
    
    if len(df_5m) < 2:
        return None
    
    # Detect FVGs created in first 45 minutes (before 10:10 cutoff)
    fvgs = detect_5min_fvgs(df_5m, FIRST_45MIN_START, FIRST_45MIN_END, debug_counters)
    
    if not fvgs:
        return None
    
    # Check if each FVG returns within first 45 minutes
    fvg_results = []
    for fvg in fvgs:
        return_check = check_fvg_return(rth_1m, fvg, FIRST_45MIN_START, FIRST_45MIN_END)
        
        fvg_result = {
            **fvg,
            **return_check,
        }
        fvg_results.append(fvg_result)
    
    # Get opening price for context
    open_mask = _within(rth_1m["ts_et"], FIRST_45MIN_START, time(9, 31))
    open_price = rth_1m[open_mask].iloc[0]["open"] if not open_mask.empty else None
    
    # Get first 45min high/low for context
    first_45min_mask = _within(rth_1m["ts_et"], FIRST_45MIN_START, FIRST_45MIN_END)
    first_45min_data = rth_1m[first_45min_mask]
    first_45min_high = first_45min_data["high"].max() if not first_45min_data.empty else None
    first_45min_low = first_45min_data["low"].min() if not first_45min_data.empty else None
    
    result = {
        "date": g["et_date"].iloc[0],
        "symbol": g["symbol"].iloc[0],
        "open_price": open_price,
        "first_45min_high": first_45min_high,
        "first_45min_low": first_45min_low,
        "num_fvgs": len(fvgs),
        "fvgs": fvg_results,
    }
    
    # Store debug info if counters were passed
    if debug_counters is not None:
        result["debug_gaps_found"] = debug_counters.get("gaps_found", 0)
        result["debug_gaps_filled"] = debug_counters.get("gaps_filled", 0)
    
    return result


def main():
    """Main processing function."""
    print("Starting 5-minute FVG returns analysis...")
    print(f"Analyzing FVGs created in {FIRST_45MIN_START}-{FIRST_45MIN_END}")
    print(f"Excluding FVGs created at {FVG_CREATION_CUTOFF} or later")
    print(f"Checking returns within {FIRST_45MIN_START}-{FIRST_45MIN_END}")
    
    tz_et = pytz.timezone("US/Eastern")
    residual = pd.DataFrame()
    
    # Storage for all days
    all_days_data = []
    all_fvg_details = []
    days_processed = 0
    days_with_fvgs = 0
    total_gaps_found = 0
    total_gaps_filled = 0
    
    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    dtypes = {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "symbol": "string"
    }
    
    print("\nLoading and processing data...")
    
    for chunk_num, chunk in enumerate(pd.read_csv(CSV_FILE, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE)):
        print(f"\nProcessing chunk {chunk_num + 1}...")
        
        df = pd.concat([residual, chunk], ignore_index=True)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz_et)
        df["et_date"] = df["ts_et"].dt.date
        
        # Weekdays only
        df = df[df["ts_et"].dt.weekday <= 4]
        if df.empty:
            continue
        
        # Keep only outright NQ contracts
        df = df[df["symbol"].astype(str).str.startswith("NQ") & ~df["symbol"].astype(str).str.contains("-")]
        
        # Split off final day
        max_date = df["et_date"].max()
        to_proc = df[df["et_date"] < max_date]
        residual = df[df["et_date"] == max_date]
        
        # Filter to front-month
        if not to_proc.empty:
            to_proc = _determine_front_month(to_proc)
        
        # Process each day
        for (sym, d), g in tqdm(to_proc.groupby(["symbol", "et_date"]), desc=f"Chunk {chunk_num + 1} days"):
            g = g.sort_values("ts_et").reset_index(drop=True)
            days_processed += 1
            
            day_data = process_day(g)
            
            if day_data is None:
                continue
            
            days_with_fvgs += 1
            
            all_days_data.append({
                "date": day_data["date"],
                "symbol": day_data["symbol"],
                "open_price": day_data["open_price"],
                "first_45min_high": day_data["first_45min_high"],
                "first_45min_low": day_data["first_45min_low"],
                "num_fvgs": day_data["num_fvgs"],
            })
            
            # Store detailed FVG results
            for fvg in day_data["fvgs"]:
                all_fvg_details.append({
                    "date": day_data["date"],
                    "symbol": day_data["symbol"],
                    "open_price": day_data["open_price"],
                    "first_45min_high": day_data["first_45min_high"],
                    "first_45min_low": day_data["first_45min_low"],
                    "fvg_type": fvg["fvg_type"],
                    "fvg_low": fvg["fvg_low"],
                    "fvg_high": fvg["fvg_high"],
                    "gap_size": fvg["gap_size"],
                    "created_time": fvg["created_time"],
                    "returned": fvg["returned"],
                    "first_touch_time": fvg["first_touch_time"],
                    "minutes_after_creation": fvg["minutes_after_creation"],
                })
    
    # Process final residual
    if not residual.empty:
        residual = _determine_front_month(residual)
        for (sym, d), g in tqdm(residual.groupby(["symbol", "et_date"]), desc="Final residual"):
            g = g.sort_values("ts_et").reset_index(drop=True)
            days_processed += 1
            
            debug_counters = {"gaps_found": 0, "gaps_filled": 0}
            day_data = process_day(g, debug_counters)
            
            if day_data is None:
                # Still count gaps found even if no FVGs survived
                total_gaps_found += debug_counters.get("gaps_found", 0)
                total_gaps_filled += debug_counters.get("gaps_filled", 0)
                continue
            
            days_with_fvgs += 1
            total_gaps_found += debug_counters.get("gaps_found", 0)
            total_gaps_filled += debug_counters.get("gaps_filled", 0)
            
            all_days_data.append({
                "date": day_data["date"],
                "symbol": day_data["symbol"],
                "open_price": day_data["open_price"],
                "first_45min_high": day_data["first_45min_high"],
                "first_45min_low": day_data["first_45min_low"],
                "num_fvgs": day_data["num_fvgs"],
            })
            
            for fvg in day_data["fvgs"]:
                all_fvg_details.append({
                    "date": day_data["date"],
                    "symbol": day_data["symbol"],
                    "open_price": day_data["open_price"],
                    "first_45min_high": day_data["first_45min_high"],
                    "first_45min_low": day_data["first_45min_low"],
                    "fvg_type": fvg["fvg_type"],
                    "fvg_low": fvg["fvg_low"],
                    "fvg_high": fvg["fvg_high"],
                    "gap_size": fvg["gap_size"],
                    "created_time": fvg["created_time"],
                    "returned": fvg["returned"],
                    "first_touch_time": fvg["first_touch_time"],
                    "minutes_after_creation": fvg["minutes_after_creation"],
                })
    
    # Create DataFrames
    print(f"\nDays processed: {days_processed}")
    print(f"Days with FVGs: {days_with_fvgs}")
    print(f"Total gaps found: {total_gaps_found}")
    print(f"Total gaps filled immediately: {total_gaps_filled}")
    print(f"Total FVG records: {len(all_fvg_details)}")
    
    if not all_days_data:
        print("\nNo FVGs found in the dataset!")
        print("This suggests 5-minute FVGs are extremely rare during the first 45 minutes.")
        print("See README.md for possible reasons and alternative approaches.")
        
        # Create empty DataFrames with proper columns
        df_days = pd.DataFrame(columns=["date", "symbol", "open_price", "first_45min_high", 
                                       "first_45min_low", "num_fvgs", "fvgs_returned", 
                                       "fvgs_total", "return_rate"])
        df_fvgs = pd.DataFrame(columns=["date", "symbol", "open_price", "first_45min_high",
                                       "first_45min_low", "fvg_type", "fvg_low", "fvg_high",
                                       "gap_size", "created_time", "returned", "first_touch_time",
                                       "minutes_after_creation"])
    else:
        df_days = pd.DataFrame(all_days_data)
        df_fvgs = pd.DataFrame(all_fvg_details)
    
    # Calculate summary statistics
    if len(df_fvgs) > 0:
        total_fvgs = len(df_fvgs)
        returned_fvgs = df_fvgs["returned"].sum()
        return_rate = (returned_fvgs / total_fvgs * 100) if total_fvgs > 0 else 0
        
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Total days analyzed: {len(df_days)}")
        print(f"Total FVGs created: {total_fvgs}")
        print(f"FVGs that returned: {returned_fvgs}")
        print(f"Return rate: {return_rate:.2f}%")
        
        if returned_fvgs > 0:
            avg_minutes_to_return = df_fvgs[df_fvgs["returned"]]["minutes_after_creation"].mean()
            print(f"Average minutes to return: {avg_minutes_to_return:.2f}")
        
        # Breakdown by FVG type
        print(f"\n{'='*60}")
        print("BREAKDOWN BY FVG TYPE")
        print(f"{'='*60}")
        for fvg_type in ["bullish", "bearish"]:
            type_fvgs = df_fvgs[df_fvgs["fvg_type"] == fvg_type]
            if len(type_fvgs) > 0:
                type_returned = type_fvgs["returned"].sum()
                type_rate = (type_returned / len(type_fvgs) * 100)
                print(f"{fvg_type.capitalize()} FVGs: {len(type_fvgs)} total, {type_returned} returned ({type_rate:.2f}%)")
        
        # Breakdown by gap size
        print(f"\n{'='*60}")
        print("BREAKDOWN BY GAP SIZE")
        print(f"{'='*60}")
        df_fvgs["gap_size_bucket"] = pd.cut(
            df_fvgs["gap_size"],
            bins=[0, 5, 10, 15, 20, 30, 50, 100, float('inf')],
            labels=["0-5", "5-10", "10-15", "15-20", "20-30", "30-50", "50-100", "100+"]
        )
        for bucket in df_fvgs["gap_size_bucket"].cat.categories:
            bucket_fvgs = df_fvgs[df_fvgs["gap_size_bucket"] == bucket]
            if len(bucket_fvgs) > 0:
                bucket_returned = bucket_fvgs["returned"].sum()
                bucket_rate = (bucket_returned / len(bucket_fvgs) * 100)
                print(f"{bucket} pts: {len(bucket_fvgs)} total, {bucket_returned} returned ({bucket_rate:.2f}%)")
    else:
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Total days analyzed: {days_processed}")
        print(f"Total FVGs created: 0")
        print(f"\nNo FVGs were found. See README.md for analysis and alternative approaches.")
    
    # Save outputs
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print(f"{'='*60}")
    
    # Day-by-day summary
    df_days_summary = df_days.copy()
    if len(df_days_summary) > 0:
        df_days_summary["date"] = pd.to_datetime(df_days_summary["date"])
        df_days_summary = df_days_summary.sort_values("date")
        
        # Add return statistics per day
        if len(df_fvgs) > 0:
            df_fvgs["date"] = pd.to_datetime(df_fvgs["date"])
            daily_stats = df_fvgs.groupby("date").agg({
                "returned": ["sum", "count"]
            }).reset_index()
            daily_stats.columns = ["date", "fvgs_returned", "fvgs_total"]
            daily_stats["return_rate"] = (daily_stats["fvgs_returned"] / daily_stats["fvgs_total"] * 100)
            daily_stats["date"] = pd.to_datetime(daily_stats["date"])
            
            df_days_summary = df_days_summary.merge(daily_stats, on="date", how="left")
        else:
            df_days_summary["fvgs_returned"] = 0
            df_days_summary["fvgs_total"] = 0
            df_days_summary["return_rate"] = 0.0
    if len(df_days_summary) > 0:
        df_days_summary["fvgs_returned"] = df_days_summary["fvgs_returned"].fillna(0).astype(int)
        df_days_summary["fvgs_total"] = df_days_summary["fvgs_total"].fillna(0).astype(int)
        
        days_output = OUT_DIR / "fvg_5min_daily_summary.csv"
        df_days_summary.to_csv(days_output, index=False)
        print(f"Saved daily summary: {days_output}")
    else:
        days_output = OUT_DIR / "fvg_5min_daily_summary.csv"
        df_days.to_csv(days_output, index=False)
        print(f"Saved empty daily summary: {days_output}")
    
    # Detailed FVG tracking
    if len(df_fvgs) > 0:
        df_fvgs["date"] = pd.to_datetime(df_fvgs["date"])
        df_fvgs = df_fvgs.sort_values(["date", "created_time"])
    
    fvgs_output = OUT_DIR / "fvg_5min_detailed.csv"
    df_fvgs.to_csv(fvgs_output, index=False)
    print(f"Saved detailed FVG tracking: {fvgs_output}")
    
    print(f"\nAnalysis complete!")


if __name__ == "__main__":
    main()

