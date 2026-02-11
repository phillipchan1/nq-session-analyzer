#!/usr/bin/env python3
"""
Gap Priority Analysis
--------------------

When multiple gaps exist (15m, 1h, 4h, daily), which gets hit first?
Analyzes gap fill priority based on distance, size, and timeframe.

Questions:
- Does distance matter more than gap size?
- Do larger timeframe gaps (4h, daily) get priority over smaller ones (15m)?
- What's the typical order of gap fills?
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
from collections import defaultdict
import re

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20250926.ohlcv-1m.csv")

RTH_START, RTH_END = time(9, 30), time(16, 0)
NY_OPEN = time(9, 30)
FIRST_45_END = time(10, 15)
CHUNKSIZE = 1_000_000
EPSILON = 0.1

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


def find_gaps(df: pd.DataFrame, gap_minutes: int, ny_open_ts) -> List[Dict]:
    """Find unmitigated gaps of specified duration."""
    gaps = []
    if len(df) < 2:
        return gaps
    
    df_sorted = df.sort_values("ts_et")
    
    for i in range(len(df_sorted) - 1):
        current = df_sorted.iloc[i]
        next_bar = df_sorted.iloc[i + 1]
        
        time_diff = (next_bar['ts_et'] - current['ts_et']).total_seconds() / 60
        if time_diff >= gap_minutes:
            gap_high = current['high']
            gap_low = next_bar['low']
            
            if gap_high < gap_low:  # Upward gap
                gap_size = gap_low - gap_high
                gap_level = gap_high
                
                # Check if gap is unmitigated
                gap_filled = False
                after_gap = df_sorted[df_sorted['ts_et'] > next_bar['ts_et']]
                before_open = after_gap[after_gap['ts_et'] < ny_open_ts]
                if not before_open.empty and (before_open['low'] <= gap_level).any():
                    gap_filled = True
                
                if not gap_filled:
                    gaps.append({
                        'timeframe': gap_minutes,
                        'type': 'up',
                        'level': gap_level,
                        'gap_size': gap_size,
                        'gap_start': current['ts_et'],
                        'gap_end': next_bar['ts_et'],
                    })
            
            elif gap_low > gap_high:  # Downward gap
                gap_size = gap_high - gap_low
                gap_level = gap_low
                
                gap_filled = False
                after_gap = df_sorted[df_sorted['ts_et'] > next_bar['ts_et']]
                before_open = after_gap[after_gap['ts_et'] < ny_open_ts]
                if not before_open.empty and (before_open['high'] >= gap_level).any():
                    gap_filled = True
                
                if not gap_filled:
                    gaps.append({
                        'timeframe': gap_minutes,
                        'type': 'down',
                        'level': gap_level,
                        'gap_size': gap_size,
                        'gap_start': current['ts_et'],
                        'gap_end': next_bar['ts_et'],
                    })
    
    return gaps


def find_daily_gap(df: pd.DataFrame, current_date, prev_day_data: Optional[pd.DataFrame]) -> Optional[Dict]:
    """Find daily (overnight) gap."""
    if prev_day_data is None or prev_day_data.empty:
        return None
    
    # Get previous day close
    prev_rth = prev_day_data[_within(prev_day_data["ts_et"], RTH_START, RTH_END)]
    if prev_rth.empty:
        return None
    
    prev_close = prev_rth["close"].iloc[-1]
    
    # Get current day open
    current_open_mask = _within(df["ts_et"], NY_OPEN, time(9, 31))
    current_open = df[current_open_mask]
    if current_open.empty:
        return None
    
    current_open_price = current_open.iloc[0]["open"]
    
    # Check for gap
    if current_open_price > prev_close:
        # Gap up
        gap_size = current_open_price - prev_close
        return {
            'timeframe': 'daily',
            'type': 'up',
            'level': prev_close,
            'gap_size': gap_size,
            'gap_start': prev_rth["ts_et"].iloc[-1],
            'gap_end': current_open["ts_et"].iloc[0],
        }
    elif current_open_price < prev_close:
        # Gap down
        gap_size = prev_close - current_open_price
        return {
            'timeframe': 'daily',
            'type': 'down',
            'level': prev_close,
            'gap_size': gap_size,
            'gap_start': prev_rth["ts_et"].iloc[-1],
            'gap_end': current_open["ts_et"].iloc[0],
        }
    
    return None


def track_gap_fills(df_1m: pd.DataFrame, gaps: List[Dict], open_price: float, 
                    start_time: time, end_time: time) -> List[Dict]:
    """Track which gaps get filled and in what order."""
    fills = []
    
    window_mask = _within(df_1m["ts_et"], start_time, end_time)
    window_data = df_1m[window_mask].copy()
    
    if window_data.empty:
        return fills
    
    open_time = df_1m[_within(df_1m["ts_et"], NY_OPEN, time(9, 31))]["ts_et"].iloc[0]
    
    for gap in gaps:
        gap_level = gap["level"]
        gap_type = gap["type"]
        gap_timeframe = gap["timeframe"]
        
        filled = False
        fill_time = None
        fill_minutes = None
        
        if gap_type == "up":
            # Gap up - filled when price comes back down
            fill_mask = window_data["low"] <= gap_level + EPSILON
            if fill_mask.any():
                filled = True
                first_fill_idx = window_data[fill_mask].index[0]
                fill_time = window_data.loc[first_fill_idx, "ts_et"]
                fill_minutes = (fill_time - open_time).total_seconds() / 60.0
        else:
            # Gap down - filled when price comes back up
            fill_mask = window_data["high"] >= gap_level - EPSILON
            if fill_mask.any():
                filled = True
                first_fill_idx = window_data[fill_mask].index[0]
                fill_time = window_data.loc[first_fill_idx, "ts_et"]
                fill_minutes = (fill_time - open_time).total_seconds() / 60.0
        
        if filled:
            fills.append({
                "timeframe": gap_timeframe,
                "type": gap_type,
                "level": gap_level,
                "gap_size": gap["gap_size"],
                "distance_from_open": abs(gap_level - open_price),
                "fill_time": fill_time,
                "fill_minutes": fill_minutes,
            })
    
    # Sort by fill time
    fills.sort(key=lambda x: x["fill_minutes"])
    
    # Add fill order
    for i, fill in enumerate(fills):
        fill["fill_order"] = i + 1
    
    return fills


def process_day(df_1m: pd.DataFrame, current_date, prev_day_data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """Process a single trading day for gap priority analysis."""
    # Get 9:30 opening price
    open_930_mask = _within(df_1m["ts_et"], NY_OPEN, time(9, 31))
    open_930 = df_1m[open_930_mask]
    if open_930.empty:
        return None
    
    open_price = open_930.iloc[0]["open"]
    ny_open_ts = open_930.iloc[0]["ts_et"]
    
    # Find all gaps
    all_gaps = []
    
    # Time-based gaps
    for gap_minutes in [15, 60, 240]:
        gaps = find_gaps(df_1m, gap_minutes, ny_open_ts)
        all_gaps.extend(gaps)
    
    # Daily gap
    daily_gap = find_daily_gap(df_1m, current_date, prev_day_data)
    if daily_gap:
        all_gaps.append(daily_gap)
    
    if not all_gaps:
        return None
    
    # Track gap fills in first 45 minutes
    fills = track_gap_fills(df_1m, all_gaps, open_price, NY_OPEN, FIRST_45_END)
    
    if not fills:
        return None
    
    # Build result
    result = {
        "date": current_date,
        "open_price": open_price,
        "total_gaps": len(all_gaps),
        "gaps_filled": len(fills),
        "fill_rate": len(fills) / len(all_gaps) * 100 if all_gaps else 0,
    }
    
    # Add gap details
    for i, gap in enumerate(all_gaps):
        result[f"gap_{i+1}_timeframe"] = gap["timeframe"]
        result[f"gap_{i+1}_type"] = gap["type"]
        result[f"gap_{i+1}_level"] = gap["level"]
        result[f"gap_{i+1}_size"] = gap["gap_size"]
        result[f"gap_{i+1}_distance"] = abs(gap["level"] - open_price)
    
    # Add fill order details
    for fill in fills:
        order = fill["fill_order"]
        result[f"fill_{order}_timeframe"] = fill["timeframe"]
        result[f"fill_{order}_type"] = fill["type"]
        result[f"fill_{order}_level"] = fill["level"]
        result[f"fill_{order}_minutes"] = fill["fill_minutes"]
        result[f"fill_{order}_distance"] = fill["distance_from_open"]
        result[f"fill_{order}_size"] = fill["gap_size"]
    
    return result


def main():
    """Main analysis function."""
    print("Starting Gap Priority Analysis...")
    
    tz_et = pytz.timezone("US/Eastern")
    all_results = []
    prev_day_cache = {}
    residual = pd.DataFrame()
    
    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    dtypes = {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "symbol": "string"
    }
    
    # Process in chunks
    for chunk_num, chunk in enumerate(pd.read_csv(CSV_FILE, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE)):
        print(f"Processing chunk {chunk_num + 1}...")
        
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
        dates = sorted(to_proc["et_date"].unique())
        for d in tqdm(dates, desc=f"Chunk {chunk_num + 1} days"):
            day_data = to_proc[to_proc["et_date"] == d].copy()
            day_data = day_data.sort_values("ts_et").reset_index(drop=True)
            
            # Get previous day data
            prev_date = d - timedelta(days=1)
            prev_day_data = None
            if prev_date in prev_day_cache:
                prev_day_data = prev_day_cache[prev_date]
            elif prev_date in dates:
                prev_day_mask = to_proc["et_date"] == prev_date
                if prev_day_mask.any():
                    prev_day_data = to_proc[prev_day_mask].copy()
                    prev_day_cache[prev_date] = prev_day_data
            
            result = process_day(day_data, d, prev_day_data)
            if result:
                all_results.append(result)
            
            # Cache current day
            prev_day_cache[d] = day_data.copy()
    
    if not all_results:
        print("No results found!")
        return
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save detailed results
    detailed_file = OUT_DIR / "gap_priority_detailed.csv"
    df_results.to_csv(detailed_file, index=False)
    print(f"Saved detailed results to {detailed_file}")
    
    # Analyze fill priority by timeframe
    print("\nAnalyzing gap fill priorities...")
    
    # Count first fills by timeframe
    first_fills = defaultdict(int)
    for _, row in df_results.iterrows():
        if pd.notna(row.get("fill_1_timeframe")):
            tf = row["fill_1_timeframe"]
            first_fills[tf] += 1
    
    priority_df = pd.DataFrame([
        {"timeframe": tf, "first_fill_count": count, "first_fill_pct": count / len(df_results) * 100}
        for tf, count in first_fills.items()
    ]).sort_values("first_fill_count", ascending=False)
    
    priority_file = OUT_DIR / "gap_priority_summary.csv"
    priority_df.to_csv(priority_file, index=False)
    print(f"Saved priority summary to {priority_file}")
    
    print(f"\nTotal days analyzed: {len(df_results)}")
    print("\nFirst fill by timeframe:")
    for _, row in priority_df.iterrows():
        print(f"  {row['timeframe']}: {row['first_fill_count']} times ({row['first_fill_pct']:.1f}%)")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()





