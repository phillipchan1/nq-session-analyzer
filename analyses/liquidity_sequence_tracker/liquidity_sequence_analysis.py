#!/usr/bin/env python3
"""
Liquidity Sequence Tracker
---------------------------

Tracks the order of liquidity hits/sweeps during the first 45 minutes (9:30-10:15 ET),
analyzing sequences like "SSL sweep → failed rally → 930 low → London high".

Tracks:
- Order of liquidity hits/sweeps
- Time between each liquidity hit
- Direction of movement between hits
- Transition probabilities
- Most common sequences
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
from collections import defaultdict, Counter
import re

# Import helper functions from c1_liquidity_sweeps module
# We'll use importlib to load the module
import importlib.util
c1_module_path = Path(__file__).resolve().parents[1] / "c1_liquidity_sweeps" / "c1_liquidity_sweeps.py"
spec = importlib.util.spec_from_file_location("c1_liquidity_sweeps", c1_module_path)
c1_module = importlib.util.module_from_spec(spec)
sys.modules["c1_liquidity_sweeps"] = c1_module
spec.loader.exec_module(c1_module)

# Use functions from the module
_within = c1_module._within
_within_wrap = c1_module._within_wrap
_determine_front_month = c1_module._determine_front_month
build_all_liquidity_levels = c1_module.build_all_liquidity_levels

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

def find_unmitigated_gaps(df: pd.DataFrame, gap_minutes: int) -> List[Dict]:
    """Find unmitigated gaps of specified duration."""
    gaps = []
    if len(df) < 2:
        return gaps
    
    df_sorted = df.sort_values("ts_et")
    ny_open_ts = pd.Timestamp.combine(df_sorted["ts_et"].dt.date.iloc[0], NY_OPEN).tz_localize(df_sorted["ts_et"].dt.tz)
    
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
                        'type': f'{gap_minutes}min_gap_up',
                        'level': gap_level,
                        'side': 'high',
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
                        'type': f'{gap_minutes}min_gap_down',
                        'level': gap_level,
                        'side': 'low',
                        'gap_size': gap_size,
                        'gap_start': current['ts_et'],
                        'gap_end': next_bar['ts_et'],
                    })
    
    return gaps


def track_liquidity_hits_in_window(df_1m: pd.DataFrame, liquidity_levels: List[Dict], 
                                   start_time: time, end_time: time) -> List[Dict]:
    """
    Track when each liquidity level gets hit/swept in a time window.
    Returns list of hits ordered by time.
    """
    hits = []
    
    window_mask = _within(df_1m["ts_et"], start_time, end_time)
    window_data = df_1m[window_mask].copy()
    
    if window_data.empty:
        return hits
    
    open_time = df_1m[_within(df_1m["ts_et"], NY_OPEN, time(9, 31))]["ts_et"].iloc[0]
    
    for level_info in liquidity_levels:
        level_type = level_info["type"]
        level = level_info["level"]
        side = level_info["side"]
        
        hit = False
        hit_timestamp = None
        hit_minutes_after_open = None
        points_beyond = 0.0
        
        if side == "high":
            hit_mask = window_data["high"] >= level - EPSILON
            if hit_mask.any():
                hit = True
                first_hit_idx = window_data[hit_mask].index[0]
                hit_timestamp = window_data.loc[first_hit_idx, "ts_et"]
                hit_minutes_after_open = (hit_timestamp - open_time).total_seconds() / 60.0
                points_beyond = window_data.loc[first_hit_idx, "high"] - level
        
        elif side == "low":
            hit_mask = window_data["low"] <= level + EPSILON
            if hit_mask.any():
                hit = True
                first_hit_idx = window_data[hit_mask].index[0]
                hit_timestamp = window_data.loc[first_hit_idx, "ts_et"]
                hit_minutes_after_open = (hit_timestamp - open_time).total_seconds() / 60.0
                points_beyond = level - window_data.loc[first_hit_idx, "low"]
        
        elif side == "mid":
            high_hit = (window_data["high"] >= level - EPSILON).any()
            low_hit = (window_data["low"] <= level + EPSILON).any()
            if high_hit or low_hit:
                hit = True
                if high_hit and low_hit:
                    high_idx = window_data[window_data["high"] >= level - EPSILON].index[0]
                    low_idx = window_data[window_data["low"] <= level + EPSILON].index[0]
                    first_hit_idx = min(high_idx, low_idx)
                elif high_hit:
                    first_hit_idx = window_data[window_data["high"] >= level - EPSILON].index[0]
                else:
                    first_hit_idx = window_data[window_data["low"] <= level + EPSILON].index[0]
                
                hit_timestamp = window_data.loc[first_hit_idx, "ts_et"]
                hit_minutes_after_open = (hit_timestamp - open_time).total_seconds() / 60.0
                points_beyond = min(
                    abs(window_data.loc[first_hit_idx, "high"] - level),
                    abs(level - window_data.loc[first_hit_idx, "low"])
                )
        
        if hit:
            hits.append({
                "liquidity_type": level_type,
                "level": level,
                "side": side,
                "hit_timestamp": hit_timestamp,
                "hit_minutes_after_open": hit_minutes_after_open,
                "points_beyond": points_beyond,
            })
    
    # Sort by hit time
    hits.sort(key=lambda x: x["hit_timestamp"])
    return hits


def process_day(df_1m: pd.DataFrame, current_date, historical_data: Optional[Dict] = None, prev_day_data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """Process a single trading day for sequence analysis."""
    # Filter to RTH
    rth_mask = _within(df_1m["ts_et"], RTH_START, RTH_END)
    rth_data = df_1m[rth_mask].copy()
    
    if rth_data.empty:
        return None
    
    # Get 9:30 opening price
    open_930_mask = _within(df_1m["ts_et"], NY_OPEN, time(9, 31))
    open_930 = df_1m[open_930_mask]
    if open_930.empty:
        return None
    
    open_price = open_930.iloc[0]["open"]
    
    # Build liquidity levels
    try:
        liquidity_levels = build_all_liquidity_levels(df_1m, current_date, prev_day_data, historical_data)
    except Exception as e:
        print(f"Warning: Failed to build liquidity levels for {current_date}: {e}")
        return None
    
    # Add gap levels
    for gap_minutes in [15, 60, 240]:
        gaps = find_unmitigated_gaps(df_1m, gap_minutes)
        for gap in gaps:
            liquidity_levels.append({
                "type": gap["type"],
                "level": gap["level"],
                "side": gap["side"],
                "origin_session": "pre_rth",
                "created_ts": gap["gap_start"],
            })
    
    # Track hits in first 45 minutes
    hits = track_liquidity_hits_in_window(df_1m, liquidity_levels, NY_OPEN, FIRST_45_END)
    
    if not hits:
        return None
    
    # Build sequence
    sequence = [h["liquidity_type"] for h in hits]
    sequence_with_sides = [(h["liquidity_type"], h["side"]) for h in hits]
    
    # Calculate time between hits
    time_deltas = []
    for i in range(1, len(hits)):
        delta_minutes = hits[i]["hit_minutes_after_open"] - hits[i-1]["hit_minutes_after_open"]
        time_deltas.append(delta_minutes)
    
    # Calculate distances traveled
    distances = []
    for i in range(1, len(hits)):
        distance = abs(hits[i]["level"] - hits[i-1]["level"])
        distances.append(distance)
    
    # Determine direction of movement
    directions = []
    for i in range(1, len(hits)):
        if hits[i]["level"] > hits[i-1]["level"]:
            directions.append("up")
        elif hits[i]["level"] < hits[i-1]["level"]:
            directions.append("down")
        else:
            directions.append("same")
    
    result = {
        "date": current_date,
        "open_price": open_price,
        "num_hits": len(hits),
        "sequence": " → ".join(sequence),
        "sequence_list": sequence,
        "first_hit_type": sequence[0] if sequence else None,
        "second_hit_type": sequence[1] if len(sequence) > 1 else None,
        "third_hit_type": sequence[2] if len(sequence) > 2 else None,
        "fourth_hit_type": sequence[3] if len(sequence) > 3 else None,
        "avg_time_between_hits": np.mean(time_deltas) if time_deltas else None,
        "median_time_between_hits": np.median(time_deltas) if time_deltas else None,
        "avg_distance_between_hits": np.mean(distances) if distances else None,
        "median_distance_between_hits": np.median(distances) if distances else None,
        "direction_changes": sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1]) if len(directions) > 1 else 0,
        "first_direction": directions[0] if directions else None,
    }
    
    # Add individual hit details
    for i, hit in enumerate(hits):
        result[f"hit_{i+1}_type"] = hit["liquidity_type"]
        result[f"hit_{i+1}_side"] = hit["side"]
        result[f"hit_{i+1}_level"] = hit["level"]
        result[f"hit_{i+1}_minutes"] = hit["hit_minutes_after_open"]
        result[f"hit_{i+1}_points_beyond"] = hit["points_beyond"]
    
    return result


def main():
    """Main analysis function."""
    print("Starting Liquidity Sequence Analysis...")
    
    tz_et = pytz.timezone("US/Eastern")
    all_results = []
    daily_closes = {}
    historical_data_cache = {}
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
            
            # Store daily close
            rth_mask = _within(day_data["ts_et"], RTH_START, RTH_END)
            rth = day_data.loc[rth_mask]
            if not rth.empty:
                daily_closes[d] = rth.iloc[-1]["close"]
            
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
            
            # Get last 5 days for historical data
            last_5_days = []
            for i in range(1, 6):
                check_date = d - timedelta(days=i)
                if check_date in prev_day_cache:
                    last_5_days.append(prev_day_cache[check_date])
                elif check_date in dates:
                    check_mask = to_proc["et_date"] == check_date
                    if check_mask.any():
                        last_5_days.append(to_proc[check_mask].copy())
            
            historical_data = {}
            if last_5_days:
                historical_data["last_5_days"] = pd.concat(last_5_days, ignore_index=True)
            else:
                historical_data["last_5_days"] = pd.DataFrame()
            historical_data["daily_closes"] = daily_closes.copy()
            
            result = process_day(day_data, d, historical_data, prev_day_data)
            if result:
                all_results.append(result)
            
            # Cache current day for next iteration
            prev_day_cache[d] = day_data.copy()
        
        # Determine front month
        chunk = _determine_front_month(chunk)
        
        # Process each day
        for date, g in chunk.groupby("et_date"):
            if date in daily_closes:
                continue
            
            # Get daily close for historical data
            rth_mask = _within(g["ts_et"], RTH_START, RTH_END)
            rth_data = g[rth_mask]
            if not rth_data.empty:
                daily_closes[date] = rth_data["close"].iloc[-1]
            
            # Build historical data cache
            if date not in historical_data_cache:
                # Get last 5 days of data
                date_obj = pd.Timestamp(date)
                last_5_days_start = date_obj - timedelta(days=7)
                
                # This is simplified - in production, you'd cache this properly
                historical_data_cache[date] = {
                    "last_5_days": None,  # Would need to build this from full dataset
                    "daily_closes": daily_closes.copy(),
                }
            
            result = process_day(g, date, historical_data_cache.get(date))
            if result:
                all_results.append(result)
    
    if not all_results:
        print("No results found!")
        return
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save detailed results
    detailed_file = OUT_DIR / "liquidity_sequence_detailed.csv"
    df_results.to_csv(detailed_file, index=False)
    print(f"Saved detailed results to {detailed_file}")
    
    # Analyze sequences
    print("\nAnalyzing sequences...")
    
    # Sequence frequency
    sequence_counts = Counter(df_results["sequence"])
    sequence_df = pd.DataFrame([
        {"sequence": seq, "count": count, "frequency_pct": count / len(df_results) * 100}
        for seq, count in sequence_counts.most_common(50)
    ])
    
    sequence_file = OUT_DIR / "liquidity_sequence_frequency.csv"
    sequence_df.to_csv(sequence_file, index=False)
    print(f"Saved sequence frequency to {sequence_file}")
    
    # Transition probabilities
    transitions = defaultdict(lambda: defaultdict(int))
    
    for _, row in df_results.iterrows():
        seq_list = row["sequence_list"]
        for i in range(len(seq_list) - 1):
            from_level = seq_list[i]
            to_level = seq_list[i + 1]
            transitions[from_level][to_level] += 1
    
    # Build transition matrix
    all_levels = set()
    for from_level, to_levels in transitions.items():
        all_levels.add(from_level)
        all_levels.update(to_levels.keys())
    
    all_levels = sorted(all_levels)
    transition_matrix = []
    
    for from_level in all_levels:
        row = {"from_level": from_level}
        total_from = sum(transitions[from_level].values())
        
        for to_level in all_levels:
            count = transitions[from_level].get(to_level, 0)
            prob = count / total_from * 100 if total_from > 0 else 0
            row[to_level] = prob
            row[f"{to_level}_count"] = count
        
        row["total_count"] = total_from
        transition_matrix.append(row)
    
    transition_df = pd.DataFrame(transition_matrix)
    transition_file = OUT_DIR / "liquidity_transition_matrix.csv"
    transition_df.to_csv(transition_file, index=False)
    print(f"Saved transition matrix to {transition_file}")
    
    # Statistics
    print(f"\nTotal days analyzed: {len(df_results)}")
    print(f"Unique sequences: {len(sequence_counts)}")
    print(f"\nTop 10 most common sequences:")
    for seq, count in sequence_counts.most_common(10):
        print(f"  {seq}: {count} times ({count/len(df_results)*100:.1f}%)")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()


