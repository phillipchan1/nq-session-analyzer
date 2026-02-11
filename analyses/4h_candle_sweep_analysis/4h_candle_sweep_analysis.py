#!/usr/bin/env python3
"""
4-Hour Candle Sweep Analysis
-----------------------------

Analyzes the probability that 4-hour candle highs/lows get swept in the first 
45 minutes (9:30-10:15 ET) when they haven't been swept before the NY open.

Focus: 6am candle (6am-10am ET) with expansion to 2am and 10pm candles.
Tracks: Both sides swept, only high swept, only low swept, neither swept.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import time, timedelta
import pytz
from collections import defaultdict

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20250926.ohlcv-1m.csv")
DETAILED_OUT_FILE = str(OUT_DIR / "4h_candle_sweep_detailed.csv")
SUMMARY_OUT_FILE = str(OUT_DIR / "4h_candle_sweep_summary.csv")
DISTANCE_BUCKET_OUT_FILE = str(OUT_DIR / "4h_candle_sweep_distance_buckets.csv")

# Post-formation analysis outputs
POST_FORMATION_WINDOWS_OUT_FILE = str(OUT_DIR / "4h_candle_post_formation_windows.csv")
POST_FORMATION_TIMING_OUT_FILE = str(OUT_DIR / "4h_candle_post_formation_timing.csv")
POST_FORMATION_DISTANCE_OUT_FILE = str(OUT_DIR / "4h_candle_post_formation_distance.csv")

NY_OPEN = time(9, 30)
FIRST_45_END = time(10, 15)
RTH_END = time(16, 0)  # Regular trading hours end

# Post-formation analysis config
POST_FORMATION_WINDOW_MINUTES = 15
ANALYZE_ALL_4H_CANDLES = False  # Set to True to analyze all 6 candles

# 4-hour candle start times (ET)
FOUR_HOUR_STARTS = [
    time(2, 0),   # 2am-6am
    time(6, 0),   # 6am-10am (PRIMARY FOCUS)
    time(10, 0),  # 10am-2pm
    time(14, 0),  # 2pm-6pm
    time(18, 0),  # 6pm-10pm
    time(22, 0),  # 10pm-2am (previous day)
]

NQ_TICK = 0.25  # NQ tick size - strict takeout requires 1+ tick
WEEKDAYS_ONLY = True
CHUNKSIZE = 1_000_000
NQ_SINGLE_CONTRACT_RE = re.compile(r"^NQ[A-Z]\d{1,2}$")
# ==============================================


def _within(ts: pd.Series, start_t: time, end_t: time) -> pd.Series:
    """Check if timestamps fall within time range."""
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)


def _determine_front_month(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to front-month per date by max volume in first 15 min."""
    mask = _within(df["ts_et"], NY_OPEN, time(9, 45))
    rth = df.loc[mask]
    if rth.empty:
        return df
    agg = rth.groupby(["et_date", "symbol"]).agg(volume=("volume", "sum")).reset_index()
    idx = agg.groupby("et_date")["volume"].idxmax()
    fm = agg.loc[idx][["et_date", "symbol"]].rename(columns={"symbol": "front_symbol"})
    out = df.merge(fm, on="et_date", how="left")
    out = out.loc[out["symbol"] == out["front_symbol"]].drop(columns=["front_symbol"])
    return out


def create_4h_candles(df: pd.DataFrame, candle_start_time: time) -> pd.DataFrame:
    """
    Create 4-hour candles aligned to a specific start time.
    
    Args:
        df: 1-minute OHLCV data
        candle_start_time: Start time for 4h candles (e.g., time(6, 0) for 6am)
    
    Returns:
        DataFrame with 4h candles
    """
    df = df.sort_values("ts_et").reset_index(drop=True).copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Get timezone from data
    tz = df["ts_et"].iloc[0].tz
    start_hour = candle_start_time.hour
    
    # Create a function to assign each timestamp to its 4h period
    def get_4h_period_start(ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Get the 4h period start timestamp for a given timestamp."""
        ts_date = ts.date()
        ts_hour = ts.hour
        
        # Handle wrap-around for 10pm candle (10pm-2am spans two days)
        if start_hour == 22:  # 10pm candle
            if ts_hour >= 22:
                # Same day, 10pm or later
                period_start = pd.Timestamp.combine(ts_date, candle_start_time)
            elif ts_hour < 2:
                # Next day, before 2am (still part of previous day's 10pm candle)
                period_start = pd.Timestamp.combine(ts_date - timedelta(days=1), candle_start_time)
            else:
                return None  # Not in this 4h period
        else:
            # Regular 4h periods (2am, 6am, 10am, 2pm, 6pm)
            if start_hour <= ts_hour < start_hour + 4:
                period_start = pd.Timestamp.combine(ts_date, candle_start_time)
            else:
                return None  # Not in this 4h period
        
        # Localize to same timezone as input
        if period_start.tz is None:
            if tz:
                period_start = period_start.tz_localize(tz)
            else:
                period_start = period_start.tz_localize("America/New_York")
        
        return period_start
    
    # Apply grouping
    df["4h_period_start"] = df["ts_et"].apply(get_4h_period_start)
    df = df[df["4h_period_start"].notna()].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Aggregate to 4h candles
    agg = df.groupby("4h_period_start").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).reset_index()
    
    agg.rename(columns={"4h_period_start": "candle_start"}, inplace=True)
    agg["candle_end"] = agg["candle_start"] + pd.Timedelta(hours=4)
    
    return agg


def find_relevant_4h_candles(df: pd.DataFrame, ny_open_ts: pd.Timestamp) -> List[Dict]:
    """
    Find relevant 4h candles for analysis at 9:30am.
    
    Returns candles that:
    - Started before 9:30am (or are currently active at 9:30am)
    - Are from the key times: 6am (primary), 2am, 10pm
    
    For 6am candle (active): Calculates partial high/low from 6:00 to 9:29 ET
    For completed candles (10pm, 2am): Uses full candle high/low
    """
    candles = []
    tz = df["ts_et"].iloc[0].tz
    current_date = ny_open_ts.date()
    
    # Freeze time for 6am candle partial levels: 9:29 ET
    freeze_time = time(9, 29)
    freeze_ts = pd.Timestamp.combine(current_date, freeze_time).tz_localize(tz)
    
    # Focus on these candle times (in order of priority)
    focus_times = [
        (time(6, 0), "active"),    # 6am-10am (PRIMARY) - still forming at 9:30
        (time(2, 0), "completed"),  # 2am-6am - completed before 9:30
        (time(22, 0), "completed"), # 10pm-2am - completed before 9:30
    ]
    
    for candle_start_time, candle_type in focus_times:
        # Create 4h candles for this start time
        df_4h = create_4h_candles(df, candle_start_time)
        
        if df_4h.empty:
            continue
        
        # Find the most recent candle that started before or at 9:30am
        relevant_candles = df_4h[df_4h["candle_start"] <= ny_open_ts].copy()
        
        if relevant_candles.empty:
            continue
        
        # Get the most recent candle (closest to 9:30am)
        relevant_candles = relevant_candles.sort_values("candle_start", ascending=False)
        row = relevant_candles.iloc[0]
        
        candle_start = row["candle_start"]
        candle_end = row["candle_end"]
        
        if candle_type == "active":
            # 6am candle: Calculate partial high/low from 6:00 to 9:29 ET only
            # Freeze levels at 9:29 to avoid tautology
            partial_data = df[
                (df["ts_et"] >= candle_start) & 
                (df["ts_et"] < freeze_ts)
            ].copy()
            
            if partial_data.empty:
                continue
            
            candle_high = partial_data["high"].max()
            candle_low = partial_data["low"].min()
            level_freeze_time = freeze_ts
        else:
            # Completed candles: Use full candle high/low
            candle_high = row["high"]
            candle_low = row["low"]
            level_freeze_time = candle_end  # Levels frozen at candle end
        
        candles.append({
            "candle_time": f"{candle_start_time.hour:02d}:{candle_start_time.minute:02d}",
            "candle_type": candle_type,
            "candle_start": candle_start,
            "candle_end": candle_end,
            "candle_high": candle_high,
            "candle_low": candle_low,
            "level_freeze_time": level_freeze_time,
            "candle_open": row["open"],
            "candle_close": row["close"],
            "candle_volume": row["volume"],
        })
    
    return candles


def check_sweep(df: pd.DataFrame, level: float, is_high: bool, 
                start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Tuple[bool, Optional[pd.Timestamp], float]:
    """
    Check if a level was swept (strict takeout) in a given time period.
    
    Strict takeout definition:
    - High sweep: price_high > level + TICK (trades above by at least 1 tick)
    - Low sweep: price_low < level - TICK (trades below by at least 1 tick)
    
    Returns:
        (was_swept, sweep_time, points_beyond)
    """
    period_data = df[(df["ts_et"] >= start_ts) & (df["ts_et"] < end_ts)].copy()
    
    if period_data.empty:
        return False, None, 0.0
    
    if is_high:
        # High swept: price high > level + TICK (strict takeout)
        swept_mask = period_data["high"] > level + NQ_TICK
        if swept_mask.any():
            first_sweep_idx = period_data[swept_mask].index[0]
            sweep_time = period_data.loc[first_sweep_idx, "ts_et"]
            points_beyond = period_data.loc[first_sweep_idx, "high"] - level
            return True, sweep_time, points_beyond
    else:
        # Low swept: price low < level - TICK (strict takeout)
        swept_mask = period_data["low"] < level - NQ_TICK
        if swept_mask.any():
            first_sweep_idx = period_data[swept_mask].index[0]
            sweep_time = period_data.loc[first_sweep_idx, "ts_et"]
            points_beyond = level - period_data.loc[first_sweep_idx, "low"]
            return True, sweep_time, points_beyond
    
    return False, None, 0.0


def get_distance_bucket(distance: float) -> str:
    """Get distance bucket for a given distance."""
    if distance < 10:
        return "0-10"
    elif distance < 20:
        return "10-20"
    elif distance < 40:
        return "20-40"
    elif distance < 80:
        return "40-80"
    else:
        return "80+"


def get_time_bucket(minutes: Optional[float]) -> Optional[str]:
    """Get time bucket for minutes after candle_end."""
    if minutes is None:
        return None
    if minutes < 15:
        return "0-15"
    elif minutes < 30:
        return "15-30"
    elif minutes < 60:
        return "30-60"
    elif minutes < 120:
        return "60-120"
    elif minutes < 240:
        return "120-240"
    else:
        return "240+"


def identify_all_4h_candles(df: pd.DataFrame, current_date, analyze_all: bool = False) -> List[Dict]:
    """
    Identify all completed 4H candles for the day.
    
    Args:
        df: 1-minute OHLCV data (may include previous day for 22:00 candle)
        current_date: Current trading date
        analyze_all: If True, analyze all 6 candles; if False, only 22-02, 02-06, 06-10
    
    Returns:
        List of candle dictionaries with candle info
    """
    candles = []
    tz = df["ts_et"].iloc[0].tz
    
    # Required candles
    required_times = [
        (time(22, 0), "22-02"),  # 10pm-2am
        (time(2, 0), "02-06"),   # 2am-6am
        (time(6, 0), "06-10"),   # 6am-10am
    ]
    
    # Optional candles
    optional_times = [
        (time(10, 0), "10-14"),  # 10am-2pm
        (time(14, 0), "14-18"),  # 2pm-6pm
        (time(18, 0), "18-22"),  # 6pm-10pm
    ]
    
    candle_times = required_times + (optional_times if analyze_all else [])
    
    for candle_start_time, candle_type in candle_times:
        # Create 4h candles for this start time (includes data from previous day if needed)
        df_4h = create_4h_candles(df, candle_start_time)
        
        if df_4h.empty:
            continue
        
        # Find candles that END on current_date
        # For 22:00 candle, it ends at 02:00 on current_date (started previous day at 22:00)
        if candle_start_time.hour == 22:
            # 22:00 candle ends at 02:00 on current_date
            candle_end_ts = pd.Timestamp.combine(current_date, time(2, 0)).tz_localize(tz)
        else:
            # Regular candles end same day
            candle_end_time = time(candle_start_time.hour + 4, 0)
            candle_end_ts = pd.Timestamp.combine(current_date, candle_end_time).tz_localize(tz)
        
        # Find candle that ended at this time
        relevant = df_4h[df_4h["candle_end"] == candle_end_ts]
        
        if relevant.empty:
            continue
        
        row = relevant.iloc[0]
        
        # Verify candle is complete (has data up to candle_end)
        candle_data_check = df[
            (df["ts_et"] >= row["candle_start"]) & 
            (df["ts_et"] < row["candle_end"])
        ]
        
        if candle_data_check.empty:
            continue
        
        candles.append({
            "candle_type": candle_type,
            "candle_start": row["candle_start"],
            "candle_end": row["candle_end"],
            "candle_high": row["high"],
            "candle_low": row["low"],
            "candle_open": row["open"],
            "candle_close": row["close"],
            "candle_volume": row["volume"],
        })
    
    return candles


def create_post_formation_windows(candle_end: pd.Timestamp, horizon_end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Create 15-minute windows from candle_end to horizon_end.
    
    Args:
        candle_end: Timestamp when candle completed
        horizon_end: End timestamp (16:00 ET same day or next day)
    
    Returns:
        List of (window_start, window_end) tuples
    """
    windows = []
    current_start = candle_end
    
    while current_start < horizon_end:
        window_end = current_start + pd.Timedelta(minutes=POST_FORMATION_WINDOW_MINUTES)
        if window_end > horizon_end:
            window_end = horizon_end
        
        windows.append((current_start, window_end))
        current_start = window_end
        
        # Safety check to avoid infinite loop
        if len(windows) > 1000:
            break
    
    return windows


def get_horizon_end(candle_end: pd.Timestamp, candle_type: str) -> pd.Timestamp:
    """
    Get horizon end timestamp for a candle.
    
    For overnight candles (22-02): horizon extends to 16:00 ET next day
    For other candles: horizon extends to 16:00 ET same day
    """
    tz = candle_end.tz
    candle_date = candle_end.date()
    
    if candle_type == "22-02":
        # Overnight candle: horizon extends to 16:00 ET next day
        horizon_date = candle_date + timedelta(days=1)
    else:
        # Same day horizon
        horizon_date = candle_date
    
    horizon_end = pd.Timestamp.combine(horizon_date, RTH_END).tz_localize(tz)
    return horizon_end


def calculate_distance_bucket_stats(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sweep probabilities by distance bucket.
    
    Returns DataFrame with columns:
    - candle_time, distance_bucket
    - high_count, high_swept_count, high_swept_pct
    - low_count, low_swept_count, low_swept_pct
    """
    bucket_stats = []
    
    for candle_time in sorted(df_results["candle_time"].unique()):
        candle_data = df_results[df_results["candle_time"] == candle_time]
        
        # Add distance buckets
        candle_data = candle_data.copy()
        candle_data["high_distance_bucket"] = candle_data["distance_high_from_open"].apply(get_distance_bucket)
        candle_data["low_distance_bucket"] = candle_data["distance_low_from_open"].apply(get_distance_bucket)
        
        # Process high buckets
        for bucket in ["0-10", "10-20", "20-40", "40-80", "80+"]:
            bucket_high_data = candle_data[candle_data["high_distance_bucket"] == bucket]
            
            if len(bucket_high_data) == 0:
                continue
            
            high_not_swept_preopen = bucket_high_data[~bucket_high_data["high_swept_pre_open"]]
            high_count = len(high_not_swept_preopen)
            
            if high_count > 0:
                high_swept_count = high_not_swept_preopen["high_swept_first_45min"].sum()
                high_swept_pct = (high_swept_count / high_count) * 100
            else:
                high_swept_count = 0
                high_swept_pct = 0.0
            
            bucket_stats.append({
                "candle_time": candle_time,
                "distance_bucket": bucket,
                "side": "high",
                "count": high_count,
                "swept_count": high_swept_count,
                "swept_pct": round(high_swept_pct, 2),
            })
        
        # Process low buckets
        for bucket in ["0-10", "10-20", "20-40", "40-80", "80+"]:
            bucket_low_data = candle_data[candle_data["low_distance_bucket"] == bucket]
            
            if len(bucket_low_data) == 0:
                continue
            
            low_not_swept_preopen = bucket_low_data[~bucket_low_data["low_swept_pre_open"]]
            low_count = len(low_not_swept_preopen)
            
            if low_count > 0:
                low_swept_count = low_not_swept_preopen["low_swept_first_45min"].sum()
                low_swept_pct = (low_swept_count / low_count) * 100
            else:
                low_swept_count = 0
                low_swept_pct = 0.0
            
            bucket_stats.append({
                "candle_time": candle_time,
                "distance_bucket": bucket,
                "side": "low",
                "count": low_count,
                "swept_count": low_swept_count,
                "swept_pct": round(low_swept_pct, 2),
            })
    
    return pd.DataFrame(bucket_stats)


def analyze_post_formation_sweeps(df: pd.DataFrame, candles: List[Dict], current_date) -> Tuple[List[Dict], List[Dict]]:
    """
    Analyze post-formation sweeps for all candles.
    
    Tracks eligibility state through 15-minute windows and records sweeps.
    
    Returns:
        (window_results, timing_results)
    """
    window_results = []
    timing_results = []
    
    for candle in candles:
        candle_type = candle["candle_type"]
        candle_end = candle["candle_end"]
        candle_high = candle["candle_high"]
        candle_low = candle["candle_low"]
        
        # Get horizon end
        horizon_end = get_horizon_end(candle_end, candle_type)
        
        # Get price at candle_end (first bar after candle_end)
        price_at_candle_end = None
        after_candle_end = df[df["ts_et"] > candle_end].copy()
        if not after_candle_end.empty:
            price_at_candle_end = after_candle_end.iloc[0]["close"]
        
        # Calculate distances
        dist_to_high = candle_high - price_at_candle_end if price_at_candle_end else None
        dist_to_low = price_at_candle_end - candle_low if price_at_candle_end else None
        
        # Initialize eligibility state
        high_eligible = True
        low_eligible = True
        high_swept_at = None
        low_swept_at = None
        
        # Create windows
        windows = create_post_formation_windows(candle_end, horizon_end)
        
        # Process each window
        for window_idx, (window_start, window_end) in enumerate(windows, start=1):
            # Count eligible at window start
            high_eligible_at_start = high_eligible
            low_eligible_at_start = low_eligible
            either_eligible_at_start = high_eligible and low_eligible
            
            # Check sweeps in this window
            high_swept_in_window = False
            low_swept_in_window = False
            
            if high_eligible:
                high_swept, sweep_time, _ = check_sweep(
                    df, candle_high, True, window_start, window_end
                )
                if high_swept:
                    high_swept_in_window = True
                    high_swept_at = sweep_time
                    high_eligible = False
            
            if low_eligible:
                low_swept, sweep_time, _ = check_sweep(
                    df, candle_low, False, window_start, window_end
                )
                if low_swept:
                    low_swept_in_window = True
                    low_swept_at = sweep_time
                    low_eligible = False
            
            # Record window results
            window_result = {
                "date": current_date,
                "candle_type": candle_type,
                "candle_end_time": candle_end.time(),
                "candle_end": candle_end,
                "window_index": window_idx,
                "window_start": window_start,
                "window_end": window_end,
                "high_eligible_at_start": high_eligible_at_start,
                "high_swept_in_window": high_swept_in_window,
                "low_eligible_at_start": low_eligible_at_start,
                "low_swept_in_window": low_swept_in_window,
                "either_eligible_at_start": either_eligible_at_start,
                "either_swept_in_window": high_swept_in_window or low_swept_in_window,
            }
            window_results.append(window_result)
        
        # Record timing results (final state after all windows)
        minutes_after_candle_end_high = None
        minutes_after_candle_end_low = None
        if high_swept_at:
            minutes_after_candle_end_high = (high_swept_at - candle_end).total_seconds() / 60.0
        if low_swept_at:
            minutes_after_candle_end_low = (low_swept_at - candle_end).total_seconds() / 60.0
        
        timing_result_high = {
            "date": current_date,
            "candle_type": candle_type,
            "side": "high",
            "minutes_after_candle_end_to_sweep": minutes_after_candle_end_high,
            "time_bucket": get_time_bucket(minutes_after_candle_end_high),
            "dist_to_level": dist_to_high,
            "distance_bucket": get_distance_bucket(dist_to_high) if dist_to_high is not None else None,
        }
        timing_result_low = {
            "date": current_date,
            "candle_type": candle_type,
            "side": "low",
            "minutes_after_candle_end_to_sweep": minutes_after_candle_end_low,
            "time_bucket": get_time_bucket(minutes_after_candle_end_low),
            "dist_to_level": dist_to_low,
            "distance_bucket": get_distance_bucket(dist_to_low) if dist_to_low is not None else None,
        }
        timing_results.append(timing_result_high)
        timing_results.append(timing_result_low)
    
    return window_results, timing_results


def process_day_post_formation(g: pd.DataFrame, prev_day_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
    """
    Process post-formation sweeps for a single day.
    
    Args:
        g: Current day's data
        prev_day_data: Previous day's data (needed for 22:00 candle that spans days)
    
    Returns:
        (window_results, timing_results)
    """
    g = g.sort_values("ts_et").reset_index(drop=True)
    current_date = g["ts_et"].dt.date.iloc[0]
    
    # Combine with previous day data if available (needed for 22:00 candle)
    if prev_day_data is not None:
        df_for_candles = pd.concat([prev_day_data, g]).sort_values("ts_et").reset_index(drop=True)
    else:
        df_for_candles = g.copy()
    
    # Identify all completed 4H candles
    candles = identify_all_4h_candles(df_for_candles, current_date, ANALYZE_ALL_4H_CANDLES)
    
    if not candles:
        return None, None
    
    # For post-formation analysis, we need data from candle_end onwards
    # Use current day's data (g) for sweeps after candle_end
    # For 22:00 candle ending at 02:00, we need data from 02:00 onwards on current_date
    # For other candles, we need data from candle_end onwards
    
    # Analyze post-formation sweeps
    # Use g (current day) for sweep detection, but candles may reference previous day
    window_results, timing_results = analyze_post_formation_sweeps(g, candles, current_date)
    
    return window_results, timing_results


def process_day(g: pd.DataFrame) -> Optional[List[Dict]]:
    """Process a single trading day."""
    g = g.sort_values("ts_et").reset_index(drop=True)
    current_date = g["ts_et"].dt.date.iloc[0]
    
    # Get 9:30am opening price
    open_930 = g.loc[_within(g["ts_et"], NY_OPEN, time(9, 31))]
    if open_930.empty:
        return None
    
    open_price = open_930.iloc[0]["open"]
    ny_open_ts = open_930.iloc[0]["ts_et"]
    
    # Get timezone from data
    tz = g["ts_et"].iloc[0].tz
    
    # Find relevant 4h candles
    candles = find_relevant_4h_candles(g, ny_open_ts)
    
    if not candles:
        return None
    
    results = []
    
    for candle in candles:
        candle_high = candle["candle_high"]
        candle_low = candle["candle_low"]
        candle_start = candle["candle_start"]
        candle_end = candle["candle_end"]
        candle_type = candle["candle_type"]
        level_freeze_time = candle["level_freeze_time"]
        
        # Calculate distances from open
        distance_high = abs(candle_high - open_price)
        distance_low = abs(candle_low - open_price)
        
        # Check pre-open sweeps - logic differs by candle type
        if candle_type == "active":
            # 6am candle: Check from 6:00 to 9:29 (levels frozen at 9:29)
            pre_open_start = candle_start
            pre_open_end = level_freeze_time  # 9:29 ET
        else:
            # Completed candles: Check from candle_end to 9:30
            # Only check if level was taken out AFTER candle completed
            pre_open_start = candle_end
            pre_open_end = ny_open_ts
        
        high_swept_pre_open, _, _ = check_sweep(
            g, candle_high, True, pre_open_start, pre_open_end
        )
        low_swept_pre_open, _, _ = check_sweep(
            g, candle_low, False, pre_open_start, pre_open_end
        )
        
        # Check first 45-minute sweeps (9:30am-10:15am)
        first_45_start = ny_open_ts
        first_45_end_ts = pd.Timestamp.combine(current_date, FIRST_45_END).tz_localize(tz)
        
        high_swept_first_45min = False
        low_swept_first_45min = False
        time_to_high_sweep = None
        time_to_low_sweep = None
        points_beyond_high = 0.0
        points_beyond_low = 0.0
        
        # Only check if not already swept pre-open
        if not high_swept_pre_open:
            high_swept_first_45min, high_sweep_time_45, points_beyond_high = check_sweep(
                g, candle_high, True, first_45_start, first_45_end_ts
            )
            if high_swept_first_45min:
                time_to_high_sweep = (high_sweep_time_45 - ny_open_ts).total_seconds() / 60.0
        
        if not low_swept_pre_open:
            low_swept_first_45min, low_sweep_time_45, points_beyond_low = check_sweep(
                g, candle_low, False, first_45_start, first_45_end_ts
            )
            if low_swept_first_45min:
                time_to_low_sweep = (low_sweep_time_45 - ny_open_ts).total_seconds() / 60.0
        
        # Determine sweep category
        if high_swept_first_45min and low_swept_first_45min:
            sweep_category = "both_swept"
        elif high_swept_first_45min:
            sweep_category = "high_only_swept"
        elif low_swept_first_45min:
            sweep_category = "low_only_swept"
        else:
            sweep_category = "neither_swept"
        
        result = {
            "date": current_date,
            "candle_time": candle["candle_time"],
            "candle_type": candle_type,
            "candle_start_time": candle_start,
            "candle_end_time": candle_end,
            "level_freeze_time": level_freeze_time,
            "candle_high": candle_high,
            "candle_low": candle_low,
            "open_price_930": open_price,
            "distance_high_from_open": distance_high,
            "distance_low_from_open": distance_low,
            "high_swept_pre_open": high_swept_pre_open,
            "low_swept_pre_open": low_swept_pre_open,
            "high_swept_first_45min": high_swept_first_45min,
            "low_swept_first_45min": low_swept_first_45min,
            "sweep_category": sweep_category,
            "time_to_high_sweep": time_to_high_sweep,
            "time_to_low_sweep": time_to_low_sweep,
            "points_beyond_high": points_beyond_high,
            "points_beyond_low": points_beyond_low,
        }
        
        results.append(result)
    
    return results


def calculate_post_formation_summary(df_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate window summary statistics aggregated across all days.
    
    Returns DataFrame with columns:
    - candle_type, window_index, window_start, window_end
    - high_eligible_count, high_swept_count, high_swept_pct
    - low_eligible_count, low_swept_count, low_swept_pct
    - either_eligible_count, either_swept_count, either_swept_pct
    """
    summary_stats = []
    
    for candle_type in sorted(df_windows["candle_type"].unique()):
        candle_data = df_windows[df_windows["candle_type"] == candle_type]
        
        for window_idx in sorted(candle_data["window_index"].unique()):
            window_data = candle_data[candle_data["window_index"] == window_idx]
            
            # Aggregate across all days
            high_eligible_count = window_data["high_eligible_at_start"].sum()
            high_swept_count = window_data["high_swept_in_window"].sum()
            high_swept_pct = (high_swept_count / high_eligible_count * 100) if high_eligible_count > 0 else 0.0
            
            low_eligible_count = window_data["low_eligible_at_start"].sum()
            low_swept_count = window_data["low_swept_in_window"].sum()
            low_swept_pct = (low_swept_count / low_eligible_count * 100) if low_eligible_count > 0 else 0.0
            
            either_eligible_count = window_data["either_eligible_at_start"].sum()
            either_swept_count = window_data["either_swept_in_window"].sum()
            either_swept_pct = (either_swept_count / either_eligible_count * 100) if either_eligible_count > 0 else 0.0
            
            # Get window times (should be same for all rows in this window)
            window_start = window_data["window_start"].iloc[0]
            window_end = window_data["window_end"].iloc[0]
            candle_end_time = window_data["candle_end_time"].iloc[0]
            
            summary_stats.append({
                "candle_type": candle_type,
                "candle_end_time": candle_end_time,
                "window_index": window_idx,
                "window_start": window_start,
                "window_end": window_end,
                "high_eligible_count": high_eligible_count,
                "high_swept_count": high_swept_count,
                "high_swept_pct": round(high_swept_pct, 2),
                "low_eligible_count": low_eligible_count,
                "low_swept_count": low_swept_count,
                "low_swept_pct": round(low_swept_pct, 2),
                "either_eligible_count": either_eligible_count,
                "either_swept_count": either_swept_count,
                "either_swept_pct": round(either_swept_pct, 2),
            })
    
    return pd.DataFrame(summary_stats)


def calculate_distance_buckets_post_formation(df_timing: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distance bucket statistics for post-formation sweeps.
    
    Returns DataFrame with columns:
    - candle_type, distance_bucket, side, count, swept_count, swept_pct
    """
    bucket_stats = []
    
    for candle_type in sorted(df_timing["candle_type"].unique()):
        candle_data = df_timing[df_timing["candle_type"] == candle_type]
        
        for side in ["high", "low"]:
            side_data = candle_data[candle_data["side"] == side]
            
            # Filter to candles with valid distance
            side_data = side_data[side_data["distance_bucket"].notna()].copy()
            
            if side_data.empty:
                continue
            
            for bucket in ["0-10", "10-20", "20-40", "40-80", "80+"]:
                bucket_data = side_data[side_data["distance_bucket"] == bucket]
                
                if len(bucket_data) == 0:
                    continue
                
                count = len(bucket_data)
                swept_count = bucket_data["minutes_after_candle_end_to_sweep"].notna().sum()
                swept_pct = (swept_count / count * 100) if count > 0 else 0.0
                
                bucket_stats.append({
                    "candle_type": candle_type,
                    "distance_bucket": bucket,
                    "side": side,
                    "count": count,
                    "swept_count": swept_count,
                    "swept_pct": round(swept_pct, 2),
                })
    
    return pd.DataFrame(bucket_stats)


def main():
    tz_et = pytz.timezone("America/New_York")
    all_results = []
    all_post_formation_windows = []
    all_post_formation_timing = []
    
    residual = pd.DataFrame()
    
    print("Processing data for 4h candle sweep analysis...")
    print("=" * 80)
    
    for chunk_num, chunk in enumerate(pd.read_csv(
        CSV_FILE,
        usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
            "symbol": "string",
        },
        chunksize=CHUNKSIZE,
    )):
        print(f"Processing chunk {chunk_num + 1}...")
        
        df = pd.concat([residual, chunk])
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz_et)
        df["et_date"] = df["ts_et"].dt.date
        df = df[df["symbol"].astype(str).str.match(NQ_SINGLE_CONTRACT_RE)]
        
        if WEEKDAYS_ONLY:
            df = df[df["ts_et"].dt.weekday <= 4]
        
        if df.empty:
            continue
        
        # Determine front month
        df = _determine_front_month(df)
        
        max_date = df["et_date"].max()
        to_proc = df[df["et_date"] < max_date]
        residual = df[df["et_date"] == max_date]
        
        # Track previous day data for 22:00 candle
        prev_day_data_by_symbol = {}
        
        for (sym, d), g in to_proc.groupby(["symbol", "et_date"]):
            # Process pre-open analysis
            results = process_day(g)
            if results is not None:
                for result in results:
                    result["symbol"] = sym
                    all_results.append(result)
            
            # Process post-formation analysis
            # Get previous day data for this symbol if available
            prev_day_data = prev_day_data_by_symbol.get(sym)
            window_results, timing_results = process_day_post_formation(g, prev_day_data)
            if window_results is not None:
                for result in window_results:
                    result["symbol"] = sym
                    all_post_formation_windows.append(result)
            if timing_results is not None:
                for result in timing_results:
                    result["symbol"] = sym
                    all_post_formation_timing.append(result)
            
            # Store current day data for next iteration
            prev_day_data_by_symbol[sym] = g.copy()
    
    # Process final residual
    if not residual.empty:
        for (sym, d), g in residual.groupby(["symbol", "et_date"]):
            # Process pre-open analysis
            results = process_day(g)
            if results is not None:
                for result in results:
                    result["symbol"] = sym
                    all_results.append(result)
            
            # Process post-formation analysis
            # Get previous day data for this symbol if available
            prev_day_data = prev_day_data_by_symbol.get(sym)
            window_results, timing_results = process_day_post_formation(g, prev_day_data)
            if window_results is not None:
                for result in window_results:
                    result["symbol"] = sym
                    all_post_formation_windows.append(result)
            if timing_results is not None:
                for result in timing_results:
                    result["symbol"] = sym
                    all_post_formation_timing.append(result)
    
    if not all_results:
        print("No data found!")
        return
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Calculate summary statistics - track high and low independently
    summary_stats = []
    
    for candle_time in sorted(df_results["candle_time"].unique()):
        candle_data = df_results[df_results["candle_time"] == candle_time]
        
        total_days = len(candle_data)
        
        # Count pre-open status (track independently)
        high_not_swept_preopen_count = (~candle_data["high_swept_pre_open"]).sum()
        low_not_swept_preopen_count = (~candle_data["low_swept_pre_open"]).sum()
        
        # Track high sweeps independently (only for high not swept pre-open)
        high_not_swept_preopen = candle_data[~candle_data["high_swept_pre_open"]]
        high_swept_9_30_to_10_15_count = high_not_swept_preopen["high_swept_first_45min"].sum()
        high_swept_9_30_to_10_15_pct = (
            (high_swept_9_30_to_10_15_count / len(high_not_swept_preopen) * 100)
            if len(high_not_swept_preopen) > 0 else 0.0
        )
        
        # Track low sweeps independently (only for low not swept pre-open)
        low_not_swept_preopen = candle_data[~candle_data["low_swept_pre_open"]]
        low_swept_9_30_to_10_15_count = low_not_swept_preopen["low_swept_first_45min"].sum()
        low_swept_9_30_to_10_15_pct = (
            (low_swept_9_30_to_10_15_count / len(low_not_swept_preopen) * 100)
            if len(low_not_swept_preopen) > 0 else 0.0
        )
        
        # Calculate either_swept and both_swept (for days where both sides not swept pre-open)
        both_not_swept_preopen = candle_data[
            (~candle_data["high_swept_pre_open"]) & 
            (~candle_data["low_swept_pre_open"])
        ]
        
        if len(both_not_swept_preopen) > 0:
            either_swept_count = (
                (both_not_swept_preopen["high_swept_first_45min"] | 
                 both_not_swept_preopen["low_swept_first_45min"]).sum()
            )
            both_swept_count = (
                (both_not_swept_preopen["high_swept_first_45min"] & 
                 both_not_swept_preopen["low_swept_first_45min"]).sum()
            )
            either_swept_pct = (either_swept_count / len(both_not_swept_preopen)) * 100
            both_swept_pct = (both_swept_count / len(both_not_swept_preopen)) * 100
        else:
            either_swept_count = both_swept_count = 0
            either_swept_pct = both_swept_pct = 0.0
        
        # Calculate averages
        avg_distance_high = candle_data["distance_high_from_open"].mean()
        avg_distance_low = candle_data["distance_low_from_open"].mean()
        
        # Average time to sweep (only for swept candles)
        high_swept = high_not_swept_preopen[high_not_swept_preopen["high_swept_first_45min"]]
        low_swept = low_not_swept_preopen[low_not_swept_preopen["low_swept_first_45min"]]
        
        avg_time_to_sweep_high = high_swept["time_to_high_sweep"].mean() if len(high_swept) > 0 else None
        avg_time_to_sweep_low = low_swept["time_to_low_sweep"].mean() if len(low_swept) > 0 else None
        
        stats = {
            "candle_time": candle_time,
            "total_days": total_days,
            "high_not_swept_preopen_count": high_not_swept_preopen_count,
            "low_not_swept_preopen_count": low_not_swept_preopen_count,
            "high_swept_9_30_to_10_15_count": high_swept_9_30_to_10_15_count,
            "high_swept_9_30_to_10_15_pct": round(high_swept_9_30_to_10_15_pct, 2),
            "low_swept_9_30_to_10_15_count": low_swept_9_30_to_10_15_count,
            "low_swept_9_30_to_10_15_pct": round(low_swept_9_30_to_10_15_pct, 2),
            "either_swept_count": either_swept_count,
            "either_swept_pct": round(either_swept_pct, 2),
            "both_swept_count": both_swept_count,
            "both_swept_pct": round(both_swept_pct, 2),
            "avg_distance_high": round(avg_distance_high, 2),
            "avg_distance_low": round(avg_distance_low, 2),
            "avg_time_to_sweep_high": round(avg_time_to_sweep_high, 2) if avg_time_to_sweep_high is not None else None,
            "avg_time_to_sweep_low": round(avg_time_to_sweep_low, 2) if avg_time_to_sweep_low is not None else None,
        }
        
        summary_stats.append(stats)
    
    # Save outputs
    df_results.to_csv(DETAILED_OUT_FILE, index=False)
    print(f"\n✅ Wrote detailed results to {DETAILED_OUT_FILE}")
    
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(SUMMARY_OUT_FILE, index=False)
    print(f"✅ Wrote summary statistics to {SUMMARY_OUT_FILE}")
    
    # Calculate distance bucket statistics
    df_distance_buckets = calculate_distance_bucket_stats(df_results)
    df_distance_buckets.to_csv(DISTANCE_BUCKET_OUT_FILE, index=False)
    print(f"✅ Wrote distance bucket statistics to {DISTANCE_BUCKET_OUT_FILE}")
    
    # Process post-formation analysis results
    if all_post_formation_windows:
        df_post_formation_windows = pd.DataFrame(all_post_formation_windows)
        df_post_formation_summary = calculate_post_formation_summary(df_post_formation_windows)
        df_post_formation_summary.to_csv(POST_FORMATION_WINDOWS_OUT_FILE, index=False)
        print(f"✅ Wrote post-formation window summary to {POST_FORMATION_WINDOWS_OUT_FILE}")
    else:
        print("⚠️  No post-formation window data found")
    
    if all_post_formation_timing:
        df_post_formation_timing = pd.DataFrame(all_post_formation_timing)
        df_post_formation_timing.to_csv(POST_FORMATION_TIMING_OUT_FILE, index=False)
        print(f"✅ Wrote post-formation timing distribution to {POST_FORMATION_TIMING_OUT_FILE}")
        
        # Calculate distance buckets for post-formation
        df_post_formation_distance = calculate_distance_buckets_post_formation(df_post_formation_timing)
        if not df_post_formation_distance.empty:
            df_post_formation_distance.to_csv(POST_FORMATION_DISTANCE_OUT_FILE, index=False)
            print(f"✅ Wrote post-formation distance buckets to {POST_FORMATION_DISTANCE_OUT_FILE}")
    else:
        print("⚠️  No post-formation timing data found")
    
    # Print key insights
    print("\n" + "=" * 80)
    print("4H CANDLE SWEEP ANALYSIS - Key Insights")
    print("=" * 80)
    
    for candle_time in ["06:00", "02:00", "22:00"]:  # 6am, 2am, 10pm
        candle_summary = df_summary[df_summary["candle_time"] == candle_time]
        if candle_summary.empty:
            continue
        
        row = candle_summary.iloc[0]
        print(f"\n{candle_time} CANDLE ({candle_time.replace(':', '')} ET):")
        print(f"  Total days analyzed: {row['total_days']}")
        print(f"  High not swept pre-open: {row['high_not_swept_preopen_count']}")
        print(f"  Low not swept pre-open: {row['low_not_swept_preopen_count']}")
        
        print(f"\n  First 45min Sweep Probabilities (conditional on NOT swept pre-open):")
        print(f"    High swept: {row['high_swept_9_30_to_10_15_pct']}% ({row['high_swept_9_30_to_10_15_count']}/{row['high_not_swept_preopen_count']})")
        print(f"    Low swept: {row['low_swept_9_30_to_10_15_pct']}% ({row['low_swept_9_30_to_10_15_count']}/{row['low_not_swept_preopen_count']})")
        print(f"    Either swept: {row['either_swept_pct']}% ({row['either_swept_count']} cases)")
        print(f"    Both swept: {row['both_swept_pct']}% ({row['both_swept_count']} cases)")
        
        if row["avg_time_to_sweep_high"] is not None:
            print(f"\n  Average time to sweep high: {row['avg_time_to_sweep_high']:.1f} minutes")
        if row["avg_time_to_sweep_low"] is not None:
            print(f"  Average time to sweep low: {row['avg_time_to_sweep_low']:.1f} minutes")
        
        print(f"\n  Average distance from 9:30am open:")
        print(f"    High: {row['avg_distance_high']:.1f} points")
        print(f"    Low: {row['avg_distance_low']:.1f} points")
    
    print("\n" + "=" * 80)
    print("POST-FORMATION SWEEP ANALYSIS")
    print("=" * 80)
    
    if all_post_formation_windows:
        df_post_formation_summary_display = calculate_post_formation_summary(pd.DataFrame(all_post_formation_windows))
        
        for candle_type in ["22-02", "02-06", "06-10"]:
            candle_data = df_post_formation_summary_display[df_post_formation_summary_display["candle_type"] == candle_type]
            if candle_data.empty:
                continue
            
            print(f"\n{candle_type} CANDLE:")
            # Show first 5 windows
            for _, row in candle_data.head(5).iterrows():
                window_start_time = row["window_start"].time()
                window_end_time = row["window_end"].time()
                print(f"  Window {row['window_index']} ({window_start_time}-{window_end_time}): "
                      f"High {row['high_swept_pct']:.1f}% ({row['high_swept_count']}/{row['high_eligible_count']}), "
                      f"Low {row['low_swept_pct']:.1f}% ({row['low_swept_count']}/{row['low_eligible_count']}), "
                      f"Either {row['either_swept_pct']:.1f}%")
            
            if len(candle_data) > 5:
                print(f"  ... ({len(candle_data) - 5} more windows)")
    
    print("\n" + "=" * 80)
    print(f"Total days analyzed: {len(df_results['date'].unique())}")
    print("\nPre-open analysis outputs:")
    print(f"  Detailed results: {DETAILED_OUT_FILE}")
    print(f"  Summary statistics: {SUMMARY_OUT_FILE}")
    print(f"  Distance bucket statistics: {DISTANCE_BUCKET_OUT_FILE}")
    print("\nPost-formation analysis outputs:")
    if all_post_formation_windows:
        print(f"  Window summary: {POST_FORMATION_WINDOWS_OUT_FILE}")
    if all_post_formation_timing:
        print(f"  Timing distribution: {POST_FORMATION_TIMING_OUT_FILE}")
        print(f"  Distance buckets: {POST_FORMATION_DISTANCE_OUT_FILE}")


if __name__ == "__main__":
    main()




