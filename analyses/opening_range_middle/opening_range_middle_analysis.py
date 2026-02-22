#!/usr/bin/env python3
"""
Opening Range Middle Analysis
------------------------------

Analyzes what happens when price sits "in the middle" of the opening range 
(first 15 minutes, 9:30-9:45 AM ET) at 9:45 AM ET.

Specifically tracks the probability that one side (high or low) gets hit 
before 10:15 AM when price is positioned in the middle of the range.

Key Questions:
- When price is in the middle at 9:45, what's the probability of hitting 
  the high or low before 10:15?
- Which side gets hit more often?
- What are the characteristics of days where price sits in the middle?
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from datetime import time, timedelta, date
import pytz
from tqdm import tqdm
from collections import defaultdict
import re

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20260221.ohlcv-1m.csv")

RTH_START, RTH_END = time(9, 30), time(16, 0)
NY_OPEN = time(9, 30)
OR_START = time(9, 30)  # Opening range start
OR_END = time(9, 45)    # Opening range end
CHECK_TIME = time(9, 45)  # Time to check if in middle
HIT_WINDOW_END = time(10, 15)  # End of hit tracking window

MIN_DISTANCE_POINTS = 20  # Minimum distance from each boundary
EPSILON = 0.1  # Tolerance for hit detection
RECENT_DAYS = 100  # Number of recent days to include in qualifying list
CHUNKSIZE = 1_000_000

NQ_SINGLE_CONTRACT_RE = re.compile(r"^NQ[A-Z]\d{1,2}$")

# =================== HELPER FUNCTIONS ===================

def _get_us_market_holidays(start_year: int = 2020, end_year: int = 2026) -> Set[date]:
    """
    Get set of US market holidays (NYSE/NASDAQ closures).
    Includes: New Year's Day, MLK Day, Presidents Day, Good Friday,
    Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas.
    """
    holidays = set()
    
    # Use pandas USFederalHolidayCalendar if available
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        holiday_dates = cal.holidays(start=pd.Timestamp(f'{start_year}-01-01'), 
                                     end=pd.Timestamp(f'{end_year}-12-31'))
        holidays.update([d.date() for d in holiday_dates])
    except (ImportError, AttributeError):
        # Fallback: manual calculation for market holidays
        for year in range(start_year, end_year + 1):
            # New Year's Day
            ny = date(year, 1, 1)
            if ny.weekday() == 5:  # Saturday
                holidays.add(date(year - 1, 12, 31))
            elif ny.weekday() == 6:  # Sunday
                holidays.add(date(year, 1, 2))
            else:
                holidays.add(ny)
            
            # MLK Day (3rd Monday in January)
            jan_mondays = [d for d in [date(year, 1, d) for d in range(1, 32)] if d.weekday() == 0]
            if len(jan_mondays) >= 3:
                holidays.add(jan_mondays[2])  # 3rd Monday (0-indexed)
            
            # Presidents Day (3rd Monday in February)
            pres_mondays = [d for d in [date(year, 2, d) for d in range(1, 29)] if d.weekday() == 0]
            holidays.add(pres_mondays[2] if len(pres_mondays) > 2 else pres_mondays[-1])
            
            # Good Friday (Friday before Easter) - simplified Easter calculation
            # Using Butcher's algorithm for Easter
            a = year % 19
            b = year // 100
            c = year % 100
            d = b // 4
            e = b % 4
            f = (b + 8) // 25
            g = (b - f + 1) // 3
            h = (19 * a + b - d - g + 15) % 30
            i = c // 4
            k = c % 4
            l = (32 + 2 * e + 2 * i - h - k) % 7
            m = (a + 11 * h + 22 * l) // 451
            month = (h + l - 7 * m + 114) // 31
            day = ((h + l - 7 * m + 114) % 31) + 1
            easter = date(year, month, day)
            good_friday = easter - timedelta(days=2)
            holidays.add(good_friday)
            
            # Memorial Day (last Monday in May)
            may_mondays = [d for d in [date(year, 5, d) for d in range(1, 32)] if d.weekday() == 0]
            holidays.add(may_mondays[-1])
            
            # Juneteenth (June 19, since 2021)
            if year >= 2021:
                jun = date(year, 6, 19)
                if jun.weekday() == 5:
                    holidays.add(date(year, 6, 18))
                elif jun.weekday() == 6:
                    holidays.add(date(year, 6, 20))
                else:
                    holidays.add(jun)
            
            # Independence Day
            july4 = date(year, 7, 4)
            if july4.weekday() == 5:
                holidays.add(date(year, 7, 3))
            elif july4.weekday() == 6:
                holidays.add(date(year, 7, 5))
            else:
                holidays.add(july4)
            
            # Labor Day (1st Monday in September)
            sep_mondays = [d for d in [date(year, 9, d) for d in range(1, 31)] if d.weekday() == 0]
            holidays.add(sep_mondays[0])
            
            # Thanksgiving (4th Thursday in November)
            nov_thursdays = [d for d in [date(year, 11, d) for d in range(1, 31)] if d.weekday() == 3]
            holidays.add(nov_thursdays[3] if len(nov_thursdays) > 3 else nov_thursdays[-1])
            
            # Christmas
            xmas = date(year, 12, 25)
            if xmas.weekday() == 5:
                holidays.add(date(year, 12, 24))
            elif xmas.weekday() == 6:
                holidays.add(date(year, 12, 26))
            else:
                holidays.add(xmas)
    
    return holidays


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


def is_in_middle(price: float, or_high: float, or_low: float, min_distance: float = MIN_DISTANCE_POINTS) -> Tuple[bool, float, float]:
    """
    Check if price is in the middle of the opening range.
    
    Returns:
        (is_middle, dist_to_high, dist_to_low)
    """
    dist_to_high = or_high - price
    dist_to_low = price - or_low
    
    is_middle = (dist_to_high >= min_distance) and (dist_to_low >= min_distance)
    
    return is_middle, dist_to_high, dist_to_low


def check_hits(df: pd.DataFrame, or_high: float, or_low: float, window_start: time, window_end: time) -> Dict:
    """
    Check if opening range high or low gets hit in the specified window.
    
    Returns dict with:
        - high_hit: bool
        - low_hit: bool
        - high_hit_time: Optional[datetime] - time of first high hit
        - low_hit_time: Optional[datetime] - time of first low hit
        - first_hit: str - 'high', 'low', 'both', or 'neither'
    """
    window_data = df.loc[_within(df["ts_et"], window_start, window_end)]
    
    if window_data.empty:
        return {
            'high_hit': False,
            'low_hit': False,
            'high_hit_time': None,
            'low_hit_time': None,
            'first_hit': 'neither'
        }
    
    # Check for high hit
    high_mask = window_data['high'] >= (or_high - EPSILON)
    high_hit = high_mask.any()
    high_hit_time = None
    if high_hit:
        first_high_idx = window_data[high_mask].index[0]
        high_hit_time = window_data.loc[first_high_idx, 'ts_et']
    
    # Check for low hit
    low_mask = window_data['low'] <= (or_low + EPSILON)
    low_hit = low_mask.any()
    low_hit_time = None
    if low_hit:
        first_low_idx = window_data[low_mask].index[0]
        low_hit_time = window_data.loc[first_low_idx, 'ts_et']
    
    # Determine first hit
    if high_hit and low_hit:
        if high_hit_time <= low_hit_time:
            first_hit = 'high'
        else:
            first_hit = 'low'
    elif high_hit:
        first_hit = 'high'
    elif low_hit:
        first_hit = 'low'
    else:
        first_hit = 'neither'
    
    return {
        'high_hit': high_hit,
        'low_hit': low_hit,
        'high_hit_time': high_hit_time,
        'low_hit_time': low_hit_time,
        'first_hit': first_hit
    }


def process_day(g: pd.DataFrame) -> Optional[Dict]:
    """
    Process a single trading day.
    
    Returns dict with analysis results or None if day cannot be processed.
    """
    g = g.sort_values("ts_et").reset_index(drop=True)
    
    # Extract opening range (9:30-9:45 AM ET)
    or_mask = _within(g["ts_et"], OR_START, OR_END)
    or_data = g.loc[or_mask]
    
    if or_data.empty:
        return None
    
    or_high = float(or_data["high"].max())
    or_low = float(or_data["low"].min())
    or_range = or_high - or_low
    
    # Check if range is valid (at least 40 points to allow 20 points on each side)
    if or_range < (MIN_DISTANCE_POINTS * 2):
        return None
    
    # Track when high and low formed
    high_idx = or_data["high"].idxmax()
    low_idx = or_data["low"].idxmin()
    high_time = or_data.loc[high_idx, "ts_et"]
    low_time = or_data.loc[low_idx, "ts_et"]
    high_formed_first = high_time < low_time
    
    # Get price at 9:45 AM (use the 9:45 bar)
    check_mask = _within(g["ts_et"], CHECK_TIME, time(9, 46))
    check_data = g.loc[check_mask]
    
    if check_data.empty:
        return None
    
    # Use the first bar at 9:45 (the 9:45-9:46 bar)
    price_945 = float(check_data.iloc[0]["close"])  # Use close of 9:45 bar
    
    # Check if price is in the middle
    is_middle, dist_to_high, dist_to_low = is_in_middle(price_945, or_high, or_low)
    
    # Check hits in 9:45-10:15 window
    hit_results = check_hits(g, or_high, or_low, CHECK_TIME, HIT_WINDOW_END)
    
    # Build result dict
    result = {
        'date': g["et_date"].iloc[0],
        'symbol': g["symbol"].iloc[0],
        'or_high': or_high,
        'or_low': or_low,
        'or_range': or_range,
        'price_945': price_945,
        'dist_to_high': dist_to_high,
        'dist_to_low': dist_to_low,
        'is_in_middle': is_middle,
        'high_formed_first': high_formed_first,
        'high_hit': hit_results['high_hit'],
        'low_hit': hit_results['low_hit'],
        'high_hit_time': hit_results['high_hit_time'],
        'low_hit_time': hit_results['low_hit_time'],
        'first_hit': hit_results['first_hit'],
    }
    
    return result


def main():
    """Main processing function."""
    print("Starting Opening Range Middle Analysis...")
    
    tz_et = pytz.timezone("US/Eastern")
    all_results = []
    residual = pd.DataFrame()
    
    # Load US market holidays
    print("Loading US market holidays...")
    market_holidays = _get_us_market_holidays(start_year=2020, end_year=2026)
    print(f"  Found {len(market_holidays)} market holidays")
    
    print("Loading and processing data...")
    
    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    dtypes = {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "symbol": "string"
    }
    
    chunk_count = 0
    for chunk in pd.read_csv(CSV_FILE, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE):
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"  Processed {chunk_count} chunks...")
        
        df = pd.concat([residual, chunk], ignore_index=True)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz_et)
        df["et_date"] = df["ts_et"].dt.date
        
        # Weekdays only
        df = df[df["ts_et"].dt.weekday <= 4]
        if df.empty:
            continue
        
        # Keep only outright NQ contracts (avoid spreads)
        df = df[df["symbol"].astype(str).str.match(NQ_SINGLE_CONTRACT_RE)]
        
        if df.empty:
            continue
        
        # Split off the final day to complete next chunk
        max_date = df["et_date"].max()
        to_proc = df[df["et_date"] < max_date]
        residual = df[df["et_date"] == max_date]
        
        # Filter to front-month per date
        if not to_proc.empty:
            to_proc = _determine_front_month(to_proc)
        
        # Filter out market holidays
        to_proc = to_proc[~to_proc["et_date"].isin(market_holidays)]
        
        # Process each day
        for (sym, d), g in to_proc.groupby(["symbol", "et_date"]):
            result = process_day(g)
            if result is not None:
                all_results.append(result)
    
    # Process final residual
    if not residual.empty:
        residual = _determine_front_month(residual)
        # Filter out market holidays
        residual = residual[~residual["et_date"].isin(market_holidays)]
        for (sym, d), g in residual.groupby(["symbol", "et_date"]):
            result = process_day(g)
            if result is not None:
                all_results.append(result)
    
    if not all_results:
        print("No data found!")
        return
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    print(f"\nProcessed {len(df_results)} trading days")
    
    # Filter to days where price was in middle
    df_middle = df_results[df_results["is_in_middle"] == True].copy()
    print(f"Days where price was in middle at 9:45: {len(df_middle)}")
    
    if df_middle.empty:
        print("No days found where price was in the middle!")
        return
    
    # Calculate summary statistics
    total_days = len(df_results)
    middle_days = len(df_middle)
    
    high_hit_count = df_middle["high_hit"].sum()
    low_hit_count = df_middle["low_hit"].sum()
    both_hit_count = ((df_middle["high_hit"]) & (df_middle["low_hit"])).sum()
    neither_hit_count = ((~df_middle["high_hit"]) & (~df_middle["low_hit"])).sum()
    
    high_hit_rate = high_hit_count / middle_days * 100
    low_hit_rate = low_hit_count / middle_days * 100
    both_hit_rate = both_hit_count / middle_days * 100
    neither_hit_rate = neither_hit_count / middle_days * 100
    
    # First hit analysis
    first_high_count = (df_middle["first_hit"] == "high").sum()
    first_low_count = (df_middle["first_hit"] == "low").sum()
    first_high_rate = first_high_count / middle_days * 100
    first_low_rate = first_low_count / middle_days * 100
    
    # Average metrics
    avg_dist_to_high = df_middle["dist_to_high"].mean()
    avg_dist_to_low = df_middle["dist_to_low"].mean()
    avg_or_range = df_middle["or_range"].mean()
    
    # Create summary statistics
    summary_stats = {
        'metric': [
            'total_days_analyzed',
            'days_in_middle_at_945',
            'middle_days_pct',
            'high_hit_count',
            'high_hit_rate_pct',
            'low_hit_count',
            'low_hit_rate_pct',
            'both_hit_count',
            'both_hit_rate_pct',
            'neither_hit_count',
            'neither_hit_rate_pct',
            'first_hit_high_count',
            'first_hit_high_rate_pct',
            'first_hit_low_count',
            'first_hit_low_rate_pct',
            'avg_dist_to_high',
            'avg_dist_to_low',
            'avg_opening_range_size',
        ],
        'value': [
            total_days,
            middle_days,
            middle_days / total_days * 100,
            high_hit_count,
            high_hit_rate,
            low_hit_count,
            low_hit_rate,
            both_hit_count,
            both_hit_rate,
            neither_hit_count,
            neither_hit_rate,
            first_high_count,
            first_high_rate,
            first_low_count,
            first_low_rate,
            avg_dist_to_high,
            avg_dist_to_low,
            avg_or_range,
        ]
    }
    
    df_summary = pd.DataFrame(summary_stats)
    summary_file = OUT_DIR / "opening_range_middle_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSummary statistics saved to: {summary_file}")
    
    # Create detailed results file
    detailed_file = OUT_DIR / "opening_range_middle_detailed.csv"
    df_results.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")
    
    # Create qualifying days list (last RECENT_DAYS days where price was in middle)
    df_middle_sorted = df_middle.sort_values("date", ascending=False)
    df_qualifying = df_middle_sorted.head(RECENT_DAYS).copy()
    
    # Format hit times for readability
    df_qualifying["high_hit_time_str"] = df_qualifying["high_hit_time"].apply(
        lambda x: x.strftime("%H:%M:%S") if pd.notna(x) else None
    )
    df_qualifying["low_hit_time_str"] = df_qualifying["low_hit_time"].apply(
        lambda x: x.strftime("%H:%M:%S") if pd.notna(x) else None
    )
    
    # Select columns for qualifying days output
    qualifying_cols = [
        'date', 'symbol',
        'or_high', 'or_low', 'or_range',
        'price_945',
        'dist_to_high', 'dist_to_low',
        'high_formed_first',
        'high_hit', 'low_hit', 'first_hit',
        'high_hit_time_str', 'low_hit_time_str',
    ]
    
    df_qualifying_output = df_qualifying[qualifying_cols].copy()
    qualifying_file = OUT_DIR / "opening_range_middle_days.csv"
    df_qualifying_output.to_csv(qualifying_file, index=False)
    print(f"Qualifying days list ({len(df_qualifying)} days) saved to: {qualifying_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("OPENING RANGE MIDDLE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total days analyzed: {total_days}")
    print(f"Days in middle at 9:45 AM: {middle_days} ({middle_days/total_days*100:.1f}%)")
    print(f"\nHit Statistics (for days in middle):")
    print(f"  High hit: {high_hit_count} ({high_hit_rate:.1f}%)")
    print(f"  Low hit: {low_hit_count} ({low_hit_rate:.1f}%)")
    print(f"  Both hit: {both_hit_count} ({both_hit_rate:.1f}%)")
    print(f"  Neither hit: {neither_hit_count} ({neither_hit_rate:.1f}%)")
    print(f"\nFirst Hit Statistics:")
    print(f"  High hit first: {first_high_count} ({first_high_rate:.1f}%)")
    print(f"  Low hit first: {first_low_count} ({first_low_rate:.1f}%)")
    print(f"\nAverage Metrics:")
    print(f"  Avg distance to high: {avg_dist_to_high:.2f} points")
    print(f"  Avg distance to low: {avg_dist_to_low:.2f} points")
    print(f"  Avg opening range size: {avg_or_range:.2f} points")
    print("="*60)


if __name__ == "__main__":
    main()
