#!/usr/bin/env python3
"""
Liquidity Hit Analysis Script
Analyzes hit probabilities for various liquidity levels based on distance from opening price.

Tracks:
- London High/Low (swing points)
- Hourly High/Low
- Unmitigated 15min gaps
- Unmitigated 1h gaps  
- Unmitigated 4h gaps
- Distance from 9:30 opening price
- Hit probability within different time windows
"""

import re, sys
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
from collections import defaultdict

# -------------------- CONFIG --------------------
CSV_FILE = "glbx-mdp3-20200927-20250926.ohlcv-1m.csv"
OUT_FILE = "liquidity_hit_analysis.csv"
DETAILED_OUT_FILE = "liquidity_hit_detailed.csv"

LONDON_START = time(3, 0)
LONDON_END = time(8, 0)
NY_OPEN = time(9, 30)

# Time windows for analysis
WINDOWS = [
    (time(9,30), time(9,45), "9:30-9:45"),
    (time(9,45), time(10,0), "9:45-10:00"), 
    (time(10,0), time(10,15), "10:00-10:15"),
    (time(9,30), time(10,15), "9:30-10:15")  # First 45min combined
]

# Distance buckets for analysis (in points)
DISTANCE_BUCKETS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000]

SWING_K = 2
EPSILON = 0.0
WEEKDAYS_ONLY = True
CHUNKSIZE = 1_000_000
NQ_SINGLE_CONTRACT_RE = re.compile(r"^NQ[A-Z]\d{1,2}$")
# ------------------------------------------------

def within(ts, start_t, end_t):
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)

def is_swing(high, low, k):
    """Identify swing highs and lows"""
    sh = pd.Series(True, index=high.index)
    sl = pd.Series(True, index=low.index)
    for off in range(1, k+1):
        sh &= (high > high.shift(+off)) & (high > high.shift(-off))
        sl &= (low  <  low.shift(+off)) & (low  <  low.shift(-off))
    sh.iloc[:k] = sh.iloc[-k:] = False
    sl.iloc[:k] = sl.iloc[-k:] = False
    return sh.fillna(False), sl.fillna(False)

def find_hourly_hl(df):
    """Find hourly high and low levels"""
    df['hour'] = df['ts_et'].dt.floor('h')
    hourly = df.groupby('hour').agg({'high': 'max', 'low': 'min'}).reset_index()
    return hourly['high'].max(), hourly['low'].min()

def find_unmitigated_gaps(df, gap_minutes):
    """Find unmitigated gaps of specified duration"""
    df_sorted = df.sort_values('ts_et')
    gaps = []
    
    for i in range(len(df_sorted) - 1):
        current = df_sorted.iloc[i]
        next_bar = df_sorted.iloc[i + 1]
        
        # Check if gap is at least gap_minutes
        time_diff = (next_bar['ts_et'] - current['ts_et']).total_seconds() / 60
        if time_diff >= gap_minutes:
            gap_high = current['high']
            gap_low = next_bar['low']
            if gap_high < gap_low:  # Upward gap
                gaps.append(('up', gap_high, gap_low, current['ts_et'], next_bar['ts_et']))
            elif gap_low > gap_high:  # Downward gap
                gaps.append(('down', gap_low, gap_high, current['ts_et'], next_bar['ts_et']))
    
    return gaps

def get_distance_bucket(distance):
    """Get distance bucket for a given distance"""
    for i, bucket in enumerate(DISTANCE_BUCKETS[1:], 1):
        if distance <= bucket:
            return f"{DISTANCE_BUCKETS[i-1]}-{bucket}"
    return f"{DISTANCE_BUCKETS[-1]}+"

def process_day(g):
    """Process a single trading day"""
    # Get 9:30 opening price
    open_930 = g.loc[within(g['ts_et'], NY_OPEN, time(9,31))]
    if open_930.empty:
        return None
    open_price = open_930.iloc[0]['open']
    
    results = []
    
    # 1. London High/Low (swing points)
    london = g.loc[within(g['ts_et'], LONDON_START, LONDON_END)]
    if not london.empty:
        sh, sl = is_swing(london['high'], london['low'], SWING_K)
        idx_h = london['high'].idxmax()
        idx_l = london['low'].idxmin()
        london_high, london_low = london.at[idx_h,'high'], london.at[idx_l,'low']
        high_is_swing, low_is_swing = sh.get(idx_h, False), sl.get(idx_l, False)
        
        # Check if levels are available (not hit in pre-open)
        pre_open = g.loc[within(g['ts_et'], LONDON_END, NY_OPEN)]
        if high_is_swing and not (pre_open['high'] >= london_high - 1e-12 + EPSILON).any():
            distance = abs(london_high - open_price)
            results.append({
                'liquidity_type': 'london_high',
                'level': london_high,
                'distance': distance,
                'distance_bucket': get_distance_bucket(distance),
                'is_swing': True
            })
        
        if low_is_swing and not (pre_open['low'] <= london_low + 1e-12 - EPSILON).any():
            distance = abs(london_low - open_price)
            results.append({
                'liquidity_type': 'london_low',
                'level': london_low,
                'distance': distance,
                'distance_bucket': get_distance_bucket(distance),
                'is_swing': True
            })
    
    # 2. Hourly High/Low
    hourly_high, hourly_low = find_hourly_hl(g)
    if hourly_high is not None:
        distance = abs(hourly_high - open_price)
        results.append({
            'liquidity_type': 'hourly_high',
            'level': hourly_high,
            'distance': distance,
            'distance_bucket': get_distance_bucket(distance),
            'is_swing': False
        })
    
    if hourly_low is not None:
        distance = abs(hourly_low - open_price)
        results.append({
            'liquidity_type': 'hourly_low',
            'level': hourly_low,
            'distance': distance,
            'distance_bucket': get_distance_bucket(distance),
            'is_swing': False
        })
    
    # 3. Unmitigated gaps
    for gap_minutes, gap_name in [(15, '15min'), (60, '1h'), (240, '4h')]:
        gaps = find_unmitigated_gaps(g, gap_minutes)
        if not gaps:  # Skip if no gaps found
            continue
        for gap_type, gap_level1, gap_level2, start_time, end_time in gaps:
            # Check if gap is unmitigated (not filled before NY open)
            gap_filled = False
            # Create timezone-aware NY open timestamp
            ny_open_ts = pd.Timestamp.combine(g['ts_et'].dt.date.iloc[0], NY_OPEN).tz_localize(g['ts_et'].dt.tz)
            
            if gap_type == 'up':
                # Check if price came back down to fill the gap
                after_gap = g[g['ts_et'] > end_time]
                before_open = after_gap[after_gap['ts_et'] < ny_open_ts]
                if not before_open.empty and (before_open['low'] <= gap_level1).any():
                    gap_filled = True
            else:  # down gap
                # Check if price came back up to fill the gap
                after_gap = g[g['ts_et'] > end_time]
                before_open = after_gap[after_gap['ts_et'] < ny_open_ts]
                if not before_open.empty and (before_open['high'] >= gap_level2).any():
                    gap_filled = True
            
            if not gap_filled:
                # Use the gap level closest to current price
                gap_level = gap_level1 if abs(gap_level1 - open_price) < abs(gap_level2 - open_price) else gap_level2
                distance = abs(gap_level - open_price)
                results.append({
                    'liquidity_type': f'{gap_name}_gap_{gap_type}',
                    'level': gap_level,
                    'distance': distance,
                    'distance_bucket': get_distance_bucket(distance),
                    'is_swing': False,
                    'gap_size': abs(gap_level2 - gap_level1)
                })
    
    # Check hits for each liquidity level
    for result in results:
        level = result['level']
        for window_start, window_end, window_name in WINDOWS:
            window_data = g.loc[within(g['ts_et'], window_start, window_end)]
            if window_data.empty:
                result[f'{window_name}_hit'] = False
                continue
            
            # Check if level was hit in this window
            hit = False
            if 'high' in result['liquidity_type'] or 'up' in result['liquidity_type']:
                hit = (window_data['high'] >= level - 1e-12 + EPSILON).any()
            else:
                hit = (window_data['low'] <= level + 1e-12 - EPSILON).any()
            
            result[f'{window_name}_hit'] = hit
    
    return results, open_price

def main():
    tz_et = pytz.timezone("US/Eastern")
    all_results = []
    daily_stats = defaultdict(lambda: defaultdict(int))
    
    residual = pd.DataFrame()
    
    print("Processing data...")
    
    for chunk_num, chunk in enumerate(pd.read_csv(CSV_FILE,
                             usecols=['ts_event','open','high','low','close','volume','symbol'],
                             dtype={'open':'float64','high':'float64','low':'float64',
                                    'close':'float64','volume':'int64','symbol':'string'},
                             chunksize=CHUNKSIZE)):
        print(f"Processing chunk {chunk_num + 1}...")
        
        df = pd.concat([residual, chunk])
        df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True, errors='coerce')
        df.dropna(subset=['ts_event'], inplace=True)
        df['ts_et'] = df['ts_event'].dt.tz_convert(tz_et)
        df['et_date'] = df['ts_et'].dt.date
        df = df[df['symbol'].astype(str).str.match(NQ_SINGLE_CONTRACT_RE)]
        
        if WEEKDAYS_ONLY:
            df = df[df['ts_et'].dt.weekday <= 4]
        
        if df.empty: 
            continue

        max_date = df['et_date'].max()
        to_proc = df[df['et_date'] < max_date]
        residual = df[df['et_date'] == max_date]
        
        for (sym, d), g in to_proc.groupby(['symbol','et_date']):
            g.sort_values('ts_et', inplace=True)
            results = process_day(g)
            if results is None:
                continue
                
            day_results, open_price = results
            for result in day_results:
                result['symbol'] = sym
                result['date'] = d
                result['open_price'] = open_price
                all_results.append(result)
                
                # Track daily stats
                daily_stats[result['liquidity_type']][result['distance_bucket']] += 1

    # Process final residual
    if not residual.empty:
        for (sym,d),g in residual.groupby(['symbol','et_date']):
            g.sort_values('ts_et', inplace=True)
            results = process_day(g)
            if results is None:
                continue
                
            day_results, open_price = results
            for result in day_results:
                result['symbol'] = sym
                result['date'] = d
                result['open_price'] = open_price
                all_results.append(result)
                
                # Track daily stats
                daily_stats[result['liquidity_type']][result['distance_bucket']] += 1

    if not all_results:
        print("No data found!")
        return

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Calculate hit probabilities by distance bucket and liquidity type
    summary_stats = []
    
    for liquidity_type in df_results['liquidity_type'].unique():
        type_data = df_results[df_results['liquidity_type'] == liquidity_type]
        
        for distance_bucket in DISTANCE_BUCKETS[1:]:
            bucket_name = f"{DISTANCE_BUCKETS[DISTANCE_BUCKETS.index(distance_bucket)-1]}-{distance_bucket}"
            if distance_bucket == DISTANCE_BUCKETS[-1]:
                bucket_name = f"{DISTANCE_BUCKETS[-1]}+"
            
            bucket_data = type_data[type_data['distance_bucket'] == bucket_name]
            if bucket_data.empty:
                continue
            
            stats = {
                'liquidity_type': liquidity_type,
                'distance_bucket': bucket_name,
                'total_levels': len(bucket_data),
                'avg_distance': bucket_data['distance'].mean(),
                'avg_gap_size': bucket_data['gap_size'].mean() if 'gap_size' in bucket_data.columns else None
            }
            
            # Calculate hit rates for each window
            for window_start, window_end, window_name in WINDOWS:
                hit_col = f'{window_name}_hit'
                if hit_col in bucket_data.columns:
                    hits = bucket_data[hit_col].sum()
                    hit_rate = hits / len(bucket_data) * 100
                    stats[f'{window_name}_hits'] = hits
                    stats[f'{window_name}_hit_rate'] = round(hit_rate, 2)
            
            summary_stats.append(stats)
    
    # Save detailed results
    df_results.to_csv(DETAILED_OUT_FILE, index=False)
    print(f"✅ Wrote detailed results to {DETAILED_OUT_FILE}")
    
    # Save summary statistics
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(OUT_FILE, index=False)
    print(f"✅ Wrote summary to {OUT_FILE}")
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS - Hit Rates by Distance")
    print("="*80)
    
    for liquidity_type in ['london_high', 'london_low', 'hourly_high', 'hourly_low']:
        if liquidity_type in df_summary['liquidity_type'].values:
            print(f"\n{liquidity_type.upper()}:")
            type_summary = df_summary[df_summary['liquidity_type'] == liquidity_type]
            for _, row in type_summary.iterrows():
                if '9:30-10:15_hit_rate' in row:
                    print(f"  {row['distance_bucket']} points: {row['9:30-10:15_hit_rate']}% hit rate ({row['total_levels']} levels)")
    
    print(f"\nTotal liquidity levels analyzed: {len(df_results)}")
    print(f"Summary saved to: {OUT_FILE}")
    print(f"Detailed data saved to: {DETAILED_OUT_FILE}")

if __name__ == "__main__":
    main()
