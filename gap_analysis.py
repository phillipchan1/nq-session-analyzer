#!/usr/bin/env python3
"""
15-Minute Gap Analysis Script
Deep dive into what makes 15min gaps more likely to be hit.

Analyzes:
- Gap size (in points)
- Distance from 9:30 open
- Gap creation time (which session)
- Gap direction (up/down)
- Gap duration (how long it lasted)
- Volume at gap creation
- Time since gap creation
- Gap position relative to recent price action
- Gap context (trend, volatility, etc.)
"""

import re, sys
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
from collections import defaultdict

# -------------------- CONFIG --------------------
CSV_FILE = "glbx-mdp3-20200927-20250926.ohlcv-1m.csv"
OUT_FILE = "gap_analysis.csv"
DETAILED_OUT_FILE = "gap_detailed.csv"

ASIA_START = time(21, 0)  # 9 PM ET
ASIA_END = time(3, 0)     # 3 AM ET
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

# Gap size buckets (in points)
GAP_SIZE_BUCKETS = [0, 2, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000]

# Distance buckets (in points)
DISTANCE_BUCKETS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000]

WEEKDAYS_ONLY = True
CHUNKSIZE = 1_000_000
NQ_SINGLE_CONTRACT_RE = re.compile(r"^NQ[A-Z]\d{1,2}$")
# ------------------------------------------------

def within(ts, start_t, end_t):
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)

def get_session_name(timestamp):
    """Get session name for a timestamp"""
    t = timestamp.time()
    if ASIA_START <= t or t < ASIA_END:
        return "asia"
    elif LONDON_START <= t < LONDON_END:
        return "london"
    elif NY_OPEN <= t < time(16, 0):
        return "ny"
    else:
        return "after_hours"

def get_gap_size_bucket(gap_size):
    """Get gap size bucket for a given size"""
    for i, bucket in enumerate(GAP_SIZE_BUCKETS[1:], 1):
        if gap_size <= bucket:
            return f"{GAP_SIZE_BUCKETS[i-1]}-{bucket}"
    return f"{GAP_SIZE_BUCKETS[-1]}+"

def get_distance_bucket(distance):
    """Get distance bucket for a given distance"""
    for i, bucket in enumerate(DISTANCE_BUCKETS[1:], 1):
        if distance <= bucket:
            return f"{DISTANCE_BUCKETS[i-1]}-{bucket}"
    return f"{DISTANCE_BUCKETS[-1]}+"

def find_15min_gaps(df):
    """Find all 15min+ gaps in the data"""
    df_sorted = df.sort_values('ts_et')
    gaps = []
    
    for i in range(len(df_sorted) - 1):
        current = df_sorted.iloc[i]
        next_bar = df_sorted.iloc[i + 1]
        
        # Check if gap is at least 15 minutes
        time_diff = (next_bar['ts_et'] - current['ts_et']).total_seconds() / 60
        if time_diff >= 15:
            gap_high = current['high']
            gap_low = next_bar['low']
            gap_size = 0
            gap_direction = None
            gap_level = None
            
            if gap_high < gap_low:  # Upward gap
                gap_size = gap_low - gap_high
                gap_direction = 'up'
                gap_level = gap_high  # Use the high as the level to hit
            elif gap_low > gap_high:  # Downward gap
                gap_size = gap_high - gap_low
                gap_direction = 'down'
                gap_level = gap_low  # Use the low as the level to hit
            
            if gap_size > 0:  # Only include actual gaps
                gaps.append({
                    'gap_start': current['ts_et'],
                    'gap_end': next_bar['ts_et'],
                    'gap_duration_minutes': time_diff,
                    'gap_size': gap_size,
                    'gap_direction': gap_direction,
                    'gap_level': gap_level,
                    'gap_high': gap_high,
                    'gap_low': gap_low,
                    'pre_gap_volume': current['volume'],
                    'post_gap_volume': next_bar['volume'],
                    'session_created': get_session_name(current['ts_et']),
                    'gap_size_bucket': get_gap_size_bucket(gap_size)
                })
    
    return gaps

def calculate_price_context(df, gap_time, lookback_hours=4):
    """Calculate price context around gap creation"""
    lookback_time = gap_time - timedelta(hours=lookback_hours)
    context_data = df[df['ts_et'] >= lookback_time]
    
    if context_data.empty:
        return {}
    
    # Calculate recent volatility (ATR-like)
    recent_highs = context_data['high'].rolling(window=20, min_periods=1).max()
    recent_lows = context_data['low'].rolling(window=20, min_periods=1).min()
    volatility = (recent_highs - recent_lows).mean()
    
    # Calculate recent trend (simple slope)
    recent_prices = context_data['close'].tail(20)
    if len(recent_prices) >= 2:
        trend_slope = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
    else:
        trend_slope = 0
    
    # Calculate volume context
    avg_volume = context_data['volume'].mean()
    recent_volume = context_data['volume'].tail(5).mean()
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
    
    return {
        'recent_volatility': volatility,
        'trend_slope': trend_slope,
        'volume_ratio': volume_ratio,
        'recent_high': context_data['high'].max(),
        'recent_low': context_data['low'].min(),
        'recent_range': context_data['high'].max() - context_data['low'].min()
    }

def process_day(g):
    """Process a single trading day for gap analysis"""
    # Get 9:30 opening price
    open_930 = g.loc[within(g['ts_et'], NY_OPEN, time(9,31))]
    if open_930.empty:
        return None
    open_price = open_930.iloc[0]['open']
    current_date = g['ts_et'].dt.date.iloc[0]
    
    # Find all 15min+ gaps
    gaps = find_15min_gaps(g)
    if not gaps:
        return None
    
    results = []
    
    for gap in gaps:
        # Check if gap is unmitigated (not filled before NY open)
        gap_filled = False
        ny_open_ts = pd.Timestamp.combine(current_date, NY_OPEN).tz_localize(g['ts_et'].dt.tz)
        
        if gap['gap_direction'] == 'up':
            after_gap = g[g['ts_et'] > gap['gap_end']]
            before_open = after_gap[after_gap['ts_et'] < ny_open_ts]
            if not before_open.empty and (before_open['low'] <= gap['gap_level']).any():
                gap_filled = True
        else:  # down gap
            after_gap = g[g['ts_et'] > gap['gap_end']]
            before_open = after_gap[after_gap['ts_et'] < ny_open_ts]
            if not before_open.empty and (before_open['high'] >= gap['gap_level']).any():
                gap_filled = True
        
        if gap_filled:
            continue  # Skip filled gaps
        
        # Calculate distance from open
        distance_from_open = abs(gap['gap_level'] - open_price)
        
        # Calculate time since gap creation
        time_since_gap = (ny_open_ts - gap['gap_start']).total_seconds() / 3600  # hours
        
        # Calculate price context
        context = calculate_price_context(g, gap['gap_start'])
        
        # Create gap result
        gap_result = {
            'date': current_date,
            'open_price': open_price,
            'gap_level': gap['gap_level'],
            'distance_from_open': distance_from_open,
            'distance_bucket': get_distance_bucket(distance_from_open),
            'gap_size': gap['gap_size'],
            'gap_size_bucket': gap['gap_size_bucket'],
            'gap_direction': gap['gap_direction'],
            'gap_duration_minutes': gap['gap_duration_minutes'],
            'session_created': gap['session_created'],
            'time_since_gap_hours': time_since_gap,
            'pre_gap_volume': gap['pre_gap_volume'],
            'post_gap_volume': gap['post_gap_volume'],
            'volume_ratio': gap['post_gap_volume'] / gap['pre_gap_volume'] if gap['pre_gap_volume'] > 0 else 1
        }
        
        # Add context data
        gap_result.update(context)
        
        # Check hits for this gap
        for window_start, window_end, window_name in WINDOWS:
            window_data = g.loc[within(g['ts_et'], window_start, window_end)]
            if window_data.empty:
                gap_result[f'{window_name}_hit'] = False
                continue
            
            # Check if gap level was hit in this window
            hit = False
            if gap['gap_direction'] == 'up':
                hit = (window_data['high'] >= gap['gap_level']).any()
            else:
                hit = (window_data['low'] <= gap['gap_level']).any()
            
            gap_result[f'{window_name}_hit'] = hit
        
        results.append(gap_result)
    
    return results

def main():
    tz_et = pytz.timezone("US/Eastern")
    all_results = []
    
    residual = pd.DataFrame()
    
    print("Processing data for 15min gap analysis...")
    
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
                
            for result in results:
                result['symbol'] = sym
                all_results.append(result)

    # Process final residual
    if not residual.empty:
        for (sym,d),g in residual.groupby(['symbol','et_date']):
            g.sort_values('ts_et', inplace=True)
            results = process_day(g)
            if results is None:
                continue
                
            for result in results:
                result['symbol'] = sym
                all_results.append(result)

    if not all_results:
        print("No gap data found!")
        return

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Calculate gap statistics
    summary_stats = []
    
    # Analyze by gap size
    for size_bucket in sorted(df_results['gap_size_bucket'].unique()):
        size_data = df_results[df_results['gap_size_bucket'] == size_bucket]
        
        stats = {
            'analysis_type': 'gap_size',
            'bucket': size_bucket,
            'total_gaps': len(size_data),
            'avg_gap_size': size_data['gap_size'].mean(),
            'avg_distance_from_open': size_data['distance_from_open'].mean(),
            'avg_time_since_gap': size_data['time_since_gap_hours'].mean()
        }
        
        # Calculate hit rates for each window
        for window_start, window_end, window_name in WINDOWS:
            hit_col = f'{window_name}_hit'
            if hit_col in size_data.columns:
                hits = size_data[hit_col].sum()
                hit_rate = hits / len(size_data) * 100
                stats[f'{window_name}_hits'] = hits
                stats[f'{window_name}_hit_rate'] = round(hit_rate, 2)
        
        summary_stats.append(stats)
    
    # Analyze by distance from open
    for dist_bucket in sorted(df_results['distance_bucket'].unique()):
        dist_data = df_results[df_results['distance_bucket'] == dist_bucket]
        
        stats = {
            'analysis_type': 'distance_from_open',
            'bucket': dist_bucket,
            'total_gaps': len(dist_data),
            'avg_distance_from_open': dist_data['distance_from_open'].mean(),
            'avg_gap_size': dist_data['gap_size'].mean()
        }
        
        # Calculate hit rates for each window
        for window_start, window_end, window_name in WINDOWS:
            hit_col = f'{window_name}_hit'
            if hit_col in dist_data.columns:
                hits = dist_data[hit_col].sum()
                hit_rate = hits / len(dist_data) * 100
                stats[f'{window_name}_hits'] = hits
                stats[f'{window_name}_hit_rate'] = round(hit_rate, 2)
        
        summary_stats.append(stats)
    
    # Analyze by session created
    for session in sorted(df_results['session_created'].unique()):
        session_data = df_results[df_results['session_created'] == session]
        
        stats = {
            'analysis_type': 'session_created',
            'bucket': session,
            'total_gaps': len(session_data),
            'avg_gap_size': session_data['gap_size'].mean(),
            'avg_distance_from_open': session_data['distance_from_open'].mean()
        }
        
        # Calculate hit rates for each window
        for window_start, window_end, window_name in WINDOWS:
            hit_col = f'{window_name}_hit'
            if hit_col in session_data.columns:
                hits = session_data[hit_col].sum()
                hit_rate = hits / len(session_data) * 100
                stats[f'{window_name}_hits'] = hits
                stats[f'{window_name}_hit_rate'] = round(hit_rate, 2)
        
        summary_stats.append(stats)
    
    # Analyze by gap direction
    for direction in sorted(df_results['gap_direction'].unique()):
        dir_data = df_results[df_results['gap_direction'] == direction]
        
        stats = {
            'analysis_type': 'gap_direction',
            'bucket': direction,
            'total_gaps': len(dir_data),
            'avg_gap_size': dir_data['gap_size'].mean(),
            'avg_distance_from_open': dir_data['distance_from_open'].mean()
        }
        
        # Calculate hit rates for each window
        for window_start, window_end, window_name in WINDOWS:
            hit_col = f'{window_name}_hit'
            if hit_col in dir_data.columns:
                hits = dir_data[hit_col].sum()
                hit_rate = hits / len(dir_data) * 100
                stats[f'{window_name}_hits'] = hits
                stats[f'{window_name}_hit_rate'] = round(hit_rate, 2)
        
        summary_stats.append(stats)
    
    # Save detailed results
    df_results.to_csv(DETAILED_OUT_FILE, index=False)
    print(f"✅ Wrote detailed gap data to {DETAILED_OUT_FILE}")
    
    # Save summary statistics
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(OUT_FILE, index=False)
    print(f"✅ Wrote gap analysis summary to {OUT_FILE}")
    
    # Print key insights
    print("\n" + "="*80)
    print("15-MINUTE GAP ANALYSIS - Key Insights")
    print("="*80)
    
    print(f"\nTotal unmitigated 15min+ gaps analyzed: {len(df_results)}")
    
    # Gap size insights
    print("\nGAP SIZE ANALYSIS:")
    size_analysis = df_summary[df_summary['analysis_type'] == 'gap_size']
    for _, row in size_analysis.iterrows():
        if '9:30-10:15_hit_rate' in row:
            print(f"  {row['bucket']} points: {row['9:30-10:15_hit_rate']}% hit rate ({row['total_gaps']} gaps)")
    
    # Distance insights
    print("\nDISTANCE FROM OPEN ANALYSIS:")
    dist_analysis = df_summary[df_summary['analysis_type'] == 'distance_from_open']
    for _, row in dist_analysis.iterrows():
        if '9:30-10:15_hit_rate' in row:
            print(f"  {row['bucket']} points: {row['9:30-10:15_hit_rate']}% hit rate ({row['total_gaps']} gaps)")
    
    # Session insights
    print("\nSESSION CREATED ANALYSIS:")
    session_analysis = df_summary[df_summary['analysis_type'] == 'session_created']
    for _, row in session_analysis.iterrows():
        if '9:30-10:15_hit_rate' in row:
            print(f"  {row['bucket']} session: {row['9:30-10:15_hit_rate']}% hit rate ({row['total_gaps']} gaps)")
    
    # Direction insights
    print("\nGAP DIRECTION ANALYSIS:")
    dir_analysis = df_summary[df_summary['analysis_type'] == 'gap_direction']
    for _, row in dir_analysis.iterrows():
        if '9:30-10:15_hit_rate' in row:
            print(f"  {row['bucket']} gaps: {row['9:30-10:15_hit_rate']}% hit rate ({row['total_gaps']} gaps)")
    
    print(f"\nSummary saved to: {OUT_FILE}")
    print(f"Detailed data saved to: {DETAILED_OUT_FILE}")

if __name__ == "__main__":
    main()
