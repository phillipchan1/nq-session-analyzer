#!/usr/bin/env python3
"""
Confluence Analysis Script
Analyzes hit probabilities when multiple liquidity levels are clustered together.

Tracks:
- Asia High/Low (21:00-03:00 ET)
- London High/Low (swing points)
- Hourly High/Low
- Unmitigated gaps (15min, 1h, 4h)
- Confluence zones where multiple levels are within X points of each other
- Hit rates based on confluence strength
"""

import re, sys
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
from collections import defaultdict

# -------------------- CONFIG --------------------
CSV_FILE = "glbx-mdp3-20200927-20250926.ohlcv-1m.csv"
OUT_FILE = "confluence_analysis.csv"
DETAILED_OUT_FILE = "confluence_detailed.csv"

ASIA_START = time(21, 0)  # 9 PM ET (Asia session)
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

# Confluence proximity thresholds (in points)
CONFLUENCE_THRESHOLDS = [5, 10, 15, 20, 25, 30]

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

def find_asia_hl(df):
    """Find Asia session high and low levels"""
    asia = df.loc[within(df['ts_et'], ASIA_START, ASIA_END)]
    if asia.empty:
        return None, None
    return asia['high'].max(), asia['low'].min()

def find_4h_hl(df):
    """Find 4-hour high and low levels"""
    df['4h_period'] = df['ts_et'].dt.floor('4h')
    four_hourly = df.groupby('4h_period').agg({'high': 'max', 'low': 'min'}).reset_index()
    return four_hourly['high'].max(), four_hourly['low'].min()

def find_daily_hl(df):
    """Find daily high and low levels"""
    daily = df.groupby(df['ts_et'].dt.date).agg({'high': 'max', 'low': 'min'}).reset_index()
    return daily['high'].max(), daily['low'].min()

def find_prev_day_hl(df, current_date):
    """Find previous day high and low levels"""
    prev_date = current_date - timedelta(days=1)
    prev_data = df[df['ts_et'].dt.date == prev_date]
    if prev_data.empty:
        return None, None
    return prev_data['high'].max(), prev_data['low'].min()

def find_london_hl(df):
    """Find London session high and low levels (swing points)"""
    london = df.loc[within(df['ts_et'], LONDON_START, LONDON_END)]
    if london.empty:
        return None, None, False, False
    
    sh, sl = is_swing(london['high'], london['low'], SWING_K)
    idx_h = london['high'].idxmax()
    idx_l = london['low'].idxmin()
    london_high, london_low = london.at[idx_h,'high'], london.at[idx_l,'low']
    high_is_swing, low_is_swing = sh.get(idx_h, False), sl.get(idx_l, False)
    
    return london_high, london_low, high_is_swing, low_is_swing

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

def find_daily_gaps(df, current_date):
    """Find daily gaps (gaps between previous day close and current day open)"""
    prev_date = current_date - timedelta(days=1)
    prev_data = df[df['ts_et'].dt.date == prev_date]
    current_data = df[df['ts_et'].dt.date == current_date]
    
    if prev_data.empty or current_data.empty:
        return []
    
    prev_close = prev_data.iloc[-1]['close']
    current_open = current_data.iloc[0]['open']
    
    gaps = []
    if prev_close < current_open:  # Upward daily gap
        gaps.append(('up', prev_close, current_open, prev_data.iloc[-1]['ts_et'], current_data.iloc[0]['ts_et']))
    elif prev_close > current_open:  # Downward daily gap
        gaps.append(('down', current_open, prev_close, prev_data.iloc[-1]['ts_et'], current_data.iloc[0]['ts_et']))
    
    return gaps

def find_confluence_zones(levels, open_price, threshold):
    """Find confluence zones where multiple levels are within threshold points of each other"""
    if len(levels) < 2:
        return []
    
    confluence_zones = []
    
    # Group levels by proximity
    for i, level1 in enumerate(levels):
        zone = [level1]
        zone_types = [level1['type']]
        
        for j, level2 in enumerate(levels[i+1:], i+1):
            if abs(level1['level'] - level2['level']) <= threshold:
                zone.append(level2)
                zone_types.append(level2['type'])
        
        if len(zone) >= 2:  # At least 2 levels in confluence
            # Calculate zone center and distance from open
            zone_levels = [l['level'] for l in zone]
            zone_center = np.mean(zone_levels)
            distance_from_open = abs(zone_center - open_price)
            
            confluence_zones.append({
                'zone_center': zone_center,
                'distance_from_open': distance_from_open,
                'confluence_strength': len(zone),
                'level_types': zone_types,
                'levels': zone,
                'max_level': max(zone_levels),
                'min_level': min(zone_levels),
                'zone_width': max(zone_levels) - min(zone_levels)
            })
    
    # Remove duplicate zones (keep strongest)
    unique_zones = []
    for zone in confluence_zones:
        is_duplicate = False
        for existing in unique_zones:
            if abs(zone['zone_center'] - existing['zone_center']) <= threshold:
                if zone['confluence_strength'] > existing['confluence_strength']:
                    unique_zones.remove(existing)
                    unique_zones.append(zone)
                is_duplicate = True
                break
        if not is_duplicate:
            unique_zones.append(zone)
    
    return unique_zones

def process_day(g):
    """Process a single trading day for confluence analysis"""
    # Get 9:30 opening price
    open_930 = g.loc[within(g['ts_et'], NY_OPEN, time(9,31))]
    if open_930.empty:
        return None
    open_price = open_930.iloc[0]['open']
    current_date = g['ts_et'].dt.date.iloc[0]
    
    # Collect all available liquidity levels
    levels = []
    
    # 1. Asia High/Low
    asia_high, asia_low = find_asia_hl(g)
    if asia_high is not None:
        levels.append({
            'type': 'asia_high',
            'level': asia_high,
            'distance': abs(asia_high - open_price),
            'is_swing': False
        })
    if asia_low is not None:
        levels.append({
            'type': 'asia_low',
            'level': asia_low,
            'distance': abs(asia_low - open_price),
            'is_swing': False
        })
    
    # 2. 4-Hour High/Low
    four_h_high, four_h_low = find_4h_hl(g)
    if four_h_high is not None:
        levels.append({
            'type': '4h_high',
            'level': four_h_high,
            'distance': abs(four_h_high - open_price),
            'is_swing': False
        })
    if four_h_low is not None:
        levels.append({
            'type': '4h_low',
            'level': four_h_low,
            'distance': abs(four_h_low - open_price),
            'is_swing': False
        })
    
    # 3. Daily High/Low
    daily_high, daily_low = find_daily_hl(g)
    if daily_high is not None:
        levels.append({
            'type': 'daily_high',
            'level': daily_high,
            'distance': abs(daily_high - open_price),
            'is_swing': False
        })
    if daily_low is not None:
        levels.append({
            'type': 'daily_low',
            'level': daily_low,
            'distance': abs(daily_low - open_price),
            'is_swing': False
        })
    
    # 4. Previous Day High/Low
    prev_high, prev_low = find_prev_day_hl(g, current_date)
    if prev_high is not None:
        levels.append({
            'type': 'prev_day_high',
            'level': prev_high,
            'distance': abs(prev_high - open_price),
            'is_swing': False
        })
    if prev_low is not None:
        levels.append({
            'type': 'prev_day_low',
            'level': prev_low,
            'distance': abs(prev_low - open_price),
            'is_swing': False
        })
    
    # 5. London High/Low (swing points)
    london_high, london_low, high_is_swing, low_is_swing = find_london_hl(g)
    if london_high is not None and high_is_swing:
        # Check if level is available (not hit in pre-open)
        pre_open = g.loc[within(g['ts_et'], LONDON_END, NY_OPEN)]
        if not (pre_open['high'] >= london_high - 1e-12 + EPSILON).any():
            levels.append({
                'type': 'london_high',
                'level': london_high,
                'distance': abs(london_high - open_price),
                'is_swing': True
            })
    
    if london_low is not None and low_is_swing:
        # Check if level is available (not hit in pre-open)
        pre_open = g.loc[within(g['ts_et'], LONDON_END, NY_OPEN)]
        if not (pre_open['low'] <= london_low + 1e-12 - EPSILON).any():
            levels.append({
                'type': 'london_low',
                'level': london_low,
                'distance': abs(london_low - open_price),
                'is_swing': True
            })
    
    # 6. Hourly High/Low
    hourly_high, hourly_low = find_hourly_hl(g)
    if hourly_high is not None:
        levels.append({
            'type': 'hourly_high',
            'level': hourly_high,
            'distance': abs(hourly_high - open_price),
            'is_swing': False
        })
    if hourly_low is not None:
        levels.append({
            'type': 'hourly_low',
            'level': hourly_low,
            'distance': abs(hourly_low - open_price),
            'is_swing': False
        })
    
    # 7. Unmitigated gaps (15min, 1h, 4h)
    for gap_minutes, gap_name in [(15, '15min'), (60, '1h'), (240, '4h')]:
        gaps = find_unmitigated_gaps(g, gap_minutes)
        if not gaps:
            continue
            
        for gap_type, gap_level1, gap_level2, start_time, end_time in gaps:
            # Check if gap is unmitigated (not filled before NY open)
            gap_filled = False
            ny_open_ts = pd.Timestamp.combine(g['ts_et'].dt.date.iloc[0], NY_OPEN).tz_localize(g['ts_et'].dt.tz)
            
            if gap_type == 'up':
                after_gap = g[g['ts_et'] > end_time]
                before_open = after_gap[after_gap['ts_et'] < ny_open_ts]
                if not before_open.empty and (before_open['low'] <= gap_level1).any():
                    gap_filled = True
            else:  # down gap
                after_gap = g[g['ts_et'] > end_time]
                before_open = after_gap[after_gap['ts_et'] < ny_open_ts]
                if not before_open.empty and (before_open['high'] >= gap_level2).any():
                    gap_filled = True
            
            if not gap_filled:
                # Use the gap level closest to current price
                gap_level = gap_level1 if abs(gap_level1 - open_price) < abs(gap_level2 - open_price) else gap_level2
                levels.append({
                    'type': f'{gap_name}_gap_{gap_type}',
                    'level': gap_level,
                    'distance': abs(gap_level - open_price),
                    'is_swing': False,
                    'gap_size': abs(gap_level2 - gap_level1)
                })
    
    # 8. Daily gaps (overnight gaps)
    daily_gaps = find_daily_gaps(g, current_date)
    for gap_type, gap_level1, gap_level2, start_time, end_time in daily_gaps:
        # Use the gap level closest to current price
        gap_level = gap_level1 if abs(gap_level1 - open_price) < abs(gap_level2 - open_price) else gap_level2
        levels.append({
            'type': f'daily_gap_{gap_type}',
            'level': gap_level,
            'distance': abs(gap_level - open_price),
            'is_swing': False,
            'gap_size': abs(gap_level2 - gap_level1)
        })
    
    if len(levels) < 2:
        return None  # Need at least 2 levels for confluence
    
    # Find confluence zones for each threshold
    results = []
    for threshold in CONFLUENCE_THRESHOLDS:
        confluence_zones = find_confluence_zones(levels, open_price, threshold)
        
        for zone in confluence_zones:
            zone_result = {
                'threshold': threshold,
                'zone_center': zone['zone_center'],
                'distance_from_open': zone['distance_from_open'],
                'confluence_strength': zone['confluence_strength'],
                'level_types': '|'.join(zone['level_types']),
                'max_level': zone['max_level'],
                'min_level': zone['min_level'],
                'zone_width': zone['zone_width'],
                'open_price': open_price
            }
            
            # Check hits for this confluence zone
            for window_start, window_end, window_name in WINDOWS:
                window_data = g.loc[within(g['ts_et'], window_start, window_end)]
                if window_data.empty:
                    zone_result[f'{window_name}_hit'] = False
                    continue
                
                # Check if any level in the zone was hit
                hit = False
                for level in zone['levels']:
                    level_hit = False
                    if 'high' in level['type'] or 'up' in level['type']:
                        level_hit = (window_data['high'] >= level['level'] - 1e-12 + EPSILON).any()
                    else:
                        level_hit = (window_data['low'] <= level['level'] + 1e-12 - EPSILON).any()
                    
                    if level_hit:
                        hit = True
                        break
                
                zone_result[f'{window_name}_hit'] = hit
            
            results.append(zone_result)
    
    return results

def main():
    tz_et = pytz.timezone("US/Eastern")
    all_results = []
    
    residual = pd.DataFrame()
    
    print("Processing data for confluence analysis...")
    
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
                result['date'] = d
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
                result['date'] = d
                all_results.append(result)

    if not all_results:
        print("No confluence data found!")
        return

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Calculate confluence statistics
    summary_stats = []
    
    for threshold in CONFLUENCE_THRESHOLDS:
        threshold_data = df_results[df_results['threshold'] == threshold]
        if threshold_data.empty:
            continue
        
        for confluence_strength in sorted(threshold_data['confluence_strength'].unique()):
            strength_data = threshold_data[threshold_data['confluence_strength'] == confluence_strength]
            
            stats = {
                'threshold': threshold,
                'confluence_strength': confluence_strength,
                'total_zones': len(strength_data),
                'avg_distance_from_open': strength_data['distance_from_open'].mean(),
                'avg_zone_width': strength_data['zone_width'].mean()
            }
            
            # Calculate hit rates for each window
            for window_start, window_end, window_name in WINDOWS:
                hit_col = f'{window_name}_hit'
                if hit_col in strength_data.columns:
                    hits = strength_data[hit_col].sum()
                    hit_rate = hits / len(strength_data) * 100
                    stats[f'{window_name}_hits'] = hits
                    stats[f'{window_name}_hit_rate'] = round(hit_rate, 2)
            
            # Analyze level type combinations
            level_combinations = strength_data['level_types'].value_counts()
            stats['top_combinations'] = '|'.join(level_combinations.head(3).index.tolist())
            
            summary_stats.append(stats)
    
    # Save detailed results
    df_results.to_csv(DETAILED_OUT_FILE, index=False)
    print(f"✅ Wrote detailed confluence data to {DETAILED_OUT_FILE}")
    
    # Save summary statistics
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(OUT_FILE, index=False)
    print(f"✅ Wrote confluence summary to {OUT_FILE}")
    
    # Print key insights
    print("\n" + "="*80)
    print("CONFLUENCE ANALYSIS - Key Insights")
    print("="*80)
    
    for threshold in CONFLUENCE_THRESHOLDS:
        threshold_data = df_summary[df_summary['threshold'] == threshold]
        if threshold_data.empty:
            continue
            
        print(f"\n{threshold} POINT CONFLUENCE ZONES:")
        for _, row in threshold_data.iterrows():
            if '9:30-10:15_hit_rate' in row:
                print(f"  {row['confluence_strength']} levels: {row['9:30-10:15_hit_rate']}% hit rate ({row['total_zones']} zones)")
                if 'top_combinations' in row and pd.notna(row['top_combinations']):
                    print(f"    Top combinations: {row['top_combinations']}")
    
    print(f"\nTotal confluence zones analyzed: {len(df_results)}")
    print(f"Summary saved to: {OUT_FILE}")
    print(f"Detailed data saved to: {DETAILED_OUT_FILE}")

if __name__ == "__main__":
    main()
