#!/usr/bin/env python3
"""
HTF Gap Context Analysis - First 45 Minutes
-------------------------------------------

Analyzes how Higher Timeframe (HTF) gap contexts at 9:30 AM correlate with:
- First 45 minutes behavior (direction, range, volatility)
- 15-minute candle patterns
- Gap fill probabilities
- Opening range behavior

HTF Gaps Analyzed:
- 1-hour gaps
- 4-hour gaps  
- Daily gaps (overnight gaps)
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

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20250926.ohlcv-1m.csv")

RTH_START, RTH_END = time(9, 30), time(16, 0)
NY_OPEN = time(9, 30)
FIRST_45_END = time(10, 15)
CHUNKSIZE = 1_000_000

# Gap timeframes
GAP_TIMEFRAMES = {
    '1h': 60,
    '4h': 240,
    'daily': None  # Special handling for daily gaps
}


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


def find_htf_gaps(df: pd.DataFrame, timeframe: str, current_date) -> List[Dict]:
    """
    Find HTF gaps active at 9:30 AM.
    
    Args:
        df: 1-minute OHLCV data
        timeframe: '1h', '4h', or 'daily'
        current_date: Current trading date
    
    Returns:
        List of gap dictionaries with gap info
    """
    gaps = []
    df_sorted = df.sort_values('ts_et').reset_index(drop=True)
    
    ny_open_ts = pd.Timestamp.combine(current_date, NY_OPEN).tz_localize(df['ts_et'].dt.tz)
    
    if timeframe == 'daily':
        # Daily gap: gap between previous day close and current day open
        prev_date = current_date - timedelta(days=1)
        prev_data = df_sorted[df_sorted['ts_et'].dt.date == prev_date]
        current_data = df_sorted[df_sorted['ts_et'].dt.date == current_date]
        
        if prev_data.empty or current_data.empty:
            return []
        
        prev_close = prev_data.iloc[-1]['close']
        current_open = current_data.iloc[0]['open']
        
        if prev_close < current_open:  # Upward gap
            gap_size = current_open - prev_close
            gaps.append({
                'timeframe': timeframe,
                'direction': 'up',
                'gap_high': prev_close,
                'gap_low': current_open,
                'gap_size': gap_size,
                'gap_level': prev_close,  # Level to fill
                'gap_start': prev_data.iloc[-1]['ts_et'],
                'gap_end': current_data.iloc[0]['ts_et'],
                'created_before_open': True,
            })
        elif prev_close > current_open:  # Downward gap
            gap_size = prev_close - current_open
            gaps.append({
                'timeframe': timeframe,
                'direction': 'down',
                'gap_high': prev_close,
                'gap_low': current_open,
                'gap_size': gap_size,
                'gap_level': prev_close,  # Level to fill
                'gap_start': prev_data.iloc[-1]['ts_et'],
                'gap_end': current_data.iloc[0]['ts_et'],
                'created_before_open': True,
            })
    else:
        # Time-based gaps (1h, 4h)
        gap_minutes = GAP_TIMEFRAMES[timeframe]
        
        for i in range(len(df_sorted) - 1):
            current = df_sorted.iloc[i]
            next_bar = df_sorted.iloc[i + 1]
            
            # Check if gap is at least gap_minutes
            time_diff = (next_bar['ts_et'] - current['ts_et']).total_seconds() / 60
            if time_diff >= gap_minutes:
                gap_high = current['high']
                gap_low = next_bar['low']
                
                if gap_high < gap_low:  # Upward gap
                    gap_size = gap_low - gap_high
                    gap_level = gap_high  # Level to fill
                    
                    # Check if gap is unmitigated (not filled before NY open)
                    gap_filled = False
                    after_gap = df_sorted[df_sorted['ts_et'] > next_bar['ts_et']]
                    before_open = after_gap[after_gap['ts_et'] < ny_open_ts]
                    if not before_open.empty and (before_open['low'] <= gap_level).any():
                        gap_filled = True
                    
                    if not gap_filled:
                        gaps.append({
                            'timeframe': timeframe,
                            'direction': 'up',
                            'gap_high': gap_high,
                            'gap_low': gap_low,
                            'gap_size': gap_size,
                            'gap_level': gap_level,
                            'gap_start': current['ts_et'],
                            'gap_end': next_bar['ts_et'],
                            'created_before_open': current['ts_et'] < ny_open_ts,
                        })
                
                elif gap_low > gap_high:  # Downward gap
                    gap_size = gap_high - gap_low
                    gap_level = gap_low  # Level to fill
                    
                    # Check if gap is unmitigated
                    gap_filled = False
                    after_gap = df_sorted[df_sorted['ts_et'] > next_bar['ts_et']]
                    before_open = after_gap[after_gap['ts_et'] < ny_open_ts]
                    if not before_open.empty and (before_open['high'] >= gap_level).any():
                        gap_filled = True
                    
                    if not gap_filled:
                        gaps.append({
                            'timeframe': timeframe,
                            'direction': 'down',
                            'gap_high': gap_high,
                            'gap_low': gap_low,
                            'gap_size': gap_size,
                            'gap_level': gap_level,
                            'gap_start': current['ts_et'],
                            'gap_end': next_bar['ts_et'],
                            'created_before_open': current['ts_et'] < ny_open_ts,
                        })
    
    return gaps


def classify_price_position(open_price: float, gaps: List[Dict]) -> Dict:
    """
    Classify price position relative to HTF gaps.
    
    Returns:
        Dictionary with position classifications for each timeframe
    """
    position = {}
    
    for timeframe in ['1h', '4h', 'daily']:
        tf_gaps = [g for g in gaps if g['timeframe'] == timeframe]
        
        if not tf_gaps:
            position[f'{timeframe}_position'] = 'no_gap'
            position[f'{timeframe}_gap_count'] = 0
            continue
        
        position[f'{timeframe}_gap_count'] = len(tf_gaps)
        
        # Find closest gap above and below
        gaps_above = [g for g in tf_gaps if g['gap_low'] > open_price]
        gaps_below = [g for g in tf_gaps if g['gap_high'] < open_price]
        gaps_within = [g for g in tf_gaps if g['gap_high'] >= open_price >= g['gap_low']]
        
        if gaps_within:
            # Price is within a gap
            gap = gaps_within[0]  # Use first gap if multiple
            gap_midpoint = (gap['gap_high'] + gap['gap_low']) / 2
            dist_to_top = gap['gap_high'] - open_price
            dist_to_bottom = open_price - gap['gap_low']
            
            if dist_to_top < dist_to_bottom:
                position[f'{timeframe}_position'] = 'within_gap_near_top'
            elif dist_to_bottom < dist_to_top:
                position[f'{timeframe}_position'] = 'within_gap_near_bottom'
            else:
                position[f'{timeframe}_position'] = 'within_gap_middle'
            
            position[f'{timeframe}_gap_direction'] = gap['direction']
            position[f'{timeframe}_gap_size'] = gap['gap_size']
            position[f'{timeframe}_dist_to_gap_top'] = dist_to_top
            position[f'{timeframe}_dist_to_gap_bottom'] = dist_to_bottom
            position[f'{timeframe}_dist_to_gap_mid'] = abs(open_price - gap_midpoint)
            
        elif gaps_above and gaps_below:
            # Price is between gaps
            closest_above = min(gaps_above, key=lambda g: g['gap_low'] - open_price)
            closest_below = max(gaps_below, key=lambda g: open_price - g['gap_high'])
            
            dist_to_above = closest_above['gap_low'] - open_price
            dist_to_below = open_price - closest_below['gap_high']
            
            if dist_to_above < dist_to_below:
                position[f'{timeframe}_position'] = 'between_gaps_closer_to_above'
            else:
                position[f'{timeframe}_position'] = 'between_gaps_closer_to_below'
            
            position[f'{timeframe}_gap_direction'] = 'mixed'
            position[f'{timeframe}_gap_size'] = None
            position[f'{timeframe}_dist_to_gap_top'] = None
            position[f'{timeframe}_dist_to_gap_bottom'] = None
            position[f'{timeframe}_dist_to_gap_mid'] = None
            
        elif gaps_above:
            # Price is below gap (delivering off gap low)
            closest_gap = min(gaps_above, key=lambda g: g['gap_low'] - open_price)
            position[f'{timeframe}_position'] = 'below_gap'
            position[f'{timeframe}_gap_direction'] = closest_gap['direction']
            position[f'{timeframe}_gap_size'] = closest_gap['gap_size']
            position[f'{timeframe}_dist_to_gap_top'] = None
            position[f'{timeframe}_dist_to_gap_bottom'] = closest_gap['gap_low'] - open_price
            position[f'{timeframe}_dist_to_gap_mid'] = None
            
        elif gaps_below:
            # Price is above gap (delivering off gap high)
            closest_gap = max(gaps_below, key=lambda g: open_price - g['gap_high'])
            position[f'{timeframe}_position'] = 'above_gap'
            position[f'{timeframe}_gap_direction'] = closest_gap['direction']
            position[f'{timeframe}_gap_size'] = closest_gap['gap_size']
            position[f'{timeframe}_dist_to_gap_top'] = open_price - closest_gap['gap_high']
            position[f'{timeframe}_dist_to_gap_bottom'] = None
            position[f'{timeframe}_dist_to_gap_mid'] = None
    
    # Multi-HTf context
    has_1h = position.get('1h_position') not in ['no_gap', None]
    has_4h = position.get('4h_position') not in ['no_gap', None]
    has_daily = position.get('daily_position') not in ['no_gap', None]
    
    if has_1h and has_4h and has_daily:
        position['multi_htf_context'] = 'all_three'
    elif has_1h and has_4h:
        position['multi_htf_context'] = '1h_4h'
    elif has_1h and has_daily:
        position['multi_htf_context'] = '1h_daily'
    elif has_4h and has_daily:
        position['multi_htf_context'] = '4h_daily'
    elif has_1h:
        position['multi_htf_context'] = '1h_only'
    elif has_4h:
        position['multi_htf_context'] = '4h_only'
    elif has_daily:
        position['multi_htf_context'] = 'daily_only'
    else:
        position['multi_htf_context'] = 'no_gaps'
    
    # Check if gaps are stacked (one contained within another)
    if has_1h and has_4h:
        gap_1h = next((g for g in gaps if g['timeframe'] == '1h'), None)
        gap_4h = next((g for g in gaps if g['timeframe'] == '4h'), None)
        if gap_1h and gap_4h:
            if gap_1h['gap_high'] >= gap_4h['gap_high'] and gap_1h['gap_low'] <= gap_4h['gap_low']:
                position['gap_stacking'] = '1h_within_4h'
            elif gap_4h['gap_high'] >= gap_1h['gap_high'] and gap_4h['gap_low'] <= gap_1h['gap_low']:
                position['gap_stacking'] = '4h_within_1h'
            else:
                position['gap_stacking'] = 'overlapping'
    
    return position


def aggregate_15m_candles(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars into 15-minute candles."""
    df_1m = df_1m.sort_values("ts_et").reset_index(drop=True)
    df_1m["candle_time"] = df_1m["ts_et"].dt.floor("15min")
    
    agg = df_1m.groupby("candle_time").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).reset_index()
    
    agg["candle_range"] = agg["high"] - agg["low"]
    agg["body_size"] = np.abs(agg["close"] - agg["open"])
    agg["direction"] = np.where(agg["close"] > agg["open"], "bullish", "bearish")
    agg["time_slot"] = agg["candle_time"].dt.time
    
    return agg


def analyze_first_45_min(df_1m: pd.DataFrame, df_15m: pd.DataFrame, open_price: float, gaps: List[Dict]) -> Dict:
    """Analyze first 45 minutes behavior."""
    first_45_mask = (df_1m["ts_et"].dt.time >= NY_OPEN) & (df_1m["ts_et"].dt.time < FIRST_45_END)
    first_45_data = df_1m[first_45_mask]
    
    if first_45_data.empty:
        return {}
    
    first_45_high = first_45_data["high"].max()
    first_45_low = first_45_data["low"].min()
    first_45_range = first_45_high - first_45_low
    first_45_close = first_45_data.iloc[-1]["close"]
    first_45_net_move = first_45_close - open_price
    first_45_direction = "up" if first_45_net_move > 0 else "down"
    
    # Get first 3 candles
    first_3_candles = df_15m[df_15m["time_slot"].isin([time(9, 30), time(9, 45), time(10, 0)])].head(3)
    
    results = {
        'first_45_high': first_45_high,
        'first_45_low': first_45_low,
        'first_45_range': first_45_range,
        'first_45_close': first_45_close,
        'first_45_net_move': first_45_net_move,
        'first_45_direction': first_45_direction,
        'first_45_volume': first_45_data["volume"].sum(),
    }
    
    # Candle patterns
    if len(first_3_candles) >= 3:
        c1_dir = first_3_candles.iloc[0]["direction"]
        c2_dir = first_3_candles.iloc[1]["direction"]
        c3_dir = first_3_candles.iloc[2]["direction"]
        
        results['c1_direction'] = c1_dir
        results['c2_direction'] = c2_dir
        results['c3_direction'] = c3_dir
        results['candle_sequence'] = f"{c1_dir}-{c2_dir}-{c3_dir}"
        results['all_same_direction'] = (c1_dir == c2_dir == c3_dir)
        results['reversal_after_c1'] = (c2_dir != c1_dir)
        results['reversal_after_c2'] = (c3_dir != c2_dir)
        
        results['c1_size'] = first_3_candles.iloc[0]["candle_range"]
        results['c2_size'] = first_3_candles.iloc[1]["candle_range"]
        results['c3_size'] = first_3_candles.iloc[2]["candle_range"]
        results['opening_range_high'] = first_3_candles["high"].max()
        results['opening_range_low'] = first_3_candles["low"].min()
        results['opening_range_size'] = results['opening_range_high'] - results['opening_range_low']
    
    # Gap fill analysis
    for gap in gaps:
        tf = gap['timeframe']
        gap_level = gap['gap_level']
        
        # Check if gap was filled in first 45 min
        if gap['direction'] == 'up':
            filled = (first_45_low <= gap_level)
        else:
            filled = (first_45_high >= gap_level)
        
        results[f'{tf}_gap_filled_first_45'] = filled
        
        if filled:
            # Time to fill
            if gap['direction'] == 'up':
                fill_mask = first_45_data["low"] <= gap_level
            else:
                fill_mask = first_45_data["high"] >= gap_level
            
            if fill_mask.any():
                fill_time = first_45_data[fill_mask].iloc[0]["ts_et"]
                minutes_to_fill = (fill_time - pd.Timestamp.combine(
                    df_1m.iloc[0]["ts_et"].date(), NY_OPEN
                ).tz_localize(df_1m.iloc[0]["ts_et"].tz)).total_seconds() / 60
                results[f'{tf}_gap_minutes_to_fill'] = minutes_to_fill
    
    return results


def process_day(g: pd.DataFrame) -> Optional[Dict]:
    """Process a single trading day."""
    if g.empty:
        return None
    
    # Get 9:30 opening price
    open_930 = g.loc[_within(g['ts_et'], NY_OPEN, time(9, 31))]
    if open_930.empty:
        return None
    
    open_price = open_930.iloc[0]['open']
    current_date = g['ts_et'].dt.date.iloc[0]
    
    # Find HTF gaps
    all_gaps = []
    for timeframe in ['1h', '4h', 'daily']:
        gaps = find_htf_gaps(g, timeframe, current_date)
        all_gaps.extend(gaps)
    
    # Classify price position
    position = classify_price_position(open_price, all_gaps)
    
    # Aggregate 15-minute candles
    df_15m = aggregate_15m_candles(g)
    
    # Analyze first 45 minutes
    first_45_analysis = analyze_first_45_min(g, df_15m, open_price, all_gaps)
    
    # Get session extremes
    rth_mask = _within(g['ts_et'], RTH_START, RTH_END)
    rth_data = g[rth_mask]
    session_high = rth_data["high"].max()
    session_low = rth_data["low"].min()
    daily_close = rth_data.iloc[-1]["close"]
    daily_bias = "bullish" if daily_close > open_price else "bearish"
    
    # Combine results
    result = {
        'date': current_date,
        'open_price': open_price,
        'session_high': session_high,
        'session_low': session_low,
        'daily_close': daily_close,
        'daily_bias': daily_bias,
        **position,
        **first_45_analysis,
    }
    
    # Add gap details
    for gap in all_gaps:
        tf = gap['timeframe']
        result[f'{tf}_gap_exists'] = True
        result[f'{tf}_gap_direction'] = gap['direction']
        result[f'{tf}_gap_size'] = gap['gap_size']
        result[f'{tf}_gap_level'] = gap['gap_level']
    
    return result


def main():
    """Main analysis function."""
    print("HTF Gap Context Analysis - First 45 Minutes")
    print("=" * 60)
    
    all_results = []
    
    # Read data in chunks
    print(f"Reading data from {CSV_FILE}...")
    tz_et = pytz.timezone("US/Eastern")
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
        
        # Filter to NQ front month
        df = df[df["symbol"].astype(str).str.startswith("NQ") & ~df["symbol"].astype(str).str.contains("-")]
        
        # Split off final day
        max_date = df["et_date"].max()
        to_proc = df[df["et_date"] < max_date]
        residual = df[df["et_date"] == max_date]
        
        # Filter to front-month
        if not to_proc.empty:
            to_proc = _determine_front_month(to_proc)
        
        chunk = to_proc
        if chunk.empty:
            continue
        
        # Process by day
        dates = sorted(chunk["et_date"].unique())
        for date in tqdm(dates, desc=f"Chunk {chunk_num + 1} days"):
            day_data = chunk[chunk["et_date"] == date].copy()
            day_data = day_data.sort_values("ts_et").reset_index(drop=True)
            
            result = process_day(day_data)
            if result:
                all_results.append(result)
    
    # Process residual (last day)
    if not residual.empty:
        residual = residual.sort_values("ts_et").reset_index(drop=True)
        result = process_day(residual)
        if result:
            all_results.append(result)
    
    print(f"\nProcessed {len(all_results)} trading days")
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    
    # Position distribution
    position_cols = ['1h_position', '4h_position', 'daily_position', 'multi_htf_context']
    for col in position_cols:
        if col in df_results.columns:
            counts = df_results[col].value_counts()
            print(f"\n{col}:")
            for pos, count in counts.items():
                pct = count / len(df_results) * 100
                print(f"  {pos}: {count} ({pct:.1f}%)")
    
    # Save detailed results
    df_results.to_csv(OUT_DIR / "htf_gap_context_detailed.csv", index=False)
    print(f"\n✅ Saved detailed results to htf_gap_context_detailed.csv ({len(df_results)} rows)")
    
    # Generate aggregate statistics by position
    print("\nGenerating aggregate statistics by position...")
    
    aggregate_stats = []
    
    for col in position_cols:
        if col not in df_results.columns:
            continue
        
        for position in df_results[col].dropna().unique():
            mask = df_results[col] == position
            subset = df_results[mask]
            
            if len(subset) < 10:  # Skip small samples
                continue
            
            stats = {
                'position_type': col,
                'position_value': position,
                'count': len(subset),
                'pct_of_total': len(subset) / len(df_results) * 100,
            }
            
            # First 45 min stats
            if 'first_45_direction' in subset.columns:
                up_pct = (subset['first_45_direction'] == 'up').mean() * 100
                stats['pct_first_45_up'] = up_pct
                stats['pct_first_45_down'] = 100 - up_pct
            
            if 'first_45_range' in subset.columns:
                stats['mean_first_45_range'] = subset['first_45_range'].mean()
                stats['median_first_45_range'] = subset['first_45_range'].median()
            
            if 'first_45_net_move' in subset.columns:
                stats['mean_first_45_net_move'] = subset['first_45_net_move'].mean()
                stats['median_first_45_net_move'] = subset['first_45_net_move'].median()
            
            # Gap fill probabilities
            for tf in ['1h', '4h', 'daily']:
                fill_col = f'{tf}_gap_filled_first_45'
                if fill_col in subset.columns:
                    fill_pct = subset[fill_col].mean() * 100
                    stats[f'{tf}_gap_fill_pct'] = fill_pct
            
            # Candle patterns
            if 'all_same_direction' in subset.columns:
                stats['pct_all_same_direction'] = subset['all_same_direction'].mean() * 100
            
            if 'reversal_after_c1' in subset.columns:
                stats['pct_reversal_after_c1'] = subset['reversal_after_c1'].mean() * 100
            
            # Daily bias alignment
            if 'daily_bias' in subset.columns and 'first_45_direction' in subset.columns:
                aligned = (subset['daily_bias'] == subset['first_45_direction'])
                stats['pct_aligned_with_daily'] = aligned.mean() * 100
            
            aggregate_stats.append(stats)
    
    df_aggregate = pd.DataFrame(aggregate_stats)
    df_aggregate.to_csv(OUT_DIR / "htf_gap_context_summary.csv", index=False)
    print(f"✅ Saved aggregate statistics to htf_gap_context_summary.csv ({len(df_aggregate)} rows)")
    
    # High-probability patterns (70%+)
    print("\nIdentifying high-probability patterns (70%+)...")
    
    high_prob_patterns = []
    
    for _, row in df_aggregate.iterrows():
        pattern = {}
        
        # Check for 70%+ directional bias
        if 'pct_first_45_up' in row and row['pct_first_45_up'] >= 70:
            pattern['type'] = 'directional_bias_up'
            pattern['probability'] = row['pct_first_45_up']
            pattern['position'] = row['position_value']
            pattern['count'] = row['count']
            high_prob_patterns.append(pattern.copy())
        
        if 'pct_first_45_down' in row and row['pct_first_45_down'] >= 70:
            pattern['type'] = 'directional_bias_down'
            pattern['probability'] = row['pct_first_45_down']
            pattern['position'] = row['position_value']
            pattern['count'] = row['count']
            high_prob_patterns.append(pattern.copy())
        
        # Check for 70%+ gap fill probability
        for tf in ['1h', '4h', 'daily']:
            fill_col = f'{tf}_gap_fill_pct'
            if fill_col in row and pd.notna(row[fill_col]) and row[fill_col] >= 70:
                pattern['type'] = f'{tf}_gap_fill'
                pattern['probability'] = row[fill_col]
                pattern['position'] = row['position_value']
                pattern['count'] = row['count']
                high_prob_patterns.append(pattern.copy())
        
        # Check for 70%+ alignment with daily
        if 'pct_aligned_with_daily' in row and row['pct_aligned_with_daily'] >= 70:
            pattern['type'] = 'aligned_with_daily'
            pattern['probability'] = row['pct_aligned_with_daily']
            pattern['position'] = row['position_value']
            pattern['count'] = row['count']
            high_prob_patterns.append(pattern.copy())
    
    if high_prob_patterns:
        df_high_prob = pd.DataFrame(high_prob_patterns)
        df_high_prob = df_high_prob.sort_values('probability', ascending=False)
        df_high_prob.to_csv(OUT_DIR / "htf_gap_context_high_prob_patterns.csv", index=False)
        print(f"✅ Saved high-probability patterns to htf_gap_context_high_prob_patterns.csv ({len(df_high_prob)} patterns)")
        
        print("\nTop high-probability patterns:")
        for _, row in df_high_prob.head(20).iterrows():
            print(f"  {row['type']}: {row['probability']:.1f}% ({row['position']}, n={row['count']})")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

