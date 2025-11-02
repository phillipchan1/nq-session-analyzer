#!/usr/bin/env python3
"""
15-Minute Candle Behavior Analysis
-----------------------------------

Analyzes 15-minute candle patterns in NQ RTH sessions, covering:
- Candle structure & volatility
- Candle sequence behavior
- Liquidity & high/low behavior
- Session phase comparison
- Bias & context integration
- Reversion & mean behavior
- Microstructure & volume
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
EVENTS_FILE = str(DATA_DIR / "us_high_impact_events_2020_to_2025.csv")

RTH_START, RTH_END = time(9, 30), time(16, 0)
CHUNKSIZE = 1_000_000

# Session blocks
SESSION_BLOCKS = {
    "opening_drive": (time(9, 30), time(10, 0)),
    "mid_morning": (time(10, 0), time(11, 30)),
    "midday": (time(11, 30), time(13, 0)),
    "afternoon": (time(13, 0), time(16, 0)),
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


def aggregate_15m_candles(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars into 15-minute candles."""
    df_1m = df_1m.sort_values("ts_et").reset_index(drop=True)
    
    # Create 15-minute time buckets
    df_1m["candle_time"] = df_1m["ts_et"].dt.floor("15min")
    
    # Aggregate OHLCV
    agg = df_1m.groupby("candle_time").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).reset_index()
    
    # Calculate metrics
    agg["candle_range"] = agg["high"] - agg["low"]
    agg["body_size"] = np.abs(agg["close"] - agg["open"])
    agg["wick_upper"] = agg["high"] - agg[["open", "close"]].max(axis=1)
    agg["wick_lower"] = agg[["open", "close"]].min(axis=1) - agg["low"]
    agg["direction"] = np.where(agg["close"] > agg["open"], "bullish", "bearish")
    
    # Wick-to-body ratio (handle division by zero)
    agg["wick_to_body_ratio"] = np.where(
        agg["body_size"] > 0,
        agg["candle_range"] / agg["body_size"],
        np.inf
    )
    
    # Tag with session block
    agg["time_slot"] = agg["candle_time"].dt.time
    agg["session_block"] = agg["time_slot"].apply(_get_session_block)
    
    return agg


def _get_session_block(t: time) -> str:
    """Determine which session block a time falls into."""
    if RTH_START <= t < time(10, 0):
        return "opening_drive"
    elif time(10, 0) <= t < time(11, 30):
        return "mid_morning"
    elif time(11, 30) <= t < time(13, 0):
        return "midday"
    elif time(13, 0) <= t < RTH_END:
        return "afternoon"
    return "unknown"


def calculate_vwap(df_1m: pd.DataFrame) -> float:
    """Calculate VWAP for the day."""
    if df_1m.empty:
        return np.nan
    typical_price = (df_1m["high"] + df_1m["low"] + df_1m["close"]) / 3
    return (typical_price * df_1m["volume"]).sum() / df_1m["volume"].sum()


def detect_sequences(df_15m: pd.DataFrame) -> Dict:
    """Detect candle sequences: streaks, outside bars, etc."""
    results = {}
    
    if len(df_15m) < 2:
        return results
    
    # Streaks
    df_15m["streak"] = (df_15m["direction"] != df_15m["direction"].shift()).cumsum()
    streak_lengths = df_15m.groupby("streak").size()
    results["streaks_3plus"] = len(streak_lengths[streak_lengths >= 3])
    
    # Outside bars
    df_15m["outside_bar"] = (
        (df_15m["high"] > df_15m["high"].shift(1)) &
        (df_15m["low"] < df_15m["low"].shift(1))
    )
    results["outside_bars"] = df_15m["outside_bar"].sum()
    
    # Large wicks (indecision)
    df_15m["large_wicks_both"] = (
        (df_15m["wick_upper"] > df_15m["body_size"]) &
        (df_15m["wick_lower"] > df_15m["body_size"])
    )
    results["indecision_candles"] = df_15m["large_wicks_both"].sum()
    
    return results


def detect_liquidity_sweeps(df_15m: pd.DataFrame) -> List[Dict]:
    """Detect candles that sweep prior candle's high/low and close opposite."""
    sweeps = []
    
    for i in range(1, len(df_15m)):
        prev = df_15m.iloc[i-1]
        curr = df_15m.iloc[i]
        
        # Sweep high and close bearish
        if (curr["high"] > prev["high"]) and (curr["close"] < prev["close"]):
            sweeps.append({
                "candle_idx": i,
                "sweep_type": "high",
                "sweep_size": curr["high"] - prev["high"],
                "reversal_size": prev["close"] - curr["close"],
            })
        
        # Sweep low and close bullish
        if (curr["low"] < prev["low"]) and (curr["close"] > prev["close"]):
            sweeps.append({
                "candle_idx": i,
                "sweep_type": "low",
                "sweep_size": prev["low"] - curr["low"],
                "reversal_size": curr["close"] - prev["close"],
            })
    
    return sweeps


def detect_fvg(df_15m: pd.DataFrame) -> List[Dict]:
    """Detect Fair Value Gaps (FVGs) - gaps between candles that don't fill."""
    fvgs = []
    
    if len(df_15m) < 2:
        return fvgs
    
    for i in range(1, len(df_15m)):
        prev = df_15m.iloc[i-1]
        curr = df_15m.iloc[i]
        
        # Bullish FVG: gap up
        if curr["low"] > prev["high"]:
            gap_size = curr["low"] - prev["high"]
            # Check if gap fills in next 2 candles
            filled = False
            for j in range(i+1, min(i+3, len(df_15m))):
                if df_15m.iloc[j]["low"] <= prev["high"]:
                    filled = True
                    break
            
            if not filled:
                fvgs.append({
                    "candle_idx": i,
                    "fvg_type": "bullish",
                    "gap_size": gap_size,
                    "time_slot": curr.get("time_slot", curr.get("candle_time", None)),
                })
        
        # Bearish FVG: gap down
        elif curr["high"] < prev["low"]:
            gap_size = prev["low"] - curr["high"]
            filled = False
            for j in range(i+1, min(i+3, len(df_15m))):
                if df_15m.iloc[j]["high"] >= prev["low"]:
                    filled = True
                    break
            
            if not filled:
                fvgs.append({
                    "candle_idx": i,
                    "fvg_type": "bearish",
                    "gap_size": gap_size,
                    "time_slot": curr.get("time_slot", curr.get("candle_time", None)),
                })
    
    return fvgs


def calculate_volume_delta(df_1m: pd.DataFrame, df_15m: pd.DataFrame) -> pd.DataFrame:
    """Approximate buy/sell volume using tick rule."""
    df_1m = df_1m.sort_values("ts_et").reset_index(drop=True)
    df_1m["candle_time"] = df_1m["ts_et"].dt.floor("15min")
    
    # Tick rule: if close > prev close, buy volume; else sell volume
    df_1m["tick_dir"] = np.where(df_1m["close"] > df_1m["close"].shift(1), 1, -1)
    df_1m.loc[df_1m.index[0], "tick_dir"] = 0  # First bar neutral
    
    df_1m["buy_volume"] = np.where(df_1m["tick_dir"] == 1, df_1m["volume"], 0)
    df_1m["sell_volume"] = np.where(df_1m["tick_dir"] == -1, df_1m["volume"], 0)
    
    vol_agg = df_1m.groupby("candle_time").agg({
        "volume": "sum",
        "buy_volume": "sum",
        "sell_volume": "sum",
    }).reset_index()
    
    vol_agg["volume_delta"] = vol_agg["buy_volume"] - vol_agg["sell_volume"]
    vol_agg["volume_imbalance"] = vol_agg["volume_delta"] / vol_agg["volume"].replace(0, np.nan)
    
    # Merge with 15m candles
    df_15m = df_15m.merge(vol_agg[["candle_time", "volume_delta", "volume_imbalance"]], 
                          on="candle_time", how="left")
    
    return df_15m


def calculate_daily_sma(daily_closes: Dict, current_date, periods: List[int]) -> Dict[int, float]:
    """Calculate daily SMAs from accumulated daily closes."""
    smas = {}
    
    # Sort dates
    sorted_dates = sorted([d for d in daily_closes.keys() if d <= current_date])
    
    for p in periods:
        if len(sorted_dates) >= p:
            recent_dates = sorted_dates[-p:]
            recent_closes = [daily_closes[d] for d in recent_dates]
            smas[p] = np.mean(recent_closes)
        else:
            smas[p] = np.nan
    
    return smas


def process_day(symbol: str, date, df_1m: pd.DataFrame, daily_closes: Dict) -> Dict:
    """Process a single day's data."""
    # Filter to RTH
    rth_mask = _within(df_1m["ts_et"], RTH_START, RTH_END)
    rth_1m = df_1m.loc[rth_mask].copy()
    
    if rth_1m.empty:
        return None
    
    # Aggregate to 15-minute candles
    df_15m = aggregate_15m_candles(rth_1m)
    
    if len(df_15m) < 2:
        return None
    
    # Calculate VWAP
    vwap = calculate_vwap(rth_1m)
    
    # Calculate daily SMAs from accumulated daily closes
    smas = calculate_daily_sma(daily_closes, date, [50, 200])
    daily_bias = "bullish" if (not pd.isna(smas[50]) and not pd.isna(smas[200]) and smas[50] > smas[200]) else "bearish"
    
    # Calculate volume delta (try/except in case of issues)
    try:
        df_15m = calculate_volume_delta(rth_1m, df_15m)
    except Exception as e:
        print(f"Warning: Volume delta calculation failed for {date}: {e}")
        df_15m["volume_delta"] = np.nan
        df_15m["volume_imbalance"] = np.nan
    
    # Detect sequences
    sequences = detect_sequences(df_15m)
    
    # Detect liquidity sweeps
    sweeps = detect_liquidity_sweeps(df_15m)
    
    # Detect FVGs
    fvgs = detect_fvg(df_15m)
    
    # Session high/low
    session_high = rth_1m["high"].max()
    session_low = rth_1m["low"].min()
    session_high_idx = rth_1m["high"].idxmax()
    session_low_idx = rth_1m["low"].idxmin()
    session_high_time = rth_1m.loc[session_high_idx, "ts_et"].time()
    session_low_time = rth_1m.loc[session_low_idx, "ts_et"].time()
    
    # First candle info
    first_candle = df_15m.iloc[0]
    first_candle_time = first_candle["time_slot"]
    
    # Daily close
    daily_close = rth_1m.iloc[-1]["close"]
    
    # Overnight range (from previous day close to 9:30 open)
    prev_date = date - timedelta(days=1)
    if prev_date in daily_closes:
        prev_close_price = daily_closes[prev_date]
        overnight_range = first_candle["open"] - prev_close_price
    else:
        prev_close_price = np.nan
        overnight_range = np.nan
    
    return {
        "symbol": symbol,
        "date": date,
        "df_15m": df_15m,
        "vwap": vwap,
        "sma50": smas[50],
        "sma200": smas[200],
        "daily_bias": daily_bias,
        "sequences": sequences,
        "sweeps": sweeps,
        "fvgs": fvgs,
        "session_high": session_high,
        "session_low": session_low,
        "session_high_time": session_high_time,
        "session_low_time": session_low_time,
        "first_candle": first_candle,
        "daily_close": daily_close,
        "overnight_range": overnight_range,
        "prev_close": prev_close_price,
    }


def main():
    """Main processing function."""
    print("Starting 15-minute candle behavior analysis...")
    
    tz_et = pytz.timezone("US/Eastern")
    residual = pd.DataFrame()
    
    # Storage for all days
    all_days_data = []
    all_candles = []
    daily_closes = {}  # Store daily closes for SMA calculation
    
    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    dtypes = {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "symbol": "string"
    }
    
    print("Loading and processing data...")
    
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
        for (sym, d), g in tqdm(to_proc.groupby(["symbol", "et_date"]), desc=f"Chunk {chunk_num + 1} days"):
            g = g.sort_values("ts_et").reset_index(drop=True)
            
            # Store daily close for SMA calculation
            rth_mask = _within(g["ts_et"], RTH_START, RTH_END)
            rth = g.loc[rth_mask]
            if not rth.empty:
                daily_closes[d] = rth.iloc[-1]["close"]
            
            day_data = process_day(sym, d, g, daily_closes)
            
            if day_data is None:
                continue
            
            all_days_data.append(day_data)
            
            # Store candles with day metadata
            candles = day_data["df_15m"].copy()
            candles["symbol"] = sym
            candles["date"] = d
            candles["vwap"] = day_data["vwap"]
            candles["daily_bias"] = day_data["daily_bias"]
            candles["sma50"] = day_data["sma50"]
            candles["sma200"] = day_data["sma200"]
            candles["overnight_range"] = day_data["overnight_range"]
            candles["session_high"] = day_data["session_high"]
            candles["session_low"] = day_data["session_low"]
            all_candles.append(candles)
    
    # Process final residual
    if not residual.empty:
        residual = _determine_front_month(residual)
        for (sym, d), g in tqdm(residual.groupby(["symbol", "et_date"]), desc="Final residual"):
            g = g.sort_values("ts_et").reset_index(drop=True)
            
            # Store daily close
            rth_mask = _within(g["ts_et"], RTH_START, RTH_END)
            rth = g.loc[rth_mask]
            if not rth.empty:
                daily_closes[d] = rth.iloc[-1]["close"]
            
            day_data = process_day(sym, d, g, daily_closes)
            
            if day_data is None:
                continue
            
            all_days_data.append(day_data)
            candles = day_data["df_15m"].copy()
            candles["symbol"] = sym
            candles["date"] = d
            candles["vwap"] = day_data["vwap"]
            candles["daily_bias"] = day_data["daily_bias"]
            candles["sma50"] = day_data["sma50"]
            candles["sma200"] = day_data["sma200"]
            candles["overnight_range"] = day_data["overnight_range"]
            candles["session_high"] = day_data["session_high"]
            candles["session_low"] = day_data["session_low"]
            all_candles.append(candles)
    
    if not all_candles:
        print("No data processed!")
        return
    
    print(f"Processed {len(all_days_data)} days")
    
    # Combine all candles
    df_all_candles = pd.concat(all_candles, ignore_index=True)
    print(f"Total candles: {len(df_all_candles)}")
    
    # Now generate all analysis outputs
    print("\nGenerating analysis outputs...")
    
    # Module 1: Candle Structure & Volatility
    generate_structure_stats(df_all_candles)
    
    # Module 2: Candle Sequence Behavior
    generate_sequence_behavior(df_all_candles, all_days_data)
    
    # Module 3: Liquidity & High/Low Behavior
    generate_liquidity_behavior(df_all_candles, all_days_data)
    
    # Module 4: Session Phase Comparison
    generate_session_phase_comparison(df_all_candles)
    
    # Module 5: Bias & Context Integration
    generate_bias_context(all_days_data)
    
    # Module 6: Reversion & Mean Behavior
    generate_reversion_behavior(df_all_candles, all_days_data)
    
    # Module 7: Microstructure & Volume
    generate_volume_analysis(df_all_candles)
    
    # Module 8: First 3 Candles Deep Dive (9:30-10:15)
    generate_first_3_candles_analysis(df_all_candles, all_days_data)
    
    print("\n✅ Analysis complete!")


def generate_structure_stats(df_candles: pd.DataFrame):
    """Module 1: Candle Structure & Volatility"""
    print("Generating candle structure stats...")
    
    results = []
    
    # Overall stats
    median_size = df_candles["candle_range"].median()
    mean_size = df_candles["candle_range"].mean()
    
    overall = {
        "metric_type": "overall",
        "metric": "all_candles",
        "mean_range": mean_size,
        "median_range": median_size,
        "std_range": df_candles["candle_range"].std(),
        "p10_range": df_candles["candle_range"].quantile(0.10),
        "p25_range": df_candles["candle_range"].quantile(0.25),
        "p75_range": df_candles["candle_range"].quantile(0.75),
        "p90_range": df_candles["candle_range"].quantile(0.90),
        "p95_range": df_candles["candle_range"].quantile(0.95),
        "p99_range": df_candles["candle_range"].quantile(0.99),
        "pct_above_1x_median": (df_candles["candle_range"] > median_size).mean() * 100,
        "pct_above_1.5x_median": (df_candles["candle_range"] > 1.5 * median_size).mean() * 100,
        "pct_above_2x_median": (df_candles["candle_range"] > 2 * median_size).mean() * 100,
        "mean_wick_to_body": df_candles[df_candles["wick_to_body_ratio"] != np.inf]["wick_to_body_ratio"].mean(),
        "median_wick_to_body": df_candles[df_candles["wick_to_body_ratio"] != np.inf]["wick_to_body_ratio"].median(),
        "mean_bullish_size": df_candles[df_candles["direction"] == "bullish"]["candle_range"].mean(),
        "mean_bearish_size": df_candles[df_candles["direction"] == "bearish"]["candle_range"].mean(),
        "bullish_count": (df_candles["direction"] == "bullish").sum(),
        "bearish_count": (df_candles["direction"] == "bearish").sum(),
    }
    results.append(overall)
    
    # Per session block
    for block in ["opening_drive", "mid_morning", "midday", "afternoon"]:
        block_data = df_candles[df_candles["session_block"] == block]
        if len(block_data) == 0:
            continue
        
        block_stats = {
            "metric_type": "session_block",
            "metric": block,
            "mean_range": block_data["candle_range"].mean(),
            "median_range": block_data["candle_range"].median(),
            "std_range": block_data["candle_range"].std(),
            "pct_above_1x_median": (block_data["candle_range"] > median_size).mean() * 100,
            "pct_above_1.5x_median": (block_data["candle_range"] > 1.5 * median_size).mean() * 100,
            "pct_above_2x_median": (block_data["candle_range"] > 2 * median_size).mean() * 100,
            "mean_wick_to_body": block_data[block_data["wick_to_body_ratio"] != np.inf]["wick_to_body_ratio"].mean(),
            "median_wick_to_body": block_data[block_data["wick_to_body_ratio"] != np.inf]["wick_to_body_ratio"].median(),
            "mean_bullish_size": block_data[block_data["direction"] == "bullish"]["candle_range"].mean(),
            "mean_bearish_size": block_data[block_data["direction"] == "bearish"]["candle_range"].mean(),
        }
        results.append(block_stats)
    
    # Per time slot (first few 15-min slots)
    slot_times = [time(9, 30), time(9, 45), time(10, 0), time(10, 15)]
    slot_labels = ["09:30-09:45", "09:45-10:00", "10:00-10:15", "10:15-10:30"]
    
    for slot_time, slot_label in zip(slot_times, slot_labels):
        slot_data = df_candles[df_candles["time_slot"] == slot_time]
        if len(slot_data) == 0:
            continue
        
        slot_stats = {
            "metric_type": "time_slot",
            "metric": slot_label,
            "mean_range": slot_data["candle_range"].mean(),
            "median_range": slot_data["candle_range"].median(),
            "std_range": slot_data["candle_range"].std(),
        }
        results.append(slot_stats)
    
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_DIR / "candle_structure_stats.csv", index=False)
    print(f"✅ Wrote candle_structure_stats.csv ({len(df_out)} rows)")


def generate_sequence_behavior(df_candles: pd.DataFrame, all_days_data: List[Dict]):
    """Module 2: Candle Sequence Behavior"""
    print("Generating sequence behavior stats...")
    
    results = []
    
    # Group by day for sequence analysis
    for day_data in all_days_data:
        df_day = day_data["df_15m"].copy()
        if len(df_day) < 2:
            continue
        
        # First candle analysis
        first = df_day.iloc[0]
        if len(df_day) > 1:
            second = df_day.iloc[1]
            
            # First candle bullish & >1.5x avg
            avg_size = df_day["candle_range"].mean()
            if first["direction"] == "bullish" and first["candle_range"] > 1.5 * avg_size:
                results.append({
                    "date": day_data["date"],
                    "pattern": "first_bullish_large",
                    "next_candle_bullish": 1 if second["direction"] == "bullish" else 0,
                    "next_candle_size": second["candle_range"],
                })
            
            # First candle with large wicks both sides
            if (first["wick_upper"] > first["body_size"]) and (first["wick_lower"] > first["body_size"]):
                same_dir = 1 if second["direction"] == first["direction"] else 0
                results.append({
                    "date": day_data["date"],
                    "pattern": "first_indecision",
                    "next_same_direction": same_dir,
                    "next_candle_size": second["candle_range"],
                })
        
        # Streak analysis
        streaks = detect_sequences(df_day)
        if streaks["streaks_3plus"] > 0:
            # Find streaks of 3+
            df_day["streak"] = (df_day["direction"] != df_day["direction"].shift()).cumsum()
            streak_groups = df_day.groupby("streak")
            
            for streak_id, group in streak_groups:
                if len(group) >= 3:
                    streak_dir = group.iloc[0]["direction"]
                    streak_end_idx = group.index[-1]
                    
                    if streak_end_idx < len(df_day) - 1:
                        next_candle = df_day.iloc[streak_end_idx + 1]
                        reversal = 1 if next_candle["direction"] != streak_dir else 0
                        
                        results.append({
                            "date": day_data["date"],
                            "pattern": "streak_3plus",
                            "streak_direction": streak_dir,
                            "streak_length": len(group),
                            "reversal": reversal,
                            "continuation_size": next_candle["candle_range"] if not reversal else np.nan,
                            "reversal_size": next_candle["candle_range"] if reversal else np.nan,
                        })
        
        # Outside bars
        for i in range(1, len(df_day)):
            curr = df_day.iloc[i]
            prev = df_day.iloc[i-1]
            
            if (curr["high"] > prev["high"]) and (curr["low"] < prev["low"]):
                if i < len(df_day) - 1:
                    next_candle = df_day.iloc[i+1]
                    continuation = 1 if next_candle["direction"] == curr["direction"] else 0
                    
                    results.append({
                        "date": day_data["date"],
                        "pattern": "outside_bar",
                        "outside_bar_direction": curr["direction"],
                        "continuation": continuation,
                        "next_candle_size": next_candle["candle_range"],
                    })
        
        # Large trend candle retracement
        median_size = df_day["candle_range"].median()
        for i in range(len(df_day) - 1):
            curr = df_day.iloc[i]
            if curr["candle_range"] > 1.5 * median_size:
                # Check next candles for retracement
                for j in range(i+1, min(i+4, len(df_day))):
                    next_candle = df_day.iloc[j]
                    retrace_pct = (curr["candle_range"] - next_candle["candle_range"]) / curr["candle_range"] * 100
                    
                    if retrace_pct > 50:  # Retraced more than 50%
                        results.append({
                            "date": day_data["date"],
                            "pattern": "large_trend_retrace",
                            "trend_candle_size": curr["candle_range"],
                            "retrace_candle_size": next_candle["candle_range"],
                            "retrace_pct": retrace_pct,
                            "bars_to_retrace": j - i,
                        })
                        break
    
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(OUT_DIR / "candle_sequence_behavior.csv", index=False)
        print(f"✅ Wrote candle_sequence_behavior.csv ({len(df_out)} rows)")
    else:
        print("⚠️ No sequence behavior data")


def generate_liquidity_behavior(df_candles: pd.DataFrame, all_days_data: List[Dict]):
    """Module 3: Liquidity & High/Low Behavior"""
    print("Generating liquidity behavior stats...")
    
    results = []
    
    for day_data in all_days_data:
        df_day = day_data["df_15m"].copy()
        if len(df_day) < 2:
            continue
        
        # Session high/low timing
        first_30min = df_day.iloc[:2] if len(df_day) >= 2 else df_day
        session_high_in_first_30min = day_data["session_high_time"] <= time(10, 0)
        session_low_in_first_30min = day_data["session_low_time"] <= time(10, 0)
        
        # Distance from first candle to session high/low
        first_candle = df_day.iloc[0]
        dist_to_session_high = day_data["session_high"] - first_candle["high"]
        dist_to_session_low = first_candle["low"] - day_data["session_low"]
        
        # Opening range persistence (9:30-9:45 high/low unbroken until 10:30)
        if len(df_day) >= 4:  # Need at least 4 candles (until 10:30)
            first_candle_high = first_candle["high"]
            first_candle_low = first_candle["low"]
            next_hour = df_day.iloc[1:4]  # 9:45, 10:00, 10:15
            
            high_unbroken = (next_hour["high"] <= first_candle_high).all()
            low_unbroken = (next_hour["low"] >= first_candle_low).all()
        else:
            high_unbroken = False
            low_unbroken = False
        
        # Reversal structure (high/low in first 30 min, then reverses)
        if session_high_in_first_30min and len(df_day) >= 3:
            # Check if price reverses down after high
            high_candle_idx = next((i for i, c in enumerate(df_day.iloc[:2].itertuples()) 
                                   if c.high == day_data["session_high"]), None)
            if high_candle_idx is not None and high_candle_idx < len(df_day) - 1:
                later_candles = df_day.iloc[high_candle_idx+1:]
                reversal_structure = (later_candles["close"] < day_data["session_high"]).any()
            else:
                reversal_structure = False
        else:
            reversal_structure = False
        
        # Liquidity sweeps (from detect_liquidity_sweeps)
        sweeps = day_data["sweeps"]
        
        results.append({
            "date": day_data["date"],
            "session_high_time": day_data["session_high_time"],
            "session_low_time": day_data["session_low_time"],
            "session_high_in_first_30min": session_high_in_first_30min,
            "session_low_in_first_30min": session_low_in_first_30min,
            "dist_first_candle_to_session_high": dist_to_session_high,
            "dist_first_candle_to_session_low": dist_to_session_low,
            "opening_range_high_unbroken_1hr": high_unbroken,
            "opening_range_low_unbroken_1hr": low_unbroken,
            "reversal_structure": reversal_structure,
            "liquidity_sweeps_count": len(sweeps),
            "liquidity_sweeps_high": len([s for s in sweeps if s["sweep_type"] == "high"]),
            "liquidity_sweeps_low": len([s for s in sweeps if s["sweep_type"] == "low"]),
        })
    
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_DIR / "liquidity_high_low_behavior.csv", index=False)
    print(f"✅ Wrote liquidity_high_low_behavior.csv ({len(df_out)} rows)")


def generate_session_phase_comparison(df_candles: pd.DataFrame):
    """Module 4: Session Phase Comparison"""
    print("Generating session phase comparison...")
    
    results = []
    
    # Average range per phase
    for block in ["opening_drive", "mid_morning", "midday", "afternoon"]:
        block_data = df_candles[df_candles["session_block"] == block]
        if len(block_data) == 0:
            continue
        
        results.append({
            "phase": block,
            "mean_range": block_data["candle_range"].mean(),
            "median_range": block_data["candle_range"].median(),
            "std_range": block_data["candle_range"].std(),
            "candle_count": len(block_data),
        })
    
    # Reversal frequency by block
    for day, day_candles in df_candles.groupby("date"):
        for block in ["opening_drive", "mid_morning", "midday", "afternoon"]:
            block_candles = day_candles[day_candles["session_block"] == block]
            if len(block_candles) < 3:
                continue
            
            # Count 3+ candle reversals in this block
            reversals = 0
            for i in range(len(block_candles) - 2):
                if (block_candles.iloc[i]["direction"] == block_candles.iloc[i+1]["direction"] == 
                    block_candles.iloc[i+2]["direction"]):
                    if i+3 < len(block_candles):
                        if block_candles.iloc[i+3]["direction"] != block_candles.iloc[i]["direction"]:
                            reversals += 1
            
            results.append({
                "phase": f"{block}_reversals",
                "date": day,
                "reversals_3plus": reversals,
            })
    
    # Volatility expansion times (by time slot) - aggregate stats
    for slot_time in sorted(df_candles["time_slot"].unique()):
        slot_data = df_candles[df_candles["time_slot"] == slot_time]
        if len(slot_data) == 0:
            continue
        
        slot_results = {
            "phase": f"volatility_slot_{slot_time}",
            "time_slot": str(slot_time),
            "mean_range": slot_data["candle_range"].mean(),
            "median_range": slot_data["candle_range"].median(),
            "std_range": slot_data["candle_range"].std(),
            "p95_range": slot_data["candle_range"].quantile(0.95),
        }
        results.append(slot_results)
    
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_DIR / "session_phase_comparison.csv", index=False)
    print(f"✅ Wrote session_phase_comparison.csv ({len(df_out)} rows)")


def generate_bias_context(all_days_data: List[Dict]):
    """Module 5: Bias & Context Integration"""
    print("Generating bias & context integration...")
    
    results = []
    
    for day_data in all_days_data:
        df_day = day_data["df_15m"].copy()
        if len(df_day) == 0:
            continue
        
        first_candle = df_day.iloc[0]
        daily_bias = day_data["daily_bias"]
        
        # First candle direction
        first_bullish = 1 if first_candle["direction"] == "bullish" else 0
        
        # First 2-hour range (9:30-11:30)
        first_2hr = df_day[df_day["time_slot"] <= time(11, 30)]
        first_2hr_range = first_2hr["high"].max() - first_2hr["low"].min() if len(first_2hr) > 0 else np.nan
        
        # VWAP position relative to first candle
        vwap = day_data["vwap"]
        first_close = first_candle["close"]
        vwap_above_first = 1 if (not pd.isna(vwap) and vwap > first_close) else 0
        
        # Daily close alignment
        daily_close = day_data["daily_close"]
        first_dir_aligns_close = 1 if (
            (first_bullish == 1 and daily_close > first_close) or
            (first_bullish == 0 and daily_close < first_close)
        ) else 0
        
        results.append({
            "date": day_data["date"],
            "daily_bias": daily_bias,
            "first_candle_bullish": first_bullish,
            "first_candle_size": first_candle["candle_range"],
            "first_2hr_range": first_2hr_range,
            "overnight_range": day_data["overnight_range"],
            "vwap": vwap,
            "vwap_above_first_candle": vwap_above_first,
            "daily_close": daily_close,
            "first_dir_aligns_close": first_dir_aligns_close,
        })
    
    df_out = pd.DataFrame(results)
    
    # Calculate aggregate statistics
    summary_rows = []
    
    # Conditional probabilities by bias
    for bias in ["bullish", "bearish"]:
        bias_days = df_out[df_out["daily_bias"] == bias]
        if len(bias_days) > 0:
            summary_rows.append({
                "metric_type": "bias_probability",
                "daily_bias": bias,
                "p_first_candle_bullish": bias_days["first_candle_bullish"].mean() * 100,
                "mean_first_2hr_range": bias_days["first_2hr_range"].mean(),
                "count": len(bias_days),
            })
    
    # Overnight range impact (bucket into quartiles)
    overnight_ranges = df_out[df_out["overnight_range"].notna()]["overnight_range"]
    if len(overnight_ranges) > 0:
        q1, q2, q3 = overnight_ranges.quantile([0.25, 0.50, 0.75])
        
        for q_name, q_val in [("Q1", q1), ("Q2", q2), ("Q3", q3), ("Q4", float("inf"))]:
            if q_name == "Q1":
                q_data = df_out[df_out["overnight_range"] <= q_val]
            elif q_name == "Q4":
                q_data = df_out[df_out["overnight_range"] > q3]
            elif q_name == "Q2":
                q_data = df_out[(df_out["overnight_range"] > q1) & (df_out["overnight_range"] <= q_val)]
            else:  # Q3
                q_data = df_out[(df_out["overnight_range"] > q2) & (df_out["overnight_range"] <= q_val)]
            
            if len(q_data) > 0:
                summary_rows.append({
                    "metric_type": "overnight_range_impact",
                    "quartile": q_name,
                    "mean_overnight_range": q_data["overnight_range"].mean(),
                    "mean_first_candle_size": q_data["first_candle_size"].mean(),
                    "count": len(q_data),
                })
    
    # VWAP correlation
    vwap_data = df_out[df_out["vwap"].notna()]
    if len(vwap_data) > 0:
        summary_rows.append({
            "metric_type": "vwap_correlation",
            "p_first_dir_aligns_close": vwap_data["first_dir_aligns_close"].mean() * 100,
            "p_vwap_above_first_when_bullish": vwap_data[vwap_data["first_candle_bullish"] == 1]["vwap_above_first_candle"].mean() * 100,
            "p_vwap_above_first_when_bearish": vwap_data[vwap_data["first_candle_bullish"] == 0]["vwap_above_first_candle"].mean() * 100,
        })
    
    # Combine detailed and summary
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_final = pd.concat([df_out, df_summary], ignore_index=True)
    else:
        df_final = df_out
    
    df_final.to_csv(OUT_DIR / "bias_context_integration.csv", index=False)
    print(f"✅ Wrote bias_context_integration.csv ({len(df_final)} rows)")


def generate_reversion_behavior(df_candles: pd.DataFrame, all_days_data: List[Dict]):
    """Module 6: Reversion & Mean Behavior"""
    print("Generating reversion behavior...")
    
    results = []
    
    for day_data in all_days_data:
        df_day = day_data["df_15m"].copy()
        if len(df_day) < 2:
            continue
        
        vwap = day_data["vwap"]
        if pd.isna(vwap):
            continue
        
        median_size = df_day["candle_range"].median()
        
        for i in range(len(df_day) - 1):
            curr = df_day.iloc[i]
            next_candle = df_day.iloc[i+1]
            
            # Extreme close reversions (top/bottom 10% of range)
            candle_range = curr["high"] - curr["low"]
            if candle_range > 0:
                close_position = (curr["close"] - curr["low"]) / candle_range
                
                if close_position <= 0.1 or close_position >= 0.9:
                    results.append({
                        "date": day_data["date"],
                        "pattern": "extreme_close",
                        "close_position": "bottom_10" if close_position <= 0.1 else "top_10",
                        "next_candle_direction": next_candle["direction"],
                        "next_candle_size": next_candle["candle_range"],
                        "reversal": 1 if (
                            (close_position <= 0.1 and next_candle["direction"] == "bullish") or
                            (close_position >= 0.9 and next_candle["direction"] == "bearish")
                        ) else 0,
                    })
            
            # VWAP mean reversion after large displacement
            if curr["candle_range"] > 1.5 * median_size:
                curr_price = curr["close"]
                dist_to_vwap = abs(curr_price - vwap)
                
                if dist_to_vwap > 0:
                    # Check next candles for return to VWAP
                    for j in range(i+1, min(i+4, len(df_day))):
                        later_candle = df_day.iloc[j]
                        later_dist = abs(later_candle["close"] - vwap)
                        
                        if later_dist < dist_to_vwap * 0.5:  # Returned halfway to VWAP
                            results.append({
                                "date": day_data["date"],
                                "pattern": "vwap_mean_reversion",
                                "large_candle_size": curr["candle_range"],
                                "bars_to_return": j - i,
                                "initial_dist_to_vwap": dist_to_vwap,
                                "final_dist_to_vwap": later_dist,
                            })
                            break
        
        # Streak reversals
        df_day_copy = df_day.copy()
        df_day_copy["streak"] = (df_day_copy["direction"] != df_day_copy["direction"].shift()).cumsum()
        streak_groups = df_day_copy.groupby("streak")
        
        for streak_id, group in streak_groups:
            if len(group) >= 3:
                streak_dir = group.iloc[0]["direction"]
                streak_end_idx = group.index[-1]
                
                if streak_end_idx < len(df_day) - 1:
                    next_candle = df_day.iloc[streak_end_idx + 1]
                    reversal = 1 if next_candle["direction"] != streak_dir else 0
                    
                    results.append({
                        "date": day_data["date"],
                        "pattern": "streak_reversal",
                        "streak_length": len(group),
                        "reversal": reversal,
                        "reversal_size": next_candle["candle_range"] if reversal else np.nan,
                    })
        
        # Volatility expansion reversion (within 2 bars)
        for i in range(len(df_day) - 3):
            curr = df_day.iloc[i]
            if curr["candle_range"] > 1.5 * median_size:
                # Check next 2 candles for reversion
                next_two = df_day.iloc[i+1:i+3]
                reversion_occurred = False
                
                for j, later_candle in enumerate(next_two.itertuples(), 1):
                    if later_candle.candle_range < median_size * 0.75:  # Reversion to below median
                        results.append({
                            "date": day_data["date"],
                            "pattern": "volatility_expansion_reversion",
                            "expansion_candle_size": curr["candle_range"],
                            "bars_to_reversion": j,
                            "reversion_candle_size": later_candle.candle_range,
                        })
                        reversion_occurred = True
                        break
    
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(OUT_DIR / "reversion_mean_behavior.csv", index=False)
        print(f"✅ Wrote reversion_mean_behavior.csv ({len(df_out)} rows)")
    else:
        print("⚠️ No reversion behavior data")


def generate_volume_analysis(df_candles: pd.DataFrame):
    """Module 7: Microstructure & Volume"""
    print("Generating volume analysis...")
    
    results = []
    
    # Volume per session block
    for block in ["opening_drive", "mid_morning", "midday", "afternoon"]:
        block_data = df_candles[df_candles["session_block"] == block]
        if len(block_data) == 0:
            continue
        
        results.append({
            "metric_type": "volume_by_block",
            "block": block,
            "mean_volume": block_data["volume"].mean(),
            "median_volume": block_data["volume"].median(),
            "std_volume": block_data["volume"].std(),
        })
    
    # Volume imbalance vs candle direction
    volume_data = df_candles[df_candles["volume_delta"].notna()].copy()
    if len(volume_data) > 0:
        for direction in ["bullish", "bearish"]:
            dir_data = volume_data[volume_data["direction"] == direction]
            if len(dir_data) > 0:
                results.append({
                    "metric_type": "volume_imbalance",
                    "candle_direction": direction,
                    "mean_volume_delta": dir_data["volume_delta"].mean(),
                    "mean_volume_imbalance": dir_data["volume_imbalance"].mean(),
                    "mean_volume": dir_data["volume"].mean(),
                })
        
        # Absorption (large volume, small movement)
        median_volume = volume_data["volume"].median()
        median_range = volume_data["candle_range"].median()
        
        absorption = volume_data[
            (volume_data["volume"] > 1.5 * median_volume) &
            (volume_data["candle_range"] < 0.75 * median_range)
        ]
        
        results.append({
            "metric_type": "absorption",
            "count": len(absorption),
            "pct_of_total": len(absorption) / len(volume_data) * 100,
            "mean_volume": absorption["volume"].mean() if len(absorption) > 0 else np.nan,
            "mean_range": absorption["candle_range"].mean() if len(absorption) > 0 else np.nan,
        })
    
    # Normalized candle size (efficiency)
    volume_data["candle_efficiency"] = volume_data["candle_range"] / volume_data["volume"].replace(0, np.nan)
    
    for block in ["opening_drive", "mid_morning", "midday", "afternoon"]:
        block_data = volume_data[volume_data["session_block"] == block]
        if len(block_data) > 0:
            results.append({
                "metric_type": "efficiency",
                "block": block,
                "mean_efficiency": block_data["candle_efficiency"].mean(),
                "median_efficiency": block_data["candle_efficiency"].median(),
            })
    
    # FVG analysis (aggregate from all days)
    all_fvgs = []
    for date, day_group in df_candles.groupby("date"):
        day_candles = day_group.sort_values("candle_time").reset_index(drop=True)
        if len(day_candles) < 2:
            continue
        fvgs = detect_fvg(day_candles)
        for fvg in fvgs:
            fvg["date"] = date
            # Get time_slot from the candle
            if "time_slot" in day_candles.columns:
                fvg["time_slot"] = day_candles.loc[fvg["candle_idx"], "time_slot"]
        all_fvgs.extend(fvgs)
    
    if all_fvgs:
        df_fvgs = pd.DataFrame(all_fvgs)
        results.append({
            "metric_type": "fvg_summary",
            "total_fvgs": len(df_fvgs),
            "bullish_fvgs": (df_fvgs["fvg_type"] == "bullish").sum(),
            "bearish_fvgs": (df_fvgs["fvg_type"] == "bearish").sum(),
            "mean_gap_size": df_fvgs["gap_size"].mean(),
        })
        
        # FVG by time slot
        if "time_slot" in df_fvgs.columns:
            for slot_time in df_fvgs["time_slot"].dropna().unique():
                slot_fvgs = df_fvgs[df_fvgs["time_slot"] == slot_time]
                results.append({
                    "metric_type": "fvg_by_slot",
                    "time_slot": str(slot_time),
                    "count": len(slot_fvgs),
                    "mean_gap_size": slot_fvgs["gap_size"].mean(),
                })
    
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_DIR / "microstructure_volume.csv", index=False)
    print(f"✅ Wrote microstructure_volume.csv ({len(df_out)} rows)")


def generate_first_3_candles_analysis(df_candles: pd.DataFrame, all_days_data: List[Dict]):
    """Module 8: Deep Dive on First 3 Candles (9:30-10:15)"""
    print("Generating first 3 candles analysis...")
    
    results = []
    
    # Filter to first 3 candles
    first_3_mask = df_candles["time_slot"].isin([time(9, 30), time(9, 45), time(10, 0)])
    first_3_candles = df_candles[first_3_mask].copy()
    
    # Aggregate stats by candle position
    for candle_num in [1, 2, 3]:
        candle_data = first_3_candles[first_3_candles["time_slot"] == [time(9, 30), time(9, 45), time(10, 0)][candle_num-1]]
        if len(candle_data) == 0:
            continue
        
        results.append({
            "metric_type": "candle_position",
            "candle_num": candle_num,
            "time_slot": str([time(9, 30), time(9, 45), time(10, 0)][candle_num-1]),
            "mean_range": candle_data["candle_range"].mean(),
            "median_range": candle_data["candle_range"].median(),
            "std_range": candle_data["candle_range"].std(),
            "p10_range": candle_data["candle_range"].quantile(0.10),
            "p90_range": candle_data["candle_range"].quantile(0.90),
            "mean_volume": candle_data["volume"].mean() if "volume" in candle_data.columns else np.nan,
            "median_volume": candle_data["volume"].median() if "volume" in candle_data.columns else np.nan,
            "pct_bullish": (candle_data["direction"] == "bullish").mean() * 100,
            "pct_bearish": (candle_data["direction"] == "bearish").mean() * 100,
            "mean_wick_to_body": candle_data[candle_data["wick_to_body_ratio"] != np.inf]["wick_to_body_ratio"].mean(),
            "median_wick_to_body": candle_data[candle_data["wick_to_body_ratio"] != np.inf]["wick_to_body_ratio"].median(),
        })
    
    # Daily patterns - analyze first 3 candles together
    for day_data in all_days_data:
        df_day = day_data["df_15m"].copy()
        if len(df_day) < 3:
            continue
        
        first_3 = df_day.iloc[:3].copy()
        
        # Opening range (high/low of first 3 candles)
        opening_range_high = first_3["high"].max()
        opening_range_low = first_3["low"].min()
        opening_range_size = opening_range_high - opening_range_low
        
        # Directional sequence
        dirs = first_3["direction"].tolist()
        sequence = "-".join(dirs)
        
        # Check if all same direction
        all_same_direction = (dirs[0] == dirs[1] == dirs[2])
        
        # Check for reversals
        reversal_after_c1 = dirs[1] != dirs[0] if len(dirs) > 1 else False
        reversal_after_c2 = dirs[2] != dirs[1] if len(dirs) > 2 else False
        
        # Candle sizes
        c1_size = first_3.iloc[0]["candle_range"]
        c2_size = first_3.iloc[1]["candle_range"]
        c3_size = first_3.iloc[2]["candle_range"]
        
        # Size relationships
        c2_larger_than_c1 = c2_size > c1_size
        c3_larger_than_c2 = c3_size > c2_size
        c2_smaller_than_c1 = c2_size < c1_size * 0.75
        c3_smaller_than_c2 = c3_size < c2_size * 0.75
        
        # Volume patterns (if available)
        if "volume" in first_3.columns and first_3["volume"].notna().any():
            c1_vol = first_3.iloc[0]["volume"]
            c2_vol = first_3.iloc[1]["volume"]
            c3_vol = first_3.iloc[2]["volume"]
            total_vol_first_3 = c1_vol + c2_vol + c3_vol
        else:
            c1_vol = c2_vol = c3_vol = total_vol_first_3 = np.nan
        
        # Liquidity sweeps in first 3 candles
        sweeps_in_first_3 = [s for s in day_data["sweeps"] if s["candle_idx"] < 3]
        
        # How first 3 candles relate to session extremes
        session_high = day_data["session_high"]
        session_low = day_data["session_low"]
        opening_range_high_is_session_high = abs(opening_range_high - session_high) < 0.1
        opening_range_low_is_session_low = abs(opening_range_low - session_low) < 0.1
        
        # Distance from opening range to session extremes
        dist_to_session_high = session_high - opening_range_high
        dist_to_session_low = opening_range_low - session_low
        
        # Price at end of first 3 candles
        end_of_first_3_price = first_3.iloc[-1]["close"]
        start_price = first_3.iloc[0]["open"]
        net_move_first_3 = end_of_first_3_price - start_price
        
        # Daily close
        daily_close = day_data["daily_close"]
        first_3_aligned_with_close = (
            (net_move_first_3 > 0 and daily_close > start_price) or
            (net_move_first_3 < 0 and daily_close < start_price)
        )
        
        # VWAP position
        vwap = day_data["vwap"]
        vwap_above_first_3_close = vwap > end_of_first_3_price if not pd.isna(vwap) else np.nan
        
        results.append({
            "date": day_data["date"],
            "metric_type": "daily_pattern",
            "c1_direction": dirs[0],
            "c2_direction": dirs[1],
            "c3_direction": dirs[2],
            "sequence": sequence,
            "all_same_direction": all_same_direction,
            "reversal_after_c1": reversal_after_c1,
            "reversal_after_c2": reversal_after_c2,
            "c1_size": c1_size,
            "c2_size": c2_size,
            "c3_size": c3_size,
            "c2_larger_than_c1": c2_larger_than_c1,
            "c3_larger_than_c2": c3_larger_than_c2,
            "c2_smaller_than_c1": c2_smaller_than_c1,
            "c3_smaller_than_c2": c3_smaller_than_c2,
            "opening_range_high": opening_range_high,
            "opening_range_low": opening_range_low,
            "opening_range_size": opening_range_size,
            "c1_volume": c1_vol,
            "c2_volume": c2_vol,
            "c3_volume": c3_vol,
            "total_vol_first_3": total_vol_first_3,
            "sweeps_in_first_3": len(sweeps_in_first_3),
            "opening_range_high_is_session_high": opening_range_high_is_session_high,
            "opening_range_low_is_session_low": opening_range_low_is_session_low,
            "dist_to_session_high": dist_to_session_high,
            "dist_to_session_low": dist_to_session_low,
            "net_move_first_3": net_move_first_3,
            "first_3_aligned_with_close": first_3_aligned_with_close,
            "vwap_above_first_3_close": vwap_above_first_3_close,
            "daily_bias": day_data["daily_bias"],
        })
    
    # Calculate aggregate statistics
    daily_patterns = pd.DataFrame([r for r in results if r.get("metric_type") == "daily_pattern"])
    
    if len(daily_patterns) > 0:
        # Sequence probabilities
        sequence_counts = daily_patterns["sequence"].value_counts()
        for seq, count in sequence_counts.head(10).items():
            results.append({
                "metric_type": "sequence_probability",
                "sequence": seq,
                "count": count,
                "pct": count / len(daily_patterns) * 100,
            })
        
        # All same direction probability
        results.append({
            "metric_type": "aggregate_stats",
            "stat": "pct_all_same_direction",
            "value": daily_patterns["all_same_direction"].mean() * 100,
            "count": daily_patterns["all_same_direction"].sum(),
        })
        
        # Reversal probabilities
        results.append({
            "metric_type": "aggregate_stats",
            "stat": "pct_reversal_after_c1",
            "value": daily_patterns["reversal_after_c1"].mean() * 100,
            "count": daily_patterns["reversal_after_c1"].sum(),
        })
        
        results.append({
            "metric_type": "aggregate_stats",
            "stat": "pct_reversal_after_c2",
            "value": daily_patterns["reversal_after_c2"].mean() * 100,
            "count": daily_patterns["reversal_after_c2"].sum(),
        })
        
        # Opening range as session extremes
        results.append({
            "metric_type": "aggregate_stats",
            "stat": "pct_opening_range_high_is_session_high",
            "value": daily_patterns["opening_range_high_is_session_high"].mean() * 100,
            "count": daily_patterns["opening_range_high_is_session_high"].sum(),
        })
        
        results.append({
            "metric_type": "aggregate_stats",
            "stat": "pct_opening_range_low_is_session_low",
            "value": daily_patterns["opening_range_low_is_session_low"].mean() * 100,
            "count": daily_patterns["opening_range_low_is_session_low"].sum(),
        })
        
        # First 3 alignment with daily close
        results.append({
            "metric_type": "aggregate_stats",
            "stat": "pct_first_3_aligned_with_close",
            "value": daily_patterns["first_3_aligned_with_close"].mean() * 100,
            "count": daily_patterns["first_3_aligned_with_close"].sum(),
        })
        
        # Average sweeps in first 3
        results.append({
            "metric_type": "aggregate_stats",
            "stat": "mean_sweeps_in_first_3",
            "value": daily_patterns["sweeps_in_first_3"].mean(),
            "count": len(daily_patterns),
        })
        
        # Average opening range size
        results.append({
            "metric_type": "aggregate_stats",
            "stat": "mean_opening_range_size",
            "value": daily_patterns["opening_range_size"].mean(),
            "count": len(daily_patterns),
        })
        
        results.append({
            "metric_type": "aggregate_stats",
            "stat": "median_opening_range_size",
            "value": daily_patterns["opening_range_size"].median(),
            "count": len(daily_patterns),
        })
        
        # Size decrease patterns
        results.append({
            "metric_type": "aggregate_stats",
            "stat": "pct_c2_smaller_than_c1",
            "value": daily_patterns["c2_smaller_than_c1"].mean() * 100,
            "count": daily_patterns["c2_smaller_than_c1"].sum(),
        })
        
        results.append({
            "metric_type": "aggregate_stats",
            "stat": "pct_c3_smaller_than_c2",
            "value": daily_patterns["c3_smaller_than_c2"].mean() * 100,
            "count": daily_patterns["c3_smaller_than_c2"].sum(),
        })
    
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_DIR / "first_3_candles_analysis.csv", index=False)
    print(f"✅ Wrote first_3_candles_analysis.csv ({len(df_out)} rows)")


if __name__ == "__main__":
    main()

