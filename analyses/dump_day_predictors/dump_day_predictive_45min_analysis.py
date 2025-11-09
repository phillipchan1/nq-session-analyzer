#!/usr/bin/env python3
"""
45-Minute Opening Move Predictive Analysis
------------------------------------------

Predicts large opening-drive moves in the first 45 minutes of NY RTH (09:30-10:15 ET)
using ONLY information available before 9:30 AM ET on the trading day.

CRITICAL CONSTRAINT: All features must use data from:
- Previous RTH sessions (days < D)
- Overnight session ending at 09:29:59 on day D
- Higher timeframe bars that closed before 09:30 ET on day D
- External data (VIX, events) as of pre-open

NO look-ahead bias: Labels use day D's 45-minute window (09:30-10:15 ET) only.

Label Definitions:
- BigMove45: abs(close_45 - open_45) >= 0.75 * ATR_20
- ExtremeMove45: abs(close_45 - open_45) >= 200 points
- Dump45: close_45 < open_45 AND (open_45 - low_45) >= 0.75 * ATR_20
- Pump45: close_45 > open_45 AND (high_45 - open_45) >= 0.75 * ATR_20

Outputs:
- dump_day_45min_predictive_features.csv: All predictive features and labels
- dump_day_45min_predictive_correlations.csv: Correlations with all 4 labels
- dump_day_45min_predictive_conditional_probs.csv: Conditional probabilities
- dump_day_45min_predictive_hypotheses.md: Pre-open checklist and top predictors
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import time, timedelta
import pytz
from tqdm import tqdm
try:
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  sklearn not available, skipping model fitting")

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20250926.ohlcv-1m.csv")
EVENTS_FILE = str(DATA_DIR / "us_high_impact_events_2020_to_2025.csv")

# Session boundaries (ET)
RTH_START, RTH_END = time(9, 30), time(16, 0)  # Full RTH for historical features
OPEN_START, OPEN_END = time(9, 30), time(10, 15)  # 45-minute window for labels
OVERNIGHT_START_HOUR = 18  # 6 PM previous day
PRE_OPEN_CUTOFF = time(9, 29, 59)  # Cutoff for pre-open data

# Chunking
CHUNKSIZE = 1_000_000

# ATR period
ATR_PERIOD = 20


def _within(ts: pd.Series, start_t: time, end_t: time) -> pd.Series:
    """Check if timestamps are within time window."""
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)


def _load_events() -> pd.DataFrame:
    """Load and aggregate event data."""
    try:
        e = pd.read_csv(EVENTS_FILE)
    except Exception:
        return pd.DataFrame()
    e["date"] = pd.to_datetime(e["date"], errors="coerce").dt.date
    agg = e.groupby("date").agg({
        "event_name": lambda x: ", ".join(x.astype(str)),
        "event_type": lambda x: ", ".join(x.astype(str)),
        "session": lambda x: ", ".join(x.astype(str)),
        "time_et": lambda x: ", ".join(x.astype(str)),
    }).reset_index()
    agg.rename(columns={
        "event_name": "event_names",
        "event_type": "event_types",
        "session": "event_sessions",
        "time_et": "event_times",
    }, inplace=True)
    return agg


def _determine_front_month(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to front-month per date by max range in 9:30–10:15 ET window."""
    mask = _within(df["ts_et"], RTH_START, time(10, 15))
    rth = df.loc[mask]
    if rth.empty:
        return df
    agg = rth.groupby(["et_date", "symbol"]).agg(
        high=("high", "max"), low=("low", "min")
    ).reset_index()
    agg["range"] = agg["high"] - agg["low"]
    idx = agg.groupby("et_date")["range"].idxmax()
    fm = agg.loc[idx][["et_date", "symbol"]].rename(columns={"symbol": "front_symbol"})
    out = df.merge(fm, on="et_date", how="left")
    out = out.loc[out["symbol"] == out["front_symbol"]].drop(columns=["front_symbol"])
    return out


def compute_atr_from_ranges(ranges: pd.Series, period: int = ATR_PERIOD) -> float:
    """Compute ATR from daily ranges (simpler than intraday ATR)."""
    if len(ranges) < period:
        return np.nan
    return float(ranges.tail(period).mean())


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Compute Simple Moving Average."""
    return series.rolling(window=period, min_periods=1).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_keltner_channels(df: pd.DataFrame, ema_period: int = 20, 
                             atr_period: int = 10, multiplier: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Keltner Channels: middle (EMA), upper, lower bands."""
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    
    # ATR calculation
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=1).mean()
    
    upper = ema + (multiplier * atr)
    lower = ema - (multiplier * atr)
    
    return ema, upper, lower


def resample_to_timeframe(df: pd.DataFrame, timeframe: str, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Resample 1-minute data to higher timeframe, only including bars that closed before cutoff_ts.
    
    Args:
        df: 1-minute OHLCV DataFrame with ts_et column
        timeframe: '1H', '4H', '1D'
        cutoff_ts: Only include bars that closed before this timestamp
    
    Returns:
        Resampled DataFrame with only bars closed before cutoff
    """
    if df.empty or 'ts_et' not in df.columns:
        return pd.DataFrame()
    
    # Make a copy and ensure ts_et is datetime
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy['ts_et']):
        df_copy['ts_et'] = pd.to_datetime(df_copy['ts_et'])
    
    # Set ts_et as index for resampling
    df_indexed = df_copy.set_index('ts_et').copy()
    
    # Resample
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    if timeframe == '1H':
        rule = '60min'
    elif timeframe == '4H':
        rule = '240min'
    elif timeframe == '1D':
        rule = '1D'
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    
    try:
        resampled = df_indexed.resample(rule).agg(agg_dict).reset_index()
        resampled.rename(columns={'ts_et': 'bar_close_time'}, inplace=True)
        
        # Filter to bars that closed before cutoff
        resampled = resampled[resampled['bar_close_time'] < cutoff_ts].copy()
    except Exception as e:
        # Fallback: return empty DataFrame
        return pd.DataFrame()
    
    return resampled


def process_day_predictive(
    g: pd.DataFrame,
    historical_daily_data: List[Dict],
    events_df: Optional[pd.DataFrame] = None
) -> Optional[Dict]:
    """
    Process a single day and compute ALL predictive features.
    
    CRITICAL: All features use only data available before 9:30 AM ET on day D.
    Labels use day D's 45-minute window (09:30-10:15 ET) only.
    
    Args:
        g: 1-minute bars for day D and prior days (needed for HTF)
        historical_daily_data: List of dicts with prior days' RTH session data
        events_df: Events dataframe
    
    Returns:
        Dict with all predictive features and labels
    """
    tz_et = pytz.timezone("US/Eastern")
    g = g.sort_values("ts_et").reset_index(drop=True)
    
    # ========== LABELS: Compute from day D's 45-minute window only ==========
    # Get the latest date in g (this is the current day D we're processing)
    # g contains all data up to date D (needed for HTF features)
    date = g["et_date"].max()
    
    # Filter to only current day's data for 45-minute window labels
    current_day_mask = g["et_date"] == date
    current_day = g.loc[current_day_mask]
    
    # 45-minute window: 09:30 - 10:15 ET
    window_mask = _within(current_day["ts_et"], OPEN_START, OPEN_END)
    window = current_day.loc[window_mask]
    
    if window.empty:
        return None
    
    open_45 = float(window.iloc[0]["open"])
    close_45 = float(window.iloc[-1]["close"])
    high_45 = float(window["high"].max())
    low_45 = float(window["low"].min())
    range_45 = high_45 - low_45
    move_45 = abs(close_45 - open_45)
    
    # ATR_20 from prior days only (still using full RTH for historical context)
    if len(historical_daily_data) >= ATR_PERIOD:
        prior_ranges = pd.Series([d["ny_range"] for d in historical_daily_data[-ATR_PERIOD:]])
        ny_atr_20_prev = compute_atr_from_ranges(prior_ranges, ATR_PERIOD)
    else:
        ny_atr_20_prev = np.nan
    
    # New labels with 0.75 * ATR threshold
    BigMove45 = 0
    ExtremeMove45 = 0
    Dump45 = 0
    Pump45 = 0
    
    if not np.isnan(ny_atr_20_prev):
        atr_threshold = 0.75 * ny_atr_20_prev
        
        # BigMove45: absolute move >= 0.75 * ATR
        if move_45 >= atr_threshold:
            BigMove45 = 1
        
        # ExtremeMove45: absolute move >= 200 points
        if move_45 >= 200.0:
            ExtremeMove45 = 1
        
        # Dump45: close < open AND (open - low) >= 0.75 * ATR
        if close_45 < open_45:
            downside_move = open_45 - low_45
            if downside_move >= atr_threshold:
                Dump45 = 1
        
        # Pump45: close > open AND (high - open) >= 0.75 * ATR
        if close_45 > open_45:
            upside_move = high_45 - open_45
            if upside_move >= atr_threshold:
                Pump45 = 1
    
    # ========== FEATURE GROUP A: Prior Day/Week Structure ==========
    # All computed from day D-1 or earlier
    
    prev_day_data = historical_daily_data[-1] if len(historical_daily_data) >= 1 else None
    
    if prev_day_data:
        prev_day_return_pts = prev_day_data.get("ny_close", np.nan) - prev_day_data.get("ny_open", np.nan)
        prev_day_return_pct = (prev_day_return_pts / prev_day_data.get("ny_open", np.nan) * 100) if prev_day_data.get("ny_open", 0) > 0 else np.nan
        prev_day_range = prev_day_data.get("ny_range", np.nan)
        prev_day_close = prev_day_data.get("ny_close", np.nan)
        prev_day_open = prev_day_data.get("ny_open", np.nan)
        prev_day_high = prev_day_data.get("ny_high", np.nan)
        prev_day_low = prev_day_data.get("ny_low", np.nan)
        
        # prev_day_close_pos: position within day D-1 range
        if prev_day_range > 0:
            prev_day_close_pos = (prev_day_close - prev_day_low) / prev_day_range
        else:
            prev_day_close_pos = np.nan
        
        # prev_day_trend_intraday: SMA(50) vs SMA(200) on day D-1 RTH only
        prev_day_rth = g[(g["ts_et"].dt.date == date - timedelta(days=1)) & _within(g["ts_et"], RTH_START, RTH_END)]
        prev_day_trend_intraday = 0
        if not prev_day_rth.empty and len(prev_day_rth) >= 200:
            sma_50 = compute_sma(prev_day_rth["close"], 50)
            sma_200 = compute_sma(prev_day_rth["close"], 200)
            if len(sma_50) > 0 and len(sma_200) > 0:
                sma_50_val = sma_50.iloc[-1]
                sma_200_val = sma_200.iloc[-1]
                if not np.isnan(sma_50_val) and not np.isnan(sma_200_val):
                    prev_day_trend_intraday = 1 if sma_50_val > sma_200_val else -1
    else:
        prev_day_return_pts = np.nan
        prev_day_return_pct = np.nan
        prev_day_range = np.nan
        prev_day_close_pos = np.nan
        prev_day_trend_intraday = 0
        prev_day_close = np.nan
        prev_day_low = np.nan
        prev_day_high = np.nan
    
    # Rolling returns and ranges
    if len(historical_daily_data) >= 10:
        recent_10 = historical_daily_data[-10:]
        rolling_10d_return = sum([d.get("ny_close", 0) - d.get("ny_open", 0) for d in recent_10])
        rolling_10d_range_avg = np.mean([d.get("ny_range", 0) for d in recent_10])
    else:
        rolling_10d_return = np.nan
        rolling_10d_range_avg = np.nan
    
    if len(historical_daily_data) >= 5:
        recent_5 = historical_daily_data[-5:]
        rolling_5d_return = sum([d.get("ny_close", 0) - d.get("ny_open", 0) for d in recent_5])
        rolling_5d_range_avg = np.mean([d.get("ny_range", 0) for d in recent_5])
    else:
        rolling_5d_return = np.nan
        rolling_5d_range_avg = np.nan
    
    # from_20d_high: distance from 20-day high
    if len(historical_daily_data) >= 20:
        recent_20_closes = [d.get("ny_close", 0) for d in historical_daily_data[-20:]]
        max_close_20d = max(recent_20_closes)
        if prev_day_close > 0:
            from_20d_high = (prev_day_close - max_close_20d) / max_close_20d
        else:
            from_20d_high = np.nan
    else:
        from_20d_high = np.nan
    
    # week_range_pos: position within 5-day high/low range
    if len(historical_daily_data) >= 5:
        recent_5 = historical_daily_data[-5:]
        week_high = max([d.get("ny_high", 0) for d in recent_5])
        week_low = min([d.get("ny_low", float('inf')) for d in recent_5])
        if week_high > week_low and prev_day_close > 0:
            week_range_pos = (prev_day_close - week_low) / (week_high - week_low)
        else:
            week_range_pos = np.nan
    else:
        week_range_pos = np.nan
    
    # ========== FEATURE GROUP B: Overnight & Gap Features ==========
    # Overnight session: 18:00 prev day to 09:29:59 current day
    
    start_ts = (pd.Timestamp(date) - pd.Timedelta(days=1)).replace(
        hour=OVERNIGHT_START_HOUR, minute=0
    )
    start_ts = tz_et.localize(start_ts)
    end_ts = tz_et.localize(pd.Timestamp.combine(date, PRE_OPEN_CUTOFF))
    
    on_mask = (g["ts_et"] >= start_ts) & (g["ts_et"] < end_ts)
    overnight = g.loc[on_mask]
    
    if not overnight.empty:
        overnight_open = float(overnight.iloc[0]["open"])
        overnight_close = float(overnight.iloc[-1]["close"])
        overnight_high = float(overnight["high"].max())
        overnight_low = float(overnight["low"].min())
        overnight_range = overnight_high - overnight_low
        overnight_return = overnight_close - overnight_open
        overnight_direction = 1 if overnight_return > 0 else (-1 if overnight_return < 0 else 0)
    else:
        overnight_open = np.nan
        overnight_close = np.nan
        overnight_high = np.nan
        overnight_low = np.nan
        overnight_range = 0.0
        overnight_return = np.nan
        overnight_direction = 0
    
    # Pre-open price (last bar before 9:30)
    preopen_bars = g[g["ts_et"] < end_ts]
    if not preopen_bars.empty:
        preopen_price = float(preopen_bars.iloc[-1]["close"])
    else:
        preopen_price = open_45  # Fallback to 45-min open
    
    # Gap features
    gap_from_yclose = preopen_price - prev_day_close if not np.isnan(prev_day_close) else np.nan
    
    # ADR_10 from prior days
    if len(historical_daily_data) >= 10:
        adr_10 = np.mean([d.get("ny_range", 0) for d in historical_daily_data[-10:]])
    else:
        adr_10 = np.nan
    
    gap_from_yclose_normalized = gap_from_yclose / adr_10 if not np.isnan(adr_10) and adr_10 > 0 else np.nan
    
    # Open vs previous day levels
    open_vs_yday_low = preopen_price - prev_day_low if not np.isnan(prev_day_low) else np.nan
    open_below_prev_low = 1 if not np.isnan(prev_day_low) and preopen_price < prev_day_low else 0
    distance_to_yday_low = abs(preopen_price - prev_day_low) if not np.isnan(prev_day_low) else np.nan
    distance_to_yday_low_normalized = distance_to_yday_low / adr_10 if not np.isnan(adr_10) and adr_10 > 0 else np.nan
    
    # Overnight range vs 10-day average
    if len(historical_daily_data) >= 10:
        # Need to compute prior overnight ranges (stored in historical data)
        avg_overnight_range_10d = np.nan  # Will compute in rolling features
        overnight_range_vs_10d_avg = np.nan  # Will compute in rolling features
    else:
        avg_overnight_range_10d = np.nan
        overnight_range_vs_10d_avg = np.nan
    
    # ========== FEATURE GROUP C: Higher Timeframe Technicals ==========
    # Daily, 4H, 1H bars that closed before 09:30 ET on day D
    
    cutoff_ts = tz_et.localize(pd.Timestamp.combine(date, RTH_START))
    
    # Daily bars - compute from historical_daily_data for continuity across chunks
    daily_sma50_vs_sma200 = 0
    daily_close_vs_sma20_D1 = np.nan
    daily_rsi_14_D1 = np.nan
    
    if len(historical_daily_data) >= 14:
        # Extract daily closes from historical data
        daily_closes = pd.Series([d.get("ny_close", np.nan) for d in historical_daily_data])
        daily_closes = daily_closes.dropna()
        
        if len(daily_closes) >= 200:
            # Full SMA 50 vs 200
            sma_50 = compute_sma(daily_closes, 50)
            sma_200 = compute_sma(daily_closes, 200)
            if len(sma_50) > 0 and len(sma_200) > 0:
                sma_50_val = sma_50.iloc[-1]
                sma_200_val = sma_200.iloc[-1]
                if not np.isnan(sma_50_val) and not np.isnan(sma_200_val):
                    daily_sma50_vs_sma200 = 1 if sma_50_val > sma_200_val else -1
        elif len(daily_closes) >= 50:
            # Use shorter periods if we don't have 200 days
            sma_50 = compute_sma(daily_closes, 50)
            sma_20 = compute_sma(daily_closes, 20)
            if len(sma_50) > 0 and len(sma_20) > 0:
                sma_50_val = sma_50.iloc[-1]
                sma_20_val = sma_20.iloc[-1]
                if not np.isnan(sma_50_val) and not np.isnan(sma_20_val):
                    daily_sma50_vs_sma200 = 1 if sma_50_val > sma_20_val else -1
        
        if len(daily_closes) >= 20:
            sma_20 = compute_sma(daily_closes, 20)
            last_close = daily_closes.iloc[-1]
            last_sma20 = sma_20.iloc[-1]
            if not np.isnan(last_close) and not np.isnan(last_sma20) and last_sma20 > 0:
                daily_close_vs_sma20_D1 = (last_close - last_sma20) / last_sma20
        
        if len(daily_closes) >= 14:
            rsi = compute_rsi(daily_closes, 14)
            daily_rsi_14_D1 = float(rsi.iloc[-1]) if not rsi.empty and not np.isnan(rsi.iloc[-1]) else np.nan
    
    # 4H bars
    h4_bars = resample_to_timeframe(g, '4H', cutoff_ts)
    h4_close_prev = np.nan
    h4_rsi_14_prev = np.nan
    h4_kc_pos_prev = np.nan
    h4_above_upper_kc = 0
    h4_below_lower_kc = 0
    
    if not h4_bars.empty:
        h4_close_prev = float(h4_bars.iloc[-1]["close"])
        
        if len(h4_bars) >= 14:
            rsi = compute_rsi(h4_bars["close"], 14)
            h4_rsi_14_prev = float(rsi.iloc[-1]) if not rsi.empty else np.nan
        
        if len(h4_bars) >= 20:
            ema, upper, lower = compute_keltner_channels(h4_bars, 20, 10, 1.5)
            last_close = h4_bars.iloc[-1]["close"]
            last_upper = upper.iloc[-1]
            last_lower = lower.iloc[-1]
            last_ema = ema.iloc[-1]
            
            if not np.isnan(last_upper) and not np.isnan(last_lower) and last_upper > last_lower:
                h4_kc_pos_prev = (last_close - last_ema) / (last_upper - last_lower)
                h4_above_upper_kc = 1 if last_close > last_upper else 0
                h4_below_lower_kc = 1 if last_close < last_lower else 0
    
    # 1H bars
    h1_bars = resample_to_timeframe(g, '1H', cutoff_ts)
    h1_rsi_14_prev = np.nan
    h1_kc_pos_prev = np.nan
    
    if not h1_bars.empty and len(h1_bars) >= 14:
        rsi = compute_rsi(h1_bars["close"], 14)
        h1_rsi_14_prev = float(rsi.iloc[-1]) if not rsi.empty else np.nan
        
        if len(h1_bars) >= 20:
            ema, upper, lower = compute_keltner_channels(h1_bars, 20, 10, 1.5)
            last_close = h1_bars.iloc[-1]["close"]
            last_upper = upper.iloc[-1]
            last_lower = lower.iloc[-1]
            last_ema = ema.iloc[-1]
            
            if not np.isnan(last_upper) and not np.isnan(last_lower) and last_upper > last_lower:
                h1_kc_pos_prev = (last_close - last_ema) / (last_upper - last_lower)
    
    # ========== FEATURE GROUP D: Volatility & Sentiment ==========
    # ny_atr_20_prev already computed above
    atr_percentile_20d_prev = np.nan  # Will compute in rolling features
    
    # VIX features (if available - placeholder for now)
    vix_level_preopen = np.nan
    vix_change_1d = np.nan
    vix_change_5d = np.nan
    
    # ========== FEATURE GROUP E: Calendar & Events ==========
    day_of_week = pd.Timestamp(date).dayofweek  # 0=Monday, 4=Friday
    
    has_red_news_830 = 0
    event_count_preopen = 0
    
    if events_df is not None:
        day_events = events_df[events_df["date"] == date]
        if not day_events.empty:
            event_times = str(day_events.iloc[0]["event_times"]).lower()
            event_types = str(day_events.iloc[0]["event_types"]).lower()
            
            # Count pre-open events
            event_count_preopen = sum(1 for t in event_times.split(',') if '08:30' in t or '8:30' in t or 'pre' in str(day_events.iloc[0]["event_sessions"]).lower())
            
            # Red news at 8:30
            if "08:30" in event_times or "8:30" in event_times:
                if any(x in event_types for x in ["inflation", "employment", "cpi", "nfp"]):
                    has_red_news_830 = 1
    
    # ========== ASSEMBLE RESULT ==========
    result = {
        "date": date,
        "symbol": g["symbol"].iloc[0],
        
        # Labels (45-minute window outcomes)
        "open_45": open_45,
        "close_45": close_45,
        "high_45": high_45,
        "low_45": low_45,
        "range_45": range_45,
        "move_45": move_45,
        "ny_atr_20_prev": ny_atr_20_prev,
        "BigMove45": BigMove45,
        "ExtremeMove45": ExtremeMove45,
        "Dump45": Dump45,
        "Pump45": Pump45,
        
        # Group A: Prior day/week structure
        "prev_day_return_pts": prev_day_return_pts,
        "prev_day_return_pct": prev_day_return_pct,
        "prev_day_range": prev_day_range,
        "prev_day_trend_intraday": prev_day_trend_intraday,
        "prev_day_close_pos": prev_day_close_pos,
        "rolling_5d_return": rolling_5d_return,
        "rolling_10d_return": rolling_10d_return,
        "rolling_5d_range_avg": rolling_5d_range_avg,
        "rolling_10d_range_avg": rolling_10d_range_avg,
        "from_20d_high": from_20d_high,
        "week_range_pos": week_range_pos,
        
        # Group B: Overnight & gap
        "overnight_open": overnight_open,
        "overnight_close": overnight_close,
        "overnight_high": overnight_high,
        "overnight_low": overnight_low,
        "overnight_range": overnight_range,
        "overnight_return": overnight_return,
        "overnight_direction": overnight_direction,
        "overnight_range_vs_10d_avg": overnight_range_vs_10d_avg,
        "gap_from_yclose": gap_from_yclose,
        "gap_from_yclose_normalized": gap_from_yclose_normalized,
        "open_vs_yday_low": open_vs_yday_low,
        "open_below_prev_low": open_below_prev_low,
        "distance_to_yday_low": distance_to_yday_low,
        "distance_to_yday_low_normalized": distance_to_yday_low_normalized,
        "preopen_price": preopen_price,
        
        # Group C: Higher timeframe technicals
        "daily_sma50_vs_sma200": daily_sma50_vs_sma200,
        "daily_close_vs_sma20_D1": daily_close_vs_sma20_D1,
        "daily_rsi_14_D1": daily_rsi_14_D1,
        "h4_close_prev": h4_close_prev,
        "h4_rsi_14_prev": h4_rsi_14_prev,
        "h4_kc_pos_prev": h4_kc_pos_prev,
        "h4_above_upper_kc": h4_above_upper_kc,
        "h4_below_lower_kc": h4_below_lower_kc,
        "h1_rsi_14_prev": h1_rsi_14_prev,
        "h1_kc_pos_prev": h1_kc_pos_prev,
        
        # Group D: Volatility
        "atr_percentile_20d_prev": atr_percentile_20d_prev,
        "vix_level_preopen": vix_level_preopen,
        "vix_change_1d": vix_change_1d,
        "vix_change_5d": vix_change_5d,
        
        # Group E: Calendar & events
        "day_of_week": day_of_week,
        "has_red_news_830": has_red_news_830,
        "event_count_preopen": event_count_preopen,
        
        # Helper fields for rolling computation
        "adr_10": adr_10,
    }
    
    return result


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features that require rolling windows across days."""
    df = df.sort_values("date").reset_index(drop=True)
    
    # Overnight range vs 10-day average
    df["avg_overnight_range_10d"] = df["overnight_range"].rolling(window=10, min_periods=1).mean()
    df["overnight_range_vs_10d_avg"] = df["overnight_range"] / df["avg_overnight_range_10d"]
    
    # ATR percentile (rank within last 20 days)
    df["atr_percentile_20d_prev"] = df["ny_atr_20_prev"].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
    )
    
    return df


def compute_correlations(df: pd.DataFrame, target: str = "BigMove45") -> pd.DataFrame:
    """Compute correlations between features and target (BigMove45, ExtremeMove45, Dump45, or Pump45)."""
    # All feature columns (exclude labels and metadata)
    exclude_cols = [
        "date", "symbol", "open_45", "close_45", "high_45", "low_45", "range_45", "move_45",
        "BigMove45", "ExtremeMove45", "Dump45", "Pump45", "adr_10", "avg_overnight_range_10d"
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    correlations = []
    for col in feature_cols:
        if col in df.columns:
            valid_mask = df[[col, target]].notna().all(axis=1)
            if valid_mask.sum() > 10:  # Need at least 10 valid pairs
                corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, target])
                correlations.append({
                    "feature": col,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                    "n_samples": valid_mask.sum(),
                    "target": target,
                })
    
    corr_df = pd.DataFrame(correlations).sort_values("abs_correlation", ascending=False)
    return corr_df


def compute_conditional_probs(df: pd.DataFrame, target: str = "BigMove45") -> pd.DataFrame:
    """Compute conditional probabilities by feature bins."""
    exclude_cols = [
        "date", "symbol", "open_45", "close_45", "high_45", "low_45", "range_45", "move_45",
        "BigMove45", "ExtremeMove45", "Dump45", "Pump45", "adr_10", "avg_overnight_range_10d"
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    results = []
    baseline_prob = df[target].mean()
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        valid_df = df[[col, target]].dropna()
        if len(valid_df) < 20:
            continue
        
        # For binary features, use direct bins
        if col in ["open_below_prev_low", "has_red_news_830", "h4_above_upper_kc", "h4_below_lower_kc"]:
            bins = [0, 1]
            labels = ["No", "Yes"]
        elif col in ["overnight_direction", "prev_day_trend_intraday", "daily_sma50_vs_sma200"]:
            bins = [-1, 0, 1]
            labels = ["Down/Bearish", "Neutral", "Up/Bullish"]
        elif col == "day_of_week":
            bins = [0, 1, 2, 3, 4, 5]
            labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        else:
            # Use quartiles for continuous features
            try:
                quartiles = valid_df[col].quantile([0, 0.25, 0.5, 0.75, 1.0])
                bins = quartiles.values
                labels = ["Q1", "Q2", "Q3", "Q4"]
            except:
                continue
        
        # Compute probabilities for each bin
        if col in ["open_below_prev_low", "has_red_news_830", "h4_above_upper_kc", "h4_below_lower_kc",
                   "overnight_direction", "prev_day_trend_intraday", "daily_sma50_vs_sma200", "day_of_week"]:
            for bin_val, label in zip(bins, labels):
                if col == "day_of_week":
                    mask = valid_df[col] == bin_val
                else:
                    mask = valid_df[col] == bin_val
                
                if mask.sum() > 0:
                    prob = valid_df.loc[mask, target].mean()
                    count = mask.sum()
                    results.append({
                        "feature": col,
                        "bin": label,
                        "bin_min": bin_val,
                        "bin_max": bin_val,
                        "probability": prob,
                        "count": count,
                        "baseline_prob": baseline_prob,
                        "target": target,
                    })
        else:
            # Quartile bins
            for i in range(len(bins) - 1):
                mask = (valid_df[col] >= bins[i]) & (valid_df[col] < bins[i+1])
                if i == len(bins) - 2:  # Include upper bound for last bin
                    mask = (valid_df[col] >= bins[i]) & (valid_df[col] <= bins[i+1])
                
                if mask.sum() > 0:
                    prob = valid_df.loc[mask, target].mean()
                    count = mask.sum()
                    results.append({
                        "feature": col,
                        "bin": labels[i],
                        "bin_min": bins[i],
                        "bin_max": bins[i+1],
                        "probability": prob,
                        "count": count,
                        "baseline_prob": baseline_prob,
                        "target": target,
                    })
    
    return pd.DataFrame(results)


def fit_logistic_models(df: pd.DataFrame) -> Dict:
    """Fit logistic regression models for BigMove45, ExtremeMove45, Dump45, and Pump45."""
    if not HAS_SKLEARN:
        return {}
    
    # Select feature columns
    exclude_cols = [
        "date", "symbol", "open_45", "close_45", "high_45", "low_45", "range_45", "move_45",
        "BigMove45", "ExtremeMove45", "Dump45", "Pump45", "adr_10", "avg_overnight_range_10d"
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    results = {}
    
    for target in ["BigMove45", "ExtremeMove45", "Dump45", "Pump45"]:
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Drop rows with NaN
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) < 50:
            continue
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # Fit model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y_clean)
        
        # Predictions
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_clean, y_pred_proba)
        
        # Feature importances (coefficients)
        feature_importances = pd.DataFrame({
            "feature": feature_cols,
            "coefficient": model.coef_[0],
            "abs_coefficient": np.abs(model.coef_[0]),
        }).sort_values("abs_coefficient", ascending=False)
        
        results[target] = {
            "model": model,
            "scaler": scaler,
            "feature_importances": feature_importances,
            "auc": auc,
            "n_samples": len(X_clean),
            "n_features": len(feature_cols),
        }
    
    return results


def generate_hypotheses(df: pd.DataFrame, 
                        corr_df_bigmove: pd.DataFrame, corr_df_extreme: pd.DataFrame,
                        corr_df_dump: pd.DataFrame, corr_df_pump: pd.DataFrame,
                        cond_prob_df_bigmove: pd.DataFrame, cond_prob_df_extreme: pd.DataFrame,
                        cond_prob_df_dump: pd.DataFrame, cond_prob_df_pump: pd.DataFrame,
                        model_results: Dict) -> str:
    """Generate natural language hypotheses and pre-open checklist for 45-minute analysis."""
    baseline_bigmove = df["BigMove45"].mean()
    baseline_extreme = df["ExtremeMove45"].mean()
    baseline_dump = df["Dump45"].mean()
    baseline_pump = df["Pump45"].mean()
    
    lines = []
    lines.append("# 45-Minute Opening Move Predictive Analysis (NQ)\n\n")
    
    lines.append("## Sanity Check\n")
    lines.append(f"Unique dates: {df['date'].nunique()}\n")
    lines.append(f"BigMove45 baseline: {baseline_bigmove:.1%}\n")
    lines.append(f"ExtremeMove45 baseline: {baseline_extreme:.1%}\n")
    lines.append(f"Dump45 baseline: {baseline_dump:.1%}\n")
    lines.append(f"Pump45 baseline: {baseline_pump:.1%}\n\n")
    
    lines.append("---\n\n")
    
    # Top Predictors - BigMove45
    lines.append("## Top Predictors – BigMove45\n\n")
    top_corr_bigmove = corr_df_bigmove.head(10)
    for _, row in top_corr_bigmove.iterrows():
        feature = row["feature"]
        corr = row["correlation"]
        direction = "increases" if corr > 0 else "decreases"
        lines.append(f"- **{feature}**: Correlation = {corr:.3f} ({direction} BigMove45 probability)\n")
    
    # Top conditional probabilities for BigMove45
    cond_prob_bigmove_sorted = cond_prob_df_bigmove.sort_values("probability", ascending=False)
    lines.append("\n### Conditional Probabilities\n")
    for feature in cond_prob_bigmove_sorted["feature"].unique()[:5]:
        feature_data = cond_prob_bigmove_sorted[cond_prob_bigmove_sorted["feature"] == feature]
        top_bin = feature_data.iloc[0]
        prob = top_bin["probability"]
        bin_label = top_bin["bin"]
        count = top_bin["count"]
        baseline = top_bin["baseline_prob"]
        if prob > baseline * 1.3:
            lines.append(f"- **{feature}** = {bin_label}: {prob:.1%} BigMove45 probability (vs {baseline:.1%} baseline, n={count})\n")
    
    lines.append("\n---\n\n")
    
    # Top Predictors - Dump45
    lines.append("## Top Predictors – Dump45\n\n")
    top_corr_dump = corr_df_dump.head(10)
    for _, row in top_corr_dump.iterrows():
        feature = row["feature"]
        corr = row["correlation"]
        direction = "increases" if corr > 0 else "decreases"
        lines.append(f"- **{feature}**: Correlation = {corr:.3f} ({direction} Dump45 probability)\n")
    
    cond_prob_dump_sorted = cond_prob_df_dump.sort_values("probability", ascending=False)
    lines.append("\n### Conditional Probabilities\n")
    for feature in cond_prob_dump_sorted["feature"].unique()[:5]:
        feature_data = cond_prob_dump_sorted[cond_prob_dump_sorted["feature"] == feature]
        top_bin = feature_data.iloc[0]
        prob = top_bin["probability"]
        bin_label = top_bin["bin"]
        count = top_bin["count"]
        baseline = top_bin["baseline_prob"]
        if prob > baseline * 1.3:
            lines.append(f"- **{feature}** = {bin_label}: {prob:.1%} Dump45 probability (vs {baseline:.1%} baseline, n={count})\n")
    
    lines.append("\n---\n\n")
    
    # Top Predictors - Pump45
    lines.append("## Top Predictors – Pump45\n\n")
    top_corr_pump = corr_df_pump.head(10)
    for _, row in top_corr_pump.iterrows():
        feature = row["feature"]
        corr = row["correlation"]
        direction = "increases" if corr > 0 else "decreases"
        lines.append(f"- **{feature}**: Correlation = {corr:.3f} ({direction} Pump45 probability)\n")
    
    cond_prob_pump_sorted = cond_prob_df_pump.sort_values("probability", ascending=False)
    lines.append("\n### Conditional Probabilities\n")
    for feature in cond_prob_pump_sorted["feature"].unique()[:5]:
        feature_data = cond_prob_pump_sorted[cond_prob_pump_sorted["feature"] == feature]
        top_bin = feature_data.iloc[0]
        prob = top_bin["probability"]
        bin_label = top_bin["bin"]
        count = top_bin["count"]
        baseline = top_bin["baseline_prob"]
        if prob > baseline * 1.3:
            lines.append(f"- **{feature}** = {bin_label}: {prob:.1%} Pump45 probability (vs {baseline:.1%} baseline, n={count})\n")
    
    lines.append("\n---\n\n")
    
    # Observations
    lines.append("## Observations\n\n")
    
    # Generate natural language insights
    if not cond_prob_dump_sorted.empty:
        top_dump = cond_prob_dump_sorted.iloc[0]
        if top_dump["probability"] > baseline_dump * 1.3:
            lines.append(f"- When **{top_dump['feature']}** = {top_dump['bin']}, Dump45 probability rises from {baseline_dump:.1%} → {top_dump['probability']:.1%}.\n")
    
    if not cond_prob_pump_sorted.empty:
        top_pump = cond_prob_pump_sorted.iloc[0]
        if top_pump["probability"] > baseline_pump * 1.3:
            lines.append(f"- When **{top_pump['feature']}** = {top_pump['bin']}, Pump45 probability rises from {baseline_pump:.1%} → {top_pump['probability']:.1%}.\n")
    
    # ExtremeMove45 overlap with BigMove45
    extreme_overlap = df[(df["ExtremeMove45"] == 1) & (df["BigMove45"] == 1)].shape[0]
    extreme_total = df["ExtremeMove45"].sum()
    if extreme_total > 0:
        overlap_pct = (extreme_overlap / extreme_total) * 100
        lines.append(f"- ExtremeMove45 (≥ 200 pts) occurs {baseline_extreme:.1%} of days; {overlap_pct:.0f}% of those overlap with BigMove45.\n")
    
    lines.append("\n---\n\n")
    lines.append("## Notes\n\n")
    lines.append("All features computed using data available before 09:30 ET.\n")
    lines.append("Labels measured over 09:30 – 10:15 ET only.\n")
    
    return "".join(lines)


def main():
    """Main processing function."""
    print("Starting 45-Minute Opening Move Predictive Analysis...")
    print("⚠️  CRITICAL: All features use only data available before 9:30 AM ET")
    print("⚠️  Labels computed from 09:30-10:15 ET window only\n")
    
    tz_et = pytz.timezone("US/Eastern")
    residual = pd.DataFrame()
    events_df = _load_events()
    
    all_days_data = []
    historical_daily_data = []  # Store prior days' RTH session data
    
    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    dtypes = {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "symbol": "string",
    }
    
    print("Loading and processing data...")
    
    # Accumulate historical data across chunks
    all_historical_daily_data = []
    
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
        
        # Use accumulated historical data (keep more for daily SMAs)
        historical_daily_data = all_historical_daily_data[-250:] if len(all_historical_daily_data) > 250 else all_historical_daily_data.copy()
        
        # Process each day
        dates = sorted(to_proc["et_date"].unique())
        for d in tqdm(dates, desc=f"Chunk {chunk_num + 1} days"):
            # Get all data up to and including day d (needed for HTF bars)
            day_data = to_proc[to_proc["et_date"] <= d]
            if day_data.empty:
                continue
            
            # Get current day's data only for RTH session metrics
            current_day_data = to_proc[to_proc["et_date"] == d]
            if current_day_data.empty:
                continue
            
            # Process day with historical context
            result = process_day_predictive(day_data, historical_daily_data, events_df)
            if result:
                all_days_data.append(result)
                
                # Store current day's RTH session data for next day's features (still use full RTH for historical context)
                rth_mask = _within(current_day_data["ts_et"], RTH_START, RTH_END)
                rth = current_day_data.loc[rth_mask]
                if not rth.empty:
                    day_summary = {
                        "date": d,
                        "ny_open": float(rth.iloc[0]["open"]),
                        "ny_close": float(rth.iloc[-1]["close"]),
                        "ny_high": float(rth["high"].max()),
                        "ny_low": float(rth["low"].min()),
                        "ny_range": float(rth["high"].max() - rth["low"].min()),
                        "overnight_range": result["overnight_range"],
                    }
                    historical_daily_data.append(day_summary)
                    all_historical_daily_data.append(day_summary)
                    # Keep only last 250 days in memory for processing (for daily SMAs)
                    if len(historical_daily_data) > 250:
                        historical_daily_data = historical_daily_data[-250:]
    
    if not all_days_data:
        print("❌ No data processed!")
        return
    
    print(f"\n✅ Processed {len(all_days_data)} days")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_days_data)
    
    # Deduplicate: ensure one row per (date, symbol)
    print(f"\nBefore deduplication: {len(df)} rows")
    df = df.sort_values("date")
    df = df.groupby(["date", "symbol"], as_index=False).last()
    print(f"After deduplication: {len(df)} rows")
    
    # Compute rolling features
    print("\nComputing rolling features...")
    df = compute_rolling_features(df)
    
    # Save features
    df.to_csv(OUT_DIR / "dump_day_45min_predictive_features.csv", index=False)
    print(f"✅ Features saved to {OUT_DIR / 'dump_day_45min_predictive_features.csv'}")
    
    # Validation checks
    print("\n=== Dataset Validation ===")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"Total rows: {len(df)}")
    for col in ["BigMove45", "ExtremeMove45", "Dump45", "Pump45"]:
        rate = df[col].mean()
        print(f"{col} rate: {rate:.2%} ({df[col].sum()} days)")
    print(f"Rows per date (avg): {len(df) / df['date'].nunique():.2f}")
    
    # Compute correlations for all 4 targets
    print("\nComputing correlations...")
    corr_df_bigmove = compute_correlations(df, "BigMove45")
    corr_df_extreme = compute_correlations(df, "ExtremeMove45")
    corr_df_dump = compute_correlations(df, "Dump45")
    corr_df_pump = compute_correlations(df, "Pump45")
    
    # Combine correlations
    corr_combined = pd.concat([corr_df_bigmove, corr_df_extreme, corr_df_dump, corr_df_pump], ignore_index=True)
    corr_combined.to_csv(OUT_DIR / "dump_day_45min_predictive_correlations.csv", index=False)
    print(f"✅ Correlations saved to {OUT_DIR / 'dump_day_45min_predictive_correlations.csv'}")
    
    print("\nTop 10 Correlations for BigMove45:")
    print(corr_df_bigmove.head(10).to_string())
    print("\nTop 10 Correlations for ExtremeMove45:")
    print(corr_df_extreme.head(10).to_string())
    print("\nTop 10 Correlations for Dump45:")
    print(corr_df_dump.head(10).to_string())
    print("\nTop 10 Correlations for Pump45:")
    print(corr_df_pump.head(10).to_string())
    
    # Compute conditional probabilities for all 4 targets
    print("\nComputing conditional probabilities...")
    cond_prob_df_bigmove = compute_conditional_probs(df, "BigMove45")
    cond_prob_df_extreme = compute_conditional_probs(df, "ExtremeMove45")
    cond_prob_df_dump = compute_conditional_probs(df, "Dump45")
    cond_prob_df_pump = compute_conditional_probs(df, "Pump45")
    
    # Combine conditional probabilities
    cond_prob_combined = pd.concat([cond_prob_df_bigmove, cond_prob_df_extreme, cond_prob_df_dump, cond_prob_df_pump], ignore_index=True)
    cond_prob_combined.to_csv(OUT_DIR / "dump_day_45min_predictive_conditional_probs.csv", index=False)
    print(f"✅ Conditional probabilities saved to {OUT_DIR / 'dump_day_45min_predictive_conditional_probs.csv'}")
    
    # Fit models
    print("\nFitting logistic regression models...")
    model_results = fit_logistic_models(df)
    
    if model_results:
        # Save model results
        model_results_df = []
        for target, res in model_results.items():
            res["feature_importances"]["target"] = target
            model_results_df.append(res["feature_importances"])
        
        if model_results_df:
            model_results_combined = pd.concat(model_results_df, ignore_index=True)
            model_results_combined.to_csv(OUT_DIR / "dump_day_45min_predictive_model_results.csv", index=False)
            print(f"✅ Model results saved to {OUT_DIR / 'dump_day_45min_predictive_model_results.csv'}")
    
    # Generate hypotheses
    print("\nGenerating hypotheses...")
    hypotheses = generate_hypotheses(
        df, corr_df_bigmove, corr_df_extreme, corr_df_dump, corr_df_pump,
        cond_prob_df_bigmove, cond_prob_df_extreme, cond_prob_df_dump, cond_prob_df_pump,
        model_results
    )
    with open(OUT_DIR / "dump_day_45min_predictive_hypotheses.md", "w") as f:
        f.write(hypotheses)
    print(f"✅ Hypotheses saved to {OUT_DIR / 'dump_day_45min_predictive_hypotheses.md'}")
    print("\n" + "="*80)
    print(hypotheses)
    print("="*80)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

