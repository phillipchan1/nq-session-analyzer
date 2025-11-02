#!/usr/bin/env python3
"""
C1 Liquidity Sweeps Analysis
-----------------------------

Comprehensive statistical analysis of liquidity sweep behavior during the first 
15-minute candle (C1, 9:30-9:45 ET), tracking which liquidity levels get swept 
vs untouched, and predicting when untouched levels will be hit later in the session.
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

RTH_START, RTH_END = time(9, 30), time(16, 0)
C1_START, C1_END = time(9, 30), time(9, 45)
NY_OPEN, NY_CLOSE = time(9, 30), time(16, 0)
LONDON_START, LONDON_END = time(2, 0), time(5, 0)
ASIA_START, ASIA_END = time(18, 0), time(1, 0)  # Wraps around midnight
OVERNIGHT_START = time(18, 0)
PREMAKET_START = time(8, 0)
NY_KILLZONE_START = time(8, 30)
NY_KILLZONE_END = time(10, 0)

CHUNKSIZE = 1_000_000
VP_BINS = 100
VP_VALUE_AREA = 0.68  # 68% value area
EPSILON = 0.1  # Price tolerance for level touches

# =================== HELPER FUNCTIONS ===================

def _within(ts: pd.Series, start_t: time, end_t: time) -> pd.Series:
    """Check if timestamps fall within time range."""
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)


def _within_wrap(ts: pd.Series, start_t: time, end_t: time) -> pd.Series:
    """Handle wrapping intervals like 18:00-1:00."""
    t = ts.dt.time
    return (t >= start_t) | (t < end_t)


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


def _is_swing(high: pd.Series, low: pd.Series, k: int):
    """Swing highs/lows with width k."""
    sh = pd.Series(True, index=high.index)
    sl = pd.Series(True, index=low.index)
    for off in range(1, k+1):
        sh &= (high > high.shift(+off)) & (high > high.shift(-off))
        sl &= (low < low.shift(+off)) & (low < low.shift(-off))
    sh.iloc[:k] = sh.iloc[-k:] = False
    sl.iloc[:k] = sl.iloc[-k:] = False
    return sh.fillna(False), sl.fillna(False)


def _build_price_grid(low: float, high: float, bins: int) -> np.ndarray:
    """Build price grid for volume profile."""
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.linspace(low, low + 1e-6, num=max(2, bins+1))
    return np.linspace(low, high, num=bins+1)


def _distribute_bar_volume_to_bins(row, edges: np.ndarray, method: str) -> np.ndarray:
    """Distribute bar volume across price bins."""
    vol = row['volume']
    if vol <= 0 or not np.isfinite(vol):
        return np.zeros(len(edges)-1, dtype=float)

    lo = float(row['low'])
    hi = float(row['high'])
    bins = len(edges)-1
    out = np.zeros(bins, dtype=float)

    if method == "spread":
        if not np.isfinite(lo) or not np.isfinite(hi):
            return out
        if hi <= lo:
            px = float(row['close'])
            idx = np.searchsorted(edges, px, side='right') - 1
            if 0 <= idx < bins:
                out[idx] += vol
            return out
        vol_density = vol / max(1e-9, (hi - lo))
        i0 = max(0, np.searchsorted(edges, lo, side='right') - 1)
        i1 = min(bins-1, np.searchsorted(edges, hi, side='left'))
        for i in range(i0, i1+1):
            a = max(lo, edges[i])
            b = min(hi, edges[i+1])
            if b > a:
                out[i] += vol_density * (b - a)
        return out
    else:
        px = float(row['close'])
        idx = np.searchsorted(edges, px, side='right') - 1
        if 0 <= idx < bins:
            out[idx] += vol
        return out


def _compute_volume_profile(df: pd.DataFrame, bins: int, method: str = "spread") -> Tuple:
    """Compute volume profile from dataframe."""
    if df.empty:
        return None, None
    lo = float(df['low'].min())
    hi = float(df['high'].max())
    edges = _build_price_grid(lo, hi, bins)
    vol_bins = np.zeros(len(edges)-1, dtype=float)
    for _, row in df.iterrows():
        vol_bins += _distribute_bar_volume_to_bins(row, edges, method)
    return edges, vol_bins


def _value_area_from_profile(edges: np.ndarray, vol_bins: np.ndarray, va_fraction: float) -> Dict:
    """Calculate value area from volume profile."""
    if edges is None or vol_bins is None or len(vol_bins) == 0 or vol_bins.sum() <= 0:
        return {}
    bins = len(vol_bins)
    poc_idx = int(np.argmax(vol_bins))
    total = float(vol_bins.sum())

    used = np.zeros(bins, dtype=bool)
    used[poc_idx] = True
    cum = float(vol_bins[poc_idx])
    L = poc_idx - 1
    R = poc_idx + 1
    while cum / total < va_fraction and (L >= 0 or R < bins):
        left_val = vol_bins[L] if L >= 0 else -1.0
        right_val = vol_bins[R] if R < bins else -1.0
        if right_val > left_val:
            used[R] = True
            cum += right_val
            R += 1
        else:
            used[L] = True
            cum += left_val
            L -= 1

    poc = 0.5 * (edges[poc_idx] + edges[poc_idx+1])
    used_idx = np.where(used)[0]
    if len(used_idx) == 0:
        return {}
    val = edges[used_idx.min()]
    vah = edges[used_idx.max() + 1]
    return {'poc': float(poc), 'val': float(val), 'vah': float(vah)}


def _detect_fvg(df_tf: pd.DataFrame, timeframe: str) -> List[Dict]:
    """Detect Fair Value Gaps at given timeframe."""
    fvgs = []
    if len(df_tf) < 2:
        return fvgs
    
    for i in range(1, len(df_tf)):
        prev = df_tf.iloc[i-1]
        curr = df_tf.iloc[i]
        
        # Bullish FVG: gap up
        if curr["low"] > prev["high"]:
            gap_low = prev["high"]
            gap_high = curr["low"]
            gap_mid = (gap_low + gap_high) / 2.0
            
            # Check if gap fills in next 2 candles
            filled = False
            for j in range(i+1, min(i+3, len(df_tf))):
                if df_tf.iloc[j]["low"] <= prev["high"]:
                    filled = True
                    break
            
            if not filled:
                fvgs.append({
                    "type": f"{timeframe}_bull_fvg",
                    "level": gap_mid,
                    "side": "mid",
                    "origin_session": "mixed",
                    "created_ts": curr.get("ts_et", curr.get("candle_time", None)),
                    "fvg_low": gap_low,
                    "fvg_high": gap_high,
                })
        
        # Bearish FVG: gap down
        elif curr["high"] < prev["low"]:
            gap_low = curr["high"]
            gap_high = prev["low"]
            gap_mid = (gap_low + gap_high) / 2.0
            
            filled = False
            for j in range(i+1, min(i+3, len(df_tf))):
                if df_tf.iloc[j]["high"] >= prev["low"]:
                    filled = True
                    break
            
            if not filled:
                fvgs.append({
                    "type": f"{timeframe}_bear_fvg",
                    "level": gap_mid,
                    "side": "mid",
                    "origin_session": "mixed",
                    "created_ts": curr.get("ts_et", curr.get("candle_time", None)),
                    "fvg_low": gap_low,
                    "fvg_high": gap_high,
                })
    
    return fvgs


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample dataframe to given timeframe."""
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    out = df.set_index('ts_et').resample(rule).agg(agg).dropna(subset=['open', 'high', 'low', 'close'])
    out['ts_et'] = out.index
    return out.reset_index(drop=True)


def _compute_atr(df: pd.DataFrame, n: int = 15) -> pd.Series:
    """Compute ATR."""
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum((df['high'] - df['close'].shift()).abs(),
                               (df['low'] - df['close'].shift()).abs()))
    return tr.rolling(n, min_periods=n).mean()


# =================== LIQUIDITY LEVEL BUILDING ===================

def build_all_liquidity_levels(df_1m: pd.DataFrame, current_date, prev_day_data: Optional[pd.DataFrame] = None,
                                historical_data: Optional[Dict] = None) -> List[Dict]:
    """
    Build all liquidity levels for a trading day.
    
    Returns list of dictionaries with keys: type, level, side, origin_session, created_ts
    """
    levels = []
    
    # Current day data
    current_day_mask = df_1m["et_date"] == current_date
    current_day = df_1m[current_day_mask].copy()
    
    # Previous day data
    if prev_day_data is not None and not prev_day_data.empty:
        prev_date = prev_day_data["et_date"].iloc[0]
        prev_day = prev_day_data.copy()
    else:
        prev_date = None
        prev_day = pd.DataFrame()
    
    # Get 9:30 open price
    open_930 = current_day[_within(current_day["ts_et"], RTH_START, time(9, 31))]
    if open_930.empty:
        return levels
    open_price = open_930.iloc[0]["open"]
    
    # ===== SESSION-BASED LEVELS =====
    
    # Previous-day high/low
    if not prev_day.empty:
        prev_day_high = prev_day["high"].max()
        prev_day_low = prev_day["low"].min()
        levels.append({
            "type": "prev_day_high",
            "level": prev_day_high,
            "side": "high",
            "origin_session": "prev_day",
            "created_ts": prev_day["ts_et"].iloc[-1] if len(prev_day) > 0 else None,
        })
        levels.append({
            "type": "prev_day_low",
            "level": prev_day_low,
            "side": "low",
            "origin_session": "prev_day",
            "created_ts": prev_day["ts_et"].iloc[-1] if len(prev_day) > 0 else None,
        })
        
        # Previous RTH high/low (9:30-16:00 of previous day)
        prev_rth = prev_day[_within(prev_day["ts_et"], RTH_START, RTH_END)]
        if not prev_rth.empty:
            prev_rth_high = prev_rth["high"].max()
            prev_rth_low = prev_rth["low"].min()
            levels.append({
                "type": "prev_rth_high",
                "level": prev_rth_high,
                "side": "high",
                "origin_session": "prev_rth",
                "created_ts": prev_rth["ts_et"].iloc[-1],
            })
            levels.append({
                "type": "prev_rth_low",
                "level": prev_rth_low,
                "side": "low",
                "origin_session": "prev_rth",
                "created_ts": prev_rth["ts_et"].iloc[-1],
            })
            
            # Daily midpoint
            daily_midpoint = (prev_rth_high + prev_rth_low) / 2.0
            levels.append({
                "type": "daily_midpoint",
                "level": daily_midpoint,
                "side": "mid",
                "origin_session": "prev_rth",
                "created_ts": prev_rth["ts_et"].iloc[-1],
            })
    
    # Overnight (Globex) high/low: 18:00-9:29 ET
    overnight_mask = _within_wrap(current_day["ts_et"], OVERNIGHT_START, RTH_START)
    overnight_day = current_day[overnight_mask].copy()
    
    # Also check previous day post-close
    if not prev_day.empty:
        prev_overnight = prev_day[_within_wrap(prev_day["ts_et"], OVERNIGHT_START, RTH_START)]
        if not prev_overnight.empty:
            overnight_day = pd.concat([prev_overnight, overnight_day], ignore_index=True).sort_values("ts_et")
    
    if not overnight_day.empty:
        overnight_high = overnight_day["high"].max()
        overnight_low = overnight_day["low"].min()
        levels.append({
            "type": "overnight_high",
            "level": overnight_high,
            "side": "high",
            "origin_session": "overnight",
            "created_ts": overnight_day["ts_et"].iloc[-1],
        })
        levels.append({
            "type": "overnight_low",
            "level": overnight_low,
            "side": "low",
            "origin_session": "overnight",
            "created_ts": overnight_day["ts_et"].iloc[-1],
        })
    
    # Premarket high/low: 08:00-9:29 ET
    premarket_mask = _within(current_day["ts_et"], PREMAKET_START, RTH_START)
    premarket = current_day[premarket_mask].copy()
    if not premarket.empty:
        premarket_high = premarket["high"].max()
        premarket_low = premarket["low"].min()
        levels.append({
            "type": "premarket_high",
            "level": premarket_high,
            "side": "high",
            "origin_session": "premarket",
            "created_ts": premarket["ts_et"].iloc[-1],
        })
        levels.append({
            "type": "premarket_low",
            "level": premarket_low,
            "side": "low",
            "origin_session": "premarket",
            "created_ts": premarket["ts_et"].iloc[-1],
        })
    
    # Asian session high/low: 18:00-1:00 ET (wrap around)
    asia_mask = _within_wrap(current_day["ts_et"], ASIA_START, ASIA_END)
    asia_day = current_day[asia_mask].copy()
    if not prev_day.empty:
        prev_asia = prev_day[_within_wrap(prev_day["ts_et"], ASIA_START, ASIA_END)]
        if not prev_asia.empty:
            asia_day = pd.concat([prev_asia, asia_day], ignore_index=True).sort_values("ts_et")
    
    if not asia_day.empty:
        asia_high = asia_day["high"].max()
        asia_low = asia_day["low"].min()
        levels.append({
            "type": "asia_high",
            "level": asia_high,
            "side": "high",
            "origin_session": "asia",
            "created_ts": asia_day["ts_et"].iloc[-1],
        })
        levels.append({
            "type": "asia_low",
            "level": asia_low,
            "side": "low",
            "origin_session": "asia",
            "created_ts": asia_day["ts_et"].iloc[-1],
        })
    
    # London session high/low: 2:00-5:00 ET
    london_mask = _within(current_day["ts_et"], LONDON_START, LONDON_END)
    london = current_day[london_mask].copy()
    if not london.empty:
        london_high = london["high"].max()
        london_low = london["low"].min()
        london_close = london["close"].iloc[-1]
        levels.append({
            "type": "london_high",
            "level": london_high,
            "side": "high",
            "origin_session": "london",
            "created_ts": london["ts_et"].iloc[-1],
        })
        levels.append({
            "type": "london_low",
            "level": london_low,
            "side": "low",
            "origin_session": "london",
            "created_ts": london["ts_et"].iloc[-1],
        })
        levels.append({
            "type": "london_close",
            "level": london_close,
            "side": "mid",
            "origin_session": "london",
            "created_ts": london["ts_et"].iloc[-1],
        })
    
    # NY kill-zone high/low: 8:30-10:00 ET
    killzone_mask = _within(current_day["ts_et"], NY_KILLZONE_START, NY_KILLZONE_END)
    killzone = current_day[killzone_mask].copy()
    if not killzone.empty:
        killzone_high = killzone["high"].max()
        killzone_low = killzone["low"].min()
        levels.append({
            "type": "ny_killzone_high",
            "level": killzone_high,
            "side": "high",
            "origin_session": "ny_killzone",
            "created_ts": killzone["ts_et"].iloc[-1],
        })
        levels.append({
            "type": "ny_killzone_low",
            "level": killzone_low,
            "side": "low",
            "origin_session": "ny_killzone",
            "created_ts": killzone["ts_et"].iloc[-1],
        })
    
    # Current-day 00:00 open
    midnight_mask = current_day["ts_et"].dt.time == time(0, 0)
    midnight = current_day[midnight_mask]
    if not midnight.empty:
        midnight_open = midnight.iloc[0]["open"]
        levels.append({
            "type": "current_day_00_open",
            "level": midnight_open,
            "side": "mid",
            "origin_session": "current_day",
            "created_ts": midnight.iloc[0]["ts_et"],
        })
    
    # ===== TIMEFRAME-BASED LEVELS =====
    
    # Previous 1-hour candle high/low
    before_930 = current_day[current_day["ts_et"].dt.time < RTH_START]
    if not before_930.empty:
        before_930 = before_930.sort_values("ts_et")
        before_930["hour"] = before_930["ts_et"].dt.floor("1h")
        last_hour = before_930["hour"].iloc[-1]
        last_hour_bars = before_930[before_930["hour"] == last_hour]
        if not last_hour_bars.empty:
            levels.append({
                "type": "prev_1h_high",
                "level": last_hour_bars["high"].max(),
                "side": "high",
                "origin_session": "pre_rth",
                "created_ts": last_hour_bars["ts_et"].iloc[-1],
            })
            levels.append({
                "type": "prev_1h_low",
                "level": last_hour_bars["low"].min(),
                "side": "low",
                "origin_session": "pre_rth",
                "created_ts": last_hour_bars["ts_et"].iloc[-1],
            })
    
    # Previous 4-hour candle high/low
    if not before_930.empty:
        before_930["4h"] = before_930["ts_et"].dt.floor("4h")
        last_4h = before_930["4h"].iloc[-1]
        last_4h_bars = before_930[before_930["4h"] == last_4h]
        if not last_4h_bars.empty:
            levels.append({
                "type": "prev_4h_high",
                "level": last_4h_bars["high"].max(),
                "side": "high",
                "origin_session": "pre_rth",
                "created_ts": last_4h_bars["ts_et"].iloc[-1],
            })
            levels.append({
                "type": "prev_4h_low",
                "level": last_4h_bars["low"].min(),
                "side": "low",
                "origin_session": "pre_rth",
                "created_ts": last_4h_bars["ts_et"].iloc[-1],
            })
    
    # Current daily/weekly opens/highs/lows (running)
    # Daily: from 00:00 to current moment
    daily_mask = current_day["ts_et"].dt.date == current_date
    daily_up_to_now = current_day[daily_mask].copy()
    if not daily_up_to_now.empty:
        levels.append({
            "type": "current_daily_high",
            "level": daily_up_to_now["high"].max(),
            "side": "high",
            "origin_session": "current_day",
            "created_ts": daily_up_to_now["ts_et"].iloc[-1],
        })
        levels.append({
            "type": "current_daily_low",
            "level": daily_up_to_now["low"].min(),
            "side": "low",
            "origin_session": "current_day",
            "created_ts": daily_up_to_now["ts_et"].iloc[-1],
        })
        levels.append({
            "type": "current_daily_open",
            "level": daily_up_to_now["open"].iloc[0],
            "side": "mid",
            "origin_session": "current_day",
            "created_ts": daily_up_to_now["ts_et"].iloc[0],
        })
    
    # Weekly: trading week (Mon-Fri)
    # Get start of week (Monday)
    current_day_obj = pd.Timestamp(current_date)
    days_since_monday = current_day_obj.weekday()
    week_start = current_day_obj - timedelta(days=days_since_monday)
    
    # Get all data from week start
    week_mask = df_1m["et_date"] >= week_start.date()
    week_mask &= df_1m["et_date"] <= current_date
    week_data = df_1m[week_mask].copy()
    if not week_data.empty:
        levels.append({
            "type": "current_weekly_high",
            "level": week_data["high"].max(),
            "side": "high",
            "origin_session": "weekly",
            "created_ts": week_data["ts_et"].iloc[-1],
        })
        levels.append({
            "type": "current_weekly_low",
            "level": week_data["low"].min(),
            "side": "low",
            "origin_session": "weekly",
            "created_ts": week_data["ts_et"].iloc[-1],
        })
        levels.append({
            "type": "current_weekly_open",
            "level": week_data["open"].iloc[0],
            "side": "mid",
            "origin_session": "weekly",
            "created_ts": week_data["ts_et"].iloc[0],
        })
    
    # ===== SWING LEVELS =====
    
    # Nearest 1-minute swing highs/lows before 9:30
    before_930 = current_day[current_day["ts_et"].dt.time < RTH_START].copy()
    if not before_930.empty and len(before_930) >= 6:
        before_930 = before_930.sort_values("ts_et")
        sh, sl = _is_swing(before_930["high"], before_930["low"], k=3)
        swing_highs = before_930[sh].sort_values("high", ascending=False)
        swing_lows = before_930[sl].sort_values("low", ascending=True)
        
        if not swing_highs.empty:
            nearest_high = swing_highs.iloc[-1]  # Last swing high before 9:30
            levels.append({
                "type": "nearest_1m_swing_high",
                "level": nearest_high["high"],
                "side": "high",
                "origin_session": "pre_rth",
                "created_ts": nearest_high["ts_et"],
            })
        if not swing_lows.empty:
            nearest_low = swing_lows.iloc[-1]
            levels.append({
                "type": "nearest_1m_swing_low",
                "level": nearest_low["low"],
                "side": "low",
                "origin_session": "pre_rth",
                "created_ts": nearest_low["ts_et"],
            })
    
    # Nearest 5-minute swing highs/lows
    if not before_930.empty:
        before_930_5m = _resample_ohlc(before_930, "5min")
        if len(before_930_5m) >= 6:
            sh, sl = _is_swing(before_930_5m["high"], before_930_5m["low"], k=2)
            swing_highs = before_930_5m[sh].sort_values("high", ascending=False)
            swing_lows = before_930_5m[sl].sort_values("low", ascending=True)
            
            if not swing_highs.empty:
                nearest_high = swing_highs.iloc[-1]
                levels.append({
                    "type": "nearest_5m_swing_high",
                    "level": nearest_high["high"],
                    "side": "high",
                    "origin_session": "pre_rth",
                    "created_ts": nearest_high["ts_et"],
                })
            if not swing_lows.empty:
                nearest_low = swing_lows.iloc[-1]
                levels.append({
                    "type": "nearest_5m_swing_low",
                    "level": nearest_low["low"],
                    "side": "low",
                    "origin_session": "pre_rth",
                    "created_ts": nearest_low["ts_et"],
                })
    
    # Nearest 15-minute swing highs/lows
    if not before_930.empty:
        before_930_15m = _resample_ohlc(before_930, "15min")
        if len(before_930_15m) >= 4:
            sh, sl = _is_swing(before_930_15m["high"], before_930_15m["low"], k=2)
            swing_highs = before_930_15m[sh].sort_values("high", ascending=False)
            swing_lows = before_930_15m[sl].sort_values("low", ascending=True)
            
            if not swing_highs.empty:
                nearest_high = swing_highs.iloc[-1]
                levels.append({
                    "type": "nearest_15m_swing_high",
                    "level": nearest_high["high"],
                    "side": "high",
                    "origin_session": "pre_rth",
                    "created_ts": nearest_high["ts_et"],
                })
            if not swing_lows.empty:
                nearest_low = swing_lows.iloc[-1]
                levels.append({
                    "type": "nearest_15m_swing_low",
                    "level": nearest_low["low"],
                    "side": "low",
                    "origin_session": "pre_rth",
                    "created_ts": nearest_low["ts_et"],
                })
    
    # ===== FVG LEVELS =====
    
    # Detect FVGs at 1min, 5min, 15min before 9:30
    if not before_930.empty:
        # 1-minute FVGs
        fvgs_1m = _detect_fvg(before_930, "1m")
        for fvg in fvgs_1m:
            levels.append({
                "type": fvg["type"],
                "level": fvg["level"],
                "side": fvg["side"],
                "origin_session": fvg["origin_session"],
                "created_ts": fvg["created_ts"],
            })
        
        # 5-minute FVGs
        before_930_5m = _resample_ohlc(before_930, "5min")
        if len(before_930_5m) >= 2:
            fvgs_5m = _detect_fvg(before_930_5m, "5m")
            for fvg in fvgs_5m:
                levels.append({
                    "type": fvg["type"],
                    "level": fvg["level"],
                    "side": fvg["side"],
                    "origin_session": fvg["origin_session"],
                    "created_ts": fvg["created_ts"],
                })
        
        # 15-minute FVGs
        before_930_15m = _resample_ohlc(before_930, "15min")
        if len(before_930_15m) >= 2:
            fvgs_15m = _detect_fvg(before_930_15m, "15m")
            for fvg in fvgs_15m:
                levels.append({
                    "type": fvg["type"],
                    "level": fvg["level"],
                    "side": fvg["side"],
                    "origin_session": fvg["origin_session"],
                    "created_ts": fvg["created_ts"],
                })
    
    # ===== VOLUME PROFILE LEVELS =====
    
    # Previous day VPOC, value-area high/low
    if not prev_day.empty:
        prev_rth = prev_day[_within(prev_day["ts_et"], RTH_START, RTH_END)]
        if not prev_rth.empty:
            edges, vol_bins = _compute_volume_profile(prev_rth, VP_BINS, "spread")
            va = _value_area_from_profile(edges, vol_bins, VP_VALUE_AREA)
            if va:
                levels.append({
                    "type": "prev_day_vpoc",
                    "level": va["poc"],
                    "side": "mid",
                    "origin_session": "prev_rth",
                    "created_ts": prev_rth["ts_et"].iloc[-1],
                })
                levels.append({
                    "type": "prev_day_vah",
                    "level": va["vah"],
                    "side": "high",
                    "origin_session": "prev_rth",
                    "created_ts": prev_rth["ts_et"].iloc[-1],
                })
                levels.append({
                    "type": "prev_day_val",
                    "level": va["val"],
                    "side": "low",
                    "origin_session": "prev_rth",
                    "created_ts": prev_rth["ts_et"].iloc[-1],
                })
    
    # Overnight VPOC
    if not overnight_day.empty:
        edges, vol_bins = _compute_volume_profile(overnight_day, VP_BINS, "spread")
        va = _value_area_from_profile(edges, vol_bins, VP_VALUE_AREA)
        if va:
            levels.append({
                "type": "overnight_vpoc",
                "level": va["poc"],
                "side": "mid",
                "origin_session": "overnight",
                "created_ts": overnight_day["ts_et"].iloc[-1],
            })
            levels.append({
                "type": "overnight_vah",
                "level": va["vah"],
                "side": "high",
                "origin_session": "overnight",
                "created_ts": overnight_day["ts_et"].iloc[-1],
            })
            levels.append({
                "type": "overnight_val",
                "level": va["val"],
                "side": "low",
                "origin_session": "overnight",
                "created_ts": overnight_day["ts_et"].iloc[-1],
            })
    
    # Composite 5-day POC
    if historical_data is not None and "last_5_days" in historical_data:
        last_5_days_data = historical_data["last_5_days"]
        if not last_5_days_data.empty:
            # Use only RTH data for composite POC
            last_5_days_rth = last_5_days_data[_within(last_5_days_data["ts_et"], RTH_START, RTH_END)]
            if not last_5_days_rth.empty:
                edges, vol_bins = _compute_volume_profile(last_5_days_rth, VP_BINS, "spread")
                va = _value_area_from_profile(edges, vol_bins, VP_VALUE_AREA)
                if va:
                    levels.append({
                        "type": "composite_5day_poc",
                        "level": va["poc"],
                        "side": "mid",
                        "origin_session": "composite",
                        "created_ts": last_5_days_rth["ts_et"].iloc[-1],
                    })
    
    return levels


# =================== PHASE A: C1 SWEEP STATISTICS ===================

def analyze_c1_sweeps(df_1m: pd.DataFrame, liquidity_levels: List[Dict], c1_end_time: time) -> List[Dict]:
    """
    Analyze which liquidity levels were swept by C1.
    
    Returns list of sweep results for each level.
    """
    results = []
    
    # Get C1 data (9:30-9:45)
    c1_mask = _within(df_1m["ts_et"], C1_START, c1_end_time)
    c1_data = df_1m[c1_mask].copy()
    
    if c1_data.empty:
        return results
    
    c1_high = c1_data["high"].max()
    c1_low = c1_data["low"].min()
    c1_open = c1_data["open"].iloc[0]
    c1_close = c1_data["close"].iloc[-1]
    
    # Track which side was swept first (high or low)
    high_sweep_times = []
    low_sweep_times = []
    
    # For each level, check if it was swept
    for level_info in liquidity_levels:
        level_type = level_info["type"]
        level = level_info["level"]
        side = level_info["side"]
        
        swept_by_c1 = False
        sweep_side = None
        points_beyond = 0.0
        
        # Check if swept based on side
        if side == "high":
            if c1_high > level:
                swept_by_c1 = True
                sweep_side = "high"
                points_beyond = c1_high - level
                # Find when it was swept
                sweep_mask = c1_data["high"] > level
                if sweep_mask.any():
                    first_sweep_idx = c1_data[sweep_mask].index[0]
                    sweep_time = c1_data.loc[first_sweep_idx, "ts_et"]
                    high_sweep_times.append((sweep_time, level_type))
        elif side == "low":
            if c1_low < level:
                swept_by_c1 = True
                sweep_side = "low"
                points_beyond = level - c1_low
                # Find when it was swept
                sweep_mask = c1_data["low"] < level
                if sweep_mask.any():
                    first_sweep_idx = c1_data[sweep_mask].index[0]
                    sweep_time = c1_data.loc[first_sweep_idx, "ts_et"]
                    low_sweep_times.append((sweep_time, level_type))
        elif side == "mid":
            # For mid levels (like POC), check if price went through
            if c1_high > level and c1_low < level:
                swept_by_c1 = True
                sweep_side = "both"
                points_beyond = min(c1_high - level, level - c1_low)
                # Determine which side was hit first
                high_hit = (c1_data["high"] > level).any()
                low_hit = (c1_data["low"] < level).any()
                if high_hit and low_hit:
                    high_idx = c1_data[c1_data["high"] > level].index[0]
                    low_idx = c1_data[c1_data["low"] < level].index[0]
                    if high_idx < low_idx:
                        sweep_side = "high"
                        high_sweep_times.append((c1_data.loc[high_idx, "ts_et"], level_type))
                    else:
                        sweep_side = "low"
                        low_sweep_times.append((c1_data.loc[low_idx, "ts_et"], level_type))
                elif high_hit:
                    sweep_side = "high"
                    high_idx = c1_data[c1_data["high"] > level].index[0]
                    high_sweep_times.append((c1_data.loc[high_idx, "ts_et"], level_type))
                elif low_hit:
                    sweep_side = "low"
                    low_idx = c1_data[c1_data["low"] < level].index[0]
                    low_sweep_times.append((c1_data.loc[low_idx, "ts_et"], level_type))
        
        results.append({
            "liquidity_type": level_type,
            "level": level,
            "side": side,
            "swept_by_c1": swept_by_c1,
            "sweep_side": sweep_side,
            "points_beyond": points_beyond,
            "c1_high": c1_high,
            "c1_low": c1_low,
            "c1_open": c1_open,
            "c1_close": c1_close,
            "swept_first": False,  # Will be set below
        })
    
    # Determine which side was swept first
    all_sweep_times = []
    for ts, lt in high_sweep_times:
        all_sweep_times.append((ts, lt, "high"))
    for ts, lt in low_sweep_times:
        all_sweep_times.append((ts, lt, "low"))
    
    if all_sweep_times:
        all_sweep_times.sort(key=lambda x: x[0])
        first_sweep_type = all_sweep_times[0][1]
        first_sweep_side = all_sweep_times[0][2]
        
        # Mark first swept level
        for r in results:
            if r["liquidity_type"] == first_sweep_type and r["sweep_side"] == first_sweep_side:
                r["swept_first"] = True
    
    return results


# =================== PHASE B: UNTOUCHED LIQUIDITY TRACKING ===================

def track_untouched_levels(df_1m: pd.DataFrame, liquidity_levels: List[Dict], 
                          c1_sweep_results: List[Dict], c1_end_time: time, 
                          session_end_time: time) -> List[Dict]:
    """
    Track which untouched levels get hit later in the session.
    """
    results = []
    
    # Get post-C1 data
    post_c1_mask = _within(df_1m["ts_et"], c1_end_time, session_end_time)
    post_c1_data = df_1m[post_c1_mask].copy()
    
    if post_c1_data.empty:
        return results
    
    # Identify untouched levels
    untouched_levels = []
    for sweep_result in c1_sweep_results:
        if not sweep_result["swept_by_c1"]:
            untouched_levels.append({
                "liquidity_type": sweep_result["liquidity_type"],
                "level": sweep_result["level"],
                "side": sweep_result["side"],
            })
    
    # Track touches
    touch_order = 0
    for level_info in untouched_levels:
        level_type = level_info["liquidity_type"]
        level = level_info["level"]
        side = level_info["side"]
        
        touched_later_in_session = False
        first_touch_timestamp = None
        first_touch_minutes_after_open = None
        touch_order_num = None
        continued_after_touch = None
        
        # Check if level was touched
        if side == "high":
            touch_mask = post_c1_data["high"] >= level - EPSILON
            if touch_mask.any():
                touched_later_in_session = True
                first_touch_idx = post_c1_data[touch_mask].index[0]
                first_touch_timestamp = post_c1_data.loc[first_touch_idx, "ts_et"]
                # Calculate minutes after 9:30 open
                open_time = df_1m[_within(df_1m["ts_et"], C1_START, time(9, 31))]["ts_et"].iloc[0]
                first_touch_minutes_after_open = (first_touch_timestamp - open_time).total_seconds() / 60.0
                touch_order += 1
                touch_order_num = touch_order
                
                # Check if continued after touch
                after_touch = post_c1_data[post_c1_data.index >= first_touch_idx]
                if len(after_touch) > 0:
                    max_after_touch = after_touch["high"].max()
                    continued_after_touch = max_after_touch > level + EPSILON
        
        elif side == "low":
            touch_mask = post_c1_data["low"] <= level + EPSILON
            if touch_mask.any():
                touched_later_in_session = True
                first_touch_idx = post_c1_data[touch_mask].index[0]
                first_touch_timestamp = post_c1_data.loc[first_touch_idx, "ts_et"]
                open_time = df_1m[_within(df_1m["ts_et"], C1_START, time(9, 31))]["ts_et"].iloc[0]
                first_touch_minutes_after_open = (first_touch_timestamp - open_time).total_seconds() / 60.0
                touch_order += 1
                touch_order_num = touch_order
                
                # Check if continued after touch
                after_touch = post_c1_data[post_c1_data.index >= first_touch_idx]
                if len(after_touch) > 0:
                    min_after_touch = after_touch["low"].min()
                    continued_after_touch = min_after_touch < level - EPSILON
        
        elif side == "mid":
            # For mid levels, check if price went through
            high_touch = (post_c1_data["high"] >= level - EPSILON).any()
            low_touch = (post_c1_data["low"] <= level + EPSILON).any()
            if high_touch or low_touch:
                touched_later_in_session = True
                if high_touch and low_touch:
                    high_idx = post_c1_data[post_c1_data["high"] >= level - EPSILON].index[0]
                    low_idx = post_c1_data[post_c1_data["low"] <= level + EPSILON].index[0]
                    first_touch_idx = min(high_idx, low_idx)
                elif high_touch:
                    first_touch_idx = post_c1_data[post_c1_data["high"] >= level - EPSILON].index[0]
                else:
                    first_touch_idx = post_c1_data[post_c1_data["low"] <= level + EPSILON].index[0]
                
                first_touch_timestamp = post_c1_data.loc[first_touch_idx, "ts_et"]
                open_time = df_1m[_within(df_1m["ts_et"], C1_START, time(9, 31))]["ts_et"].iloc[0]
                first_touch_minutes_after_open = (first_touch_timestamp - open_time).total_seconds() / 60.0
                touch_order += 1
                touch_order_num = touch_order
        
        results.append({
            "liquidity_type": level_type,
            "level": level,
            "side": side,  # Add side information
            "untouched_after_c1": True,
            "touched_later_in_session": touched_later_in_session,
            "first_touch_timestamp": first_touch_timestamp,
            "first_touch_minutes_after_open": first_touch_minutes_after_open,
            "touch_order": touch_order_num,
            "continued_after_touch": continued_after_touch,
        })
    
    return results


# =================== PHASE C: CONFLUENCE & SEQUENCE ANALYSIS ===================

def analyze_confluence_and_sequences(c1_sweep_results: List[Dict], df_15m: pd.DataFrame, 
                                     daily_close: float, daily_bias: str) -> Dict:
    """
    Analyze confluence patterns and sequences.
    """
    # Count swept vs untouched
    num_levels_swept = sum(1 for r in c1_sweep_results if r["swept_by_c1"])
    num_levels_untouched = sum(1 for r in c1_sweep_results if not r["swept_by_c1"])
    
    # Identify which side was swept first
    first_swept = next((r for r in c1_sweep_results if r.get("swept_first", False)), None)
    first_side_swept = None
    if first_swept:
        first_side_swept = first_swept["sweep_side"]
    
    # Get C2 and C3 directions
    if len(df_15m) >= 2:
        c2_direction = "bullish" if df_15m.iloc[1]["close"] > df_15m.iloc[1]["open"] else "bearish"
    else:
        c2_direction = None
    
    if len(df_15m) >= 3:
        c3_direction = "bullish" if df_15m.iloc[2]["close"] > df_15m.iloc[2]["open"] else "bearish"
    else:
        c3_direction = None
    
    # Determine pattern type (continuation vs reversal)
    c1_direction = "bullish" if df_15m.iloc[0]["close"] > df_15m.iloc[0]["open"] else "bearish"
    pattern_type = "continuation"
    if first_side_swept:
        if (first_side_swept == "low" and c1_direction == "bearish") or \
           (first_side_swept == "high" and c1_direction == "bullish"):
            pattern_type = "continuation"
        else:
            pattern_type = "reversal"
    
    # Find swept combinations
    swept_types = [r["liquidity_type"] for r in c1_sweep_results if r["swept_by_c1"]]
    swept_combinations = []
    
    # Common combinations
    if "overnight_low" in swept_types and "london_low" in swept_types:
        swept_combinations.append("overnight_low+london_low")
    if "overnight_high" in swept_types and "london_high" in swept_types:
        swept_combinations.append("overnight_high+london_high")
    if "prev_day_low" in swept_types:
        swept_combinations.append("prev_day_low")
    if "prev_day_high" in swept_types:
        swept_combinations.append("prev_day_high")
    
    return {
        "num_levels_swept": num_levels_swept,
        "num_levels_untouched": num_levels_untouched,
        "swept_combinations": ",".join(swept_combinations) if swept_combinations else None,
        "first_side_swept": first_side_swept,
        "c2_direction": c2_direction,
        "c3_direction": c3_direction,
        "daily_bias": daily_bias,
        "pattern_type": pattern_type,
    }


# =================== PHASE D: CONTEXT SPLITS ===================

def calculate_daily_sma(daily_closes: Dict, current_date, periods: List[int]) -> Dict[int, float]:
    """Calculate daily SMAs from accumulated daily closes."""
    smas = {}
    sorted_dates = sorted([d for d in daily_closes.keys() if d <= current_date])
    
    for p in periods:
        if len(sorted_dates) >= p:
            recent_dates = sorted_dates[-p:]
            recent_closes = [daily_closes[d] for d in recent_dates]
            smas[p] = np.mean(recent_closes)
        else:
            smas[p] = np.nan
    
    return smas


def determine_trend_vs_range(df_1m: pd.DataFrame) -> str:
    """Classify day as trend or range based on price action."""
    rth = df_1m[_within(df_1m["ts_et"], RTH_START, RTH_END)].copy()
    if rth.empty:
        return "unknown"
    
    session_range = rth["high"].max() - rth["low"].min()
    open_price = rth["open"].iloc[0]
    close_price = rth["close"].iloc[-1]
    net_move = abs(close_price - open_price)
    
    # If net move is > 50% of range, it's a trend day
    if session_range > 0:
        trend_ratio = net_move / session_range
        return "trend" if trend_ratio > 0.5 else "range"
    
    return "unknown"


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
    agg["direction"] = np.where(agg["close"] > agg["open"], "bullish", "bearish")
    
    return agg


# =================== OUTPUT GENERATION ===================

def generate_output_tables(all_c1_sweeps: List[Dict], all_untouched: List[Dict],
                          all_confluence: List[Dict], all_context: List[Dict]) -> None:
    """Generate output tables."""
    
    # Table A - Top Liquidity Levels Swept During C1
    df_sweeps = pd.DataFrame(all_c1_sweeps)
    if not df_sweeps.empty:
        # Aggregate by liquidity type
        sweep_stats = df_sweeps.groupby("liquidity_type").agg({
            "swept_by_c1": ["sum", "count", "mean"],
            "points_beyond": ["mean", "median"],
        }).reset_index()
        
        sweep_stats.columns = ["liquidity_type", "sweep_count", "total_count", "sweep_frequency_pct",
                               "avg_points_beyond", "median_points_beyond"]
        sweep_stats["sweep_frequency_pct"] = sweep_stats["sweep_frequency_pct"] * 100
        
        # Calculate continuation probability (merge with daily context)
        df_context = pd.DataFrame(all_context)
        if not df_context.empty:
            df_context["date"] = pd.to_datetime(df_context["date"])
            df_sweeps["date"] = pd.to_datetime(df_sweeps["date"])
            
            # Merge to get daily bias
            swept_with_context = df_sweeps.merge(df_context[["date", "daily_bias"]], on="date", how="left")
            
            # For each liquidity type that was swept first, calculate continuation probability
            if "swept_first" in swept_with_context.columns:
                swept_first = swept_with_context[swept_with_context["swept_first"] == True]
                if not swept_first.empty:
                    continuation_stats = swept_first.groupby("liquidity_type").agg({
                        "daily_bias": lambda x: (x == "bullish").mean() if len(x) > 0 else 0,
                    }).reset_index()
                    continuation_stats.columns = ["liquidity_type", "continuation_probability"]
                    continuation_stats["continuation_probability"] = continuation_stats["continuation_probability"] * 100
                    
                    sweep_stats = sweep_stats.merge(continuation_stats, on="liquidity_type", how="left")
                    sweep_stats["continuation_probability"] = sweep_stats["continuation_probability"].fillna(0)
                else:
                    sweep_stats["continuation_probability"] = 0
            else:
                sweep_stats["continuation_probability"] = 0
        
        sweep_stats = sweep_stats.sort_values("sweep_frequency_pct", ascending=False)
        sweep_stats.to_csv(OUT_DIR / "c1_sweep_statistics.csv", index=False)
        print(f"✅ Wrote c1_sweep_statistics.csv ({len(sweep_stats)} rows)")
    
    # Table B - Untouched Levels Hit Later
    df_untouched = pd.DataFrame(all_untouched)
    if not df_untouched.empty:
        untouched_stats = df_untouched.groupby("liquidity_type").agg({
            "touched_later_in_session": ["sum", "count", "mean"],
            "first_touch_minutes_after_open": ["mean", "median"],
            "continued_after_touch": "mean",
        }).reset_index()
        
        untouched_stats.columns = ["liquidity_type", "hit_count", "total_count", "hit_probability_pct",
                                   "avg_time_to_hit_minutes", "median_time_to_hit_minutes",
                                   "continued_after_touch_pct"]
        untouched_stats["hit_probability_pct"] = untouched_stats["hit_probability_pct"] * 100
        untouched_stats["continued_after_touch_pct"] = untouched_stats["continued_after_touch_pct"] * 100
        
        untouched_stats = untouched_stats.sort_values("hit_probability_pct", ascending=False)
        untouched_stats.to_csv(OUT_DIR / "untouched_levels_tracking.csv", index=False)
        print(f"✅ Wrote untouched_levels_tracking.csv ({len(untouched_stats)} rows)")
    
    # Table C - Confluence Sequences
    df_confluence = pd.DataFrame(all_confluence)
    if not df_confluence.empty:
        df_confluence.to_csv(OUT_DIR / "confluence_sequences.csv", index=False)
        print(f"✅ Wrote confluence_sequences.csv ({len(df_confluence)} rows)")
    
    # Context splits
    df_context = pd.DataFrame(all_context)
    if not df_context.empty:
        df_context.to_csv(OUT_DIR / "context_splits.csv", index=False)
        print(f"✅ Wrote context_splits.csv ({len(df_context)} rows)")
    
    # Conditional Matrix (Table C)
    if not df_sweeps.empty and not df_untouched.empty and not df_confluence.empty:
        conditional_matrix = []
        
        # Merge dataframes
        df_sweeps["date"] = pd.to_datetime(df_sweeps["date"])
        df_untouched["date"] = pd.to_datetime(df_untouched["date"])
        df_confluence["date"] = pd.to_datetime(df_confluence["date"])
        df_context["date"] = pd.to_datetime(df_context["date"])
        
        # For each day, build scenario
        for date in df_confluence["date"].unique():
            day_sweeps = df_sweeps[df_sweeps["date"] == date]
            day_untouched = df_untouched[df_untouched["date"] == date]
            day_confluence = df_confluence[df_confluence["date"] == date].iloc[0]
            day_context_rows = df_context[df_context["date"] == date]
            day_context = day_context_rows.iloc[0] if len(day_context_rows) > 0 else None
            
            # Build scenario string
            swept_lows = day_sweeps[(day_sweeps["swept_by_c1"]) & (day_sweeps["side"] == "low")]["liquidity_type"].tolist()
            swept_highs = day_sweeps[(day_sweeps["swept_by_c1"]) & (day_sweeps["side"] == "high")]["liquidity_type"].tolist()
            untouched_lows = day_untouched[day_untouched["side"] == "low"]["liquidity_type"].tolist()
            untouched_highs = day_untouched[day_untouched["side"] == "high"]["liquidity_type"].tolist()
            
            # Create scenario
            scenario_parts = []
            if swept_lows:
                scenario_parts.append(f"swept_low:{','.join(swept_lows[:3])}")  # Top 3
            if swept_highs:
                scenario_parts.append(f"swept_high:{','.join(swept_highs[:3])}")
            if untouched_lows:
                scenario_parts.append(f"untouched_low:{','.join(untouched_lows[:3])}")
            if untouched_highs:
                scenario_parts.append(f"untouched_high:{','.join(untouched_highs[:3])}")
            
            scenario = "|".join(scenario_parts) if scenario_parts else "none"
            
            # Check if untouched highs were hit later
            untouched_highs_hit = day_untouched[
                (day_untouched["side"] == "high") & 
                (day_untouched["touched_later_in_session"] == True)
            ]
            hit_by_1130 = untouched_highs_hit[
                untouched_highs_hit["first_touch_minutes_after_open"] <= 120  # 11:30 ET = 120 min after 9:30
            ]
            
            daily_close_up = day_context["daily_bias"] == "bullish" if day_context is not None else None
            
            conditional_matrix.append({
                "date": date,
                "scenario": scenario,
                "num_swept_lows": len(swept_lows),
                "num_swept_highs": len(swept_highs),
                "num_untouched_lows": len(untouched_lows),
                "num_untouched_highs": len(untouched_highs),
                "untouched_high_hit_later": len(untouched_highs_hit) > 0,
                "untouched_high_hit_by_1130": len(hit_by_1130) > 0,
                "day_closed_up": daily_close_up,
                "first_side_swept": day_confluence.get("first_side_swept"),
                "pattern_type": day_confluence.get("pattern_type"),
            })
        
        if conditional_matrix:
            df_conditional = pd.DataFrame(conditional_matrix)
            
            # Aggregate by scenario
            scenario_stats = df_conditional.groupby("scenario").agg({
                "untouched_high_hit_later": "mean",
                "untouched_high_hit_by_1130": "mean",
                "day_closed_up": "mean",
            }).reset_index()
            scenario_stats.columns = ["scenario", "pct_untouched_high_hit_later", 
                                     "pct_untouched_high_hit_by_1130", "pct_day_closed_up"]
            scenario_stats["pct_untouched_high_hit_later"] *= 100
            scenario_stats["pct_untouched_high_hit_by_1130"] *= 100
            scenario_stats["pct_day_closed_up"] *= 100
            
            df_conditional.to_csv(OUT_DIR / "conditional_matrix.csv", index=False)
            scenario_stats.to_csv(OUT_DIR / "conditional_matrix_summary.csv", index=False)
            print(f"✅ Wrote conditional_matrix.csv ({len(df_conditional)} rows)")
            print(f"✅ Wrote conditional_matrix_summary.csv ({len(scenario_stats)} rows)")
    
    # Summary tables (Table A/B/C format)
    summary_rows = []
    
    # Table A summary
    if not df_sweeps.empty:
        for _, row in sweep_stats.head(20).iterrows():
            summary_rows.append({
                "table": "A",
                "liquidity_type": row["liquidity_type"],
                "sweep_frequency_pct": row["sweep_frequency_pct"],
                "avg_points_beyond": row["avg_points_beyond"],
                "median_points_beyond": row["median_points_beyond"],
                "continuation_probability": row.get("continuation_probability", 0),
            })
    
    # Table B summary
    if not df_untouched.empty:
        for _, row in untouched_stats.head(20).iterrows():
            summary_rows.append({
                "table": "B",
                "liquidity_type": row["liquidity_type"],
                "hit_probability_pct": row["hit_probability_pct"],
                "avg_time_to_hit_minutes": row["avg_time_to_hit_minutes"],
                "continued_after_touch_pct": row["continued_after_touch_pct"],
            })
    
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(OUT_DIR / "summary_tables.csv", index=False)
        print(f"✅ Wrote summary_tables.csv ({len(df_summary)} rows)")


# =================== MAIN PROCESSING ===================

def main():
    """Main processing function."""
    print("Starting C1 Liquidity Sweeps Analysis...")
    
    tz_et = pytz.timezone("US/Eastern")
    residual = pd.DataFrame()
    
    # Storage
    all_c1_sweeps = []
    all_untouched = []
    all_confluence = []
    all_context = []
    
    daily_closes = {}
    historical_data = {}  # For storing last 5 days data
    prev_day_cache = {}  # Cache previous day data across chunks
    all_atr_values = []  # Collect all ATR values for global median
    
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
            
            # Get last 5 days for composite POC
            last_5_days = []
            for i in range(1, 6):
                check_date = d - timedelta(days=i)
                if check_date in prev_day_cache:
                    last_5_days.append(prev_day_cache[check_date])
                elif check_date in dates:
                    check_mask = to_proc["et_date"] == check_date
                    if check_mask.any():
                        last_5_days.append(to_proc[check_mask].copy())
            
            if last_5_days:
                historical_data["last_5_days"] = pd.concat(last_5_days, ignore_index=True)
            else:
                historical_data["last_5_days"] = pd.DataFrame()
            
            # Build liquidity levels
            liquidity_levels = build_all_liquidity_levels(day_data, d, prev_day_data, historical_data)
            
            if not liquidity_levels:
                continue
            
            # Analyze C1 sweeps
            c1_sweep_results = analyze_c1_sweeps(day_data, liquidity_levels, C1_END)
            
            # Add date to sweep results
            for result in c1_sweep_results:
                result["date"] = d
            all_c1_sweeps.extend(c1_sweep_results)
            
            # Track untouched levels
            untouched_results = track_untouched_levels(day_data, liquidity_levels, c1_sweep_results,
                                                      C1_END, RTH_END)
            for result in untouched_results:
                result["date"] = d
            all_untouched.extend(untouched_results)
            
            # Confluence analysis
            df_15m = aggregate_15m_candles(day_data[_within(day_data["ts_et"], RTH_START, RTH_END)])
            if not df_15m.empty:
                daily_close = rth.iloc[-1]["close"]
                smas = calculate_daily_sma(daily_closes, d, [50, 200])
                daily_bias = "bullish" if (not pd.isna(smas[50]) and not pd.isna(smas[200]) and 
                                          smas[50] > smas[200]) else "bearish"
                
                confluence_result = analyze_confluence_and_sequences(c1_sweep_results, df_15m, 
                                                                    daily_close, daily_bias)
                confluence_result["date"] = d
                all_confluence.append(confluence_result)
                
                # Context splits
                atr_series = _compute_atr(day_data, 15)
                current_atr = atr_series.iloc[-1] if len(atr_series) > 0 and not atr_series.isna().iloc[-1] else np.nan
                if not pd.isna(current_atr):
                    all_atr_values.append(current_atr)
                # Will calculate volatility_regime after collecting all ATR values
                volatility_regime = "unknown"  # Set later
                
                trend_vs_range = determine_trend_vs_range(day_data)
                day_of_week = pd.Timestamp(d).weekday()  # 0=Monday, 4=Friday
                
                all_context.append({
                    "date": d,
                    "daily_bias": daily_bias,
                    "volatility_regime": volatility_regime,
                    "trend_vs_range": trend_vs_range,
                    "day_of_week": day_of_week,
                    "sma50": smas[50],
                    "sma200": smas[200],
                    "atr_value": current_atr,
                })
            
            # Cache current day for next iteration
            prev_day_cache[d] = day_data.copy()
    
    # Process final residual
    if not residual.empty:
        residual = _determine_front_month(residual)
        dates = sorted(residual["et_date"].unique())
        for d in tqdm(dates, desc="Final residual"):
            day_data = residual[residual["et_date"] == d].copy()
            day_data = day_data.sort_values("ts_et").reset_index(drop=True)
            
            rth_mask = _within(day_data["ts_et"], RTH_START, RTH_END)
            rth = day_data.loc[rth_mask]
            if not rth.empty:
                daily_closes[d] = rth.iloc[-1]["close"]
            
            prev_date = d - timedelta(days=1)
            prev_day_data = None
            if prev_date in prev_day_cache:
                prev_day_data = prev_day_cache[prev_date]
            
            liquidity_levels = build_all_liquidity_levels(day_data, d, prev_day_data, historical_data)
            if liquidity_levels:
                c1_sweep_results = analyze_c1_sweeps(day_data, liquidity_levels, C1_END)
                for result in c1_sweep_results:
                    result["date"] = d
                all_c1_sweeps.extend(c1_sweep_results)
                
                untouched_results = track_untouched_levels(day_data, liquidity_levels, c1_sweep_results,
                                                          C1_END, RTH_END)
                for result in untouched_results:
                    result["date"] = d
                all_untouched.extend(untouched_results)
                
                df_15m = aggregate_15m_candles(day_data[_within(day_data["ts_et"], RTH_START, RTH_END)])
                if not df_15m.empty:
                    daily_close = rth.iloc[-1]["close"]
                    smas = calculate_daily_sma(daily_closes, d, [50, 200])
                    daily_bias = "bullish" if (not pd.isna(smas[50]) and not pd.isna(smas[200]) and 
                                              smas[50] > smas[200]) else "bearish"
                    
                    confluence_result = analyze_confluence_and_sequences(c1_sweep_results, df_15m,
                                                                        daily_close, daily_bias)
                    confluence_result["date"] = d
                    all_confluence.append(confluence_result)
                    
                    # Context splits
                    atr_series = _compute_atr(day_data, 15)
                    current_atr = atr_series.iloc[-1] if len(atr_series) > 0 and not atr_series.isna().iloc[-1] else np.nan
                    if not pd.isna(current_atr):
                        all_atr_values.append(current_atr)
                    volatility_regime = "unknown"  # Set later
                    
                    trend_vs_range = determine_trend_vs_range(day_data)
                    day_of_week = pd.Timestamp(d).weekday()
                    
                    all_context.append({
                        "date": d,
                        "daily_bias": daily_bias,
                        "volatility_regime": volatility_regime,
                        "trend_vs_range": trend_vs_range,
                        "day_of_week": day_of_week,
                        "sma50": smas[50],
                        "sma200": smas[200],
                        "atr_value": current_atr,
                    })
            
            # Cache current day
            prev_day_cache[d] = day_data.copy()
    
    if not all_c1_sweeps:
        print("No data processed!")
        return
    
    print(f"\nProcessed {len(set(r['date'] for r in all_c1_sweeps))} days")
    
    # Calculate global ATR median and update volatility regimes
    if all_atr_values:
        atr_median = np.median(all_atr_values)
        for ctx in all_context:
            if not pd.isna(ctx.get("atr_value")):
                ctx["volatility_regime"] = "high" if ctx["atr_value"] > atr_median else "low"
            else:
                ctx["volatility_regime"] = "unknown"
    
    # Generate output tables
    print("\nGenerating output tables...")
    generate_output_tables(all_c1_sweeps, all_untouched, all_confluence, all_context)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()


