#!/usr/bin/env python3
"""
Reversal Power Analysis (1-minute OHLCV)
----------------------------------------
Quantifies reversal power for many liquidity frameworks (ICT + institutional)
using 1m OHLCV data. Logs per-touch events and aggregates stats to rank
which level types most often produce strong, sustained reversals after being hit.

INPUT
-----
CSV with columns: ts_event, open, high, low, close, volume, symbol
- ts_event must be UTC (ISO datetime).

WHAT IT DOES
------------
Per day & symbol:
1) Builds candidate levels:
   - Sessions: Asia/London H/L (swing-filtered)
   - IB(60) H/L/Mid (you can switch to OR(30/45) if you only trade early)
   - ICT-ish: FVGs (1m/5m/15m), unmitigated time gaps (15m/60m/240m)
   - Institutional: VWAP ±σ bands (dynamic intraday), SMA50/200 (9:30 snapshot),
                    Bollinger(20,2σ) & Keltner(EMA20,ATR10*1.5) (9:30 snapshot)
   - Volume Profile (Overnight 21:00–09:30 ET): POC, VAH, VAL
   - High-Timeframe (last completed bars): H1/H4/D1/W1 highs & lows
2) Finds the first touch during NY session windows (9:30–10:15 slices).
3) Computes reversal metrics after touch (default 30m):
   - MFE_opp (away), MAE_thru (through), ReversalRatio, DisplacementATR, RetestWithin30m, TimeToPeak
4) Aggregates per level_type and ranks by composite rev_power_score (robust z-scores).
5) Confluence “matrix”: for each touch, tags **pairwise combinations** of levels within
   a proximity threshold (e.g., ±10pts), and writes combo summary tables with
   success rates (e.g., dispATR ≥ 1.0) and typical reversal distances (median / q75 / q90 MFE).

OUTPUT
------
- reversal_events.csv           : one row per touch event
- reversal_summary.csv          : ranked leaderboard by level_type
- reversal_combo_{X}pt.csv      : combo tables for each proximity X in CONF_THRESH_POINTS

Author: ChatGPT
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import time, timedelta, datetime

import numpy as np
import pandas as pd
import pytz

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20250926.ohlcv-1m.csv")
OUT_EVENTS = str(OUT_DIR / "reversal_events.csv")
OUT_SUMMARY = str(OUT_DIR / "reversal_summary.csv")

WEEKDAYS_ONLY = True
CHUNKSIZE = 1_000_000
SYMBOL_REGEX = re.compile(r"^NQ[A-Z]\d{1,2}$|^NQ1!?$|^NQ=F$|^MNQ[A-Z]\d{1,2}$", re.IGNORECASE)

# Sessions (ET)
ASIA_START, ASIA_END     = time(21,0), time(3,0)   # 9pm-3am (wraps midnight)
LONDON_START, LONDON_END = time(3,0), time(8,0)    # 3am-8am
NY_OPEN, NY_CLOSE        = time(9,30), time(16,0)  # 9:30am-4pm

# Analysis windows (NY morning)
WINDOWS = [
    (time(9,30), time(9,45), "9:30-9:45"),
    (time(9,45), time(10,0), "9:45-10:00"),
    (time(10,0), time(10,15), "10:00-10:15"),
    (time(9,30), time(10,15), "9:30-10:15"),
]

# Swing detection for London highs/lows
SWING_K = 2

# Reversal metrics
ATR_LEN = 14
REV_WINDOW_MIN = 30
RETEST_WINDOW_MIN = 30
DISP_THRESHOLDS = [0.5, 1.0]

# VWAP stdev band
VWAP_STDEV_LEN = 60
VWAP_BANDS = [1.0, 2.0]

# Bollinger / Keltner (snapshot at 9:30 ET)
BB_LEN = 20; BB_STD = 2.0
KELT_EMA = 20; KELT_ATR = 10; KELT_MULT = 1.5

# -------- Volume Profile (Overnight) --------
VP_BINS = 200           # number of price bins
VP_VALUE_AREA = 0.70    # 70% value area
VP_METHOD = "spread"    # "spread" distributes vol over bar's high-low; "close" uses close-only

# -------- Confluence / combos --------
CONF_THRESH_POINTS = [10]   # proximity thresholds in points for combo tagging; add 5,15 if desired
# ============================================


# ------------- Helpers -------------
def within(ts: pd.Series, start_t: time, end_t: time) -> pd.Series:
    """Boolean mask for times within [start, end) on a tz-aware ts series (ET).
       Handles simple non-wrapping intervals. For wrapping (e.g., 21:00-03:00), call twice."""
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)

def within_wrap(ts: pd.Series, start_t: time, end_t: time) -> pd.Series:
    """Handles wrapping intervals like 21:00–03:00."""
    t = ts.dt.time
    return (t >= start_t) | (t < end_t)

def is_swing(high: pd.Series, low: pd.Series, k: int):
    """Swing highs/lows with width k (True where swing occurs)."""
    sh = pd.Series(True, index=high.index)
    sl = pd.Series(True, index=low.index)
    for off in range(1, k+1):
        sh &= (high > high.shift(+off)) & (high > high.shift(-off))
        sl &= (low  <  low.shift(+off)) & (low  <  low.shift(-off))
    sh.iloc[:k] = sh.iloc[-k:] = False
    sl.iloc[:k] = sl.iloc[-k:] = False
    return sh.fillna(False), sl.fillna(False)

def compute_atr(df: pd.DataFrame, n: int) -> pd.Series:
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum((df['high'] - df['close'].shift()).abs(),
                               (df['low'] - df['close'].shift()).abs()))
    return tr.rolling(n, min_periods=n).mean()

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    out = df.resample(rule, on='ts_et').agg(agg).dropna(subset=['open','high','low','close'])
    out['ts_et'] = out.index
    return out.reset_index(drop=True)

def find_fvgs(df_tf: pd.DataFrame) -> List[Tuple[str, float, pd.Timestamp]]:
    """Classic 3-candle FVG detection; returns tuples (type, level, created_ts)."""
    out = []
    for i in range(2, len(df_tf)):
        h_prev = df_tf.loc[i-2, 'high']
        l_prev = df_tf.loc[i-2, 'low']
        if df_tf.loc[i-1, 'low'] > h_prev:
            mid = (df_tf.loc[i-1, 'low'] + h_prev) / 2.0
            out.append(('bull_fvg', mid, df_tf.loc[i, 'ts_et']))
        if df_tf.loc[i-1, 'high'] < l_prev:
            mid = (df_tf.loc[i-1, 'high'] + l_prev) / 2.0
            out.append(('bear_fvg', mid, df_tf.loc[i, 'ts_et']))
    return out

def find_time_gaps(df: pd.DataFrame, gap_minutes: int):
    """Find time gaps of >= gap_minutes between consecutive 1m bars."""
    df_sorted = df.sort_values('ts_et').reset_index(drop=True)
    out = []
    for i in range(len(df_sorted)-1):
        cur = df_sorted.iloc[i]; nxt = df_sorted.iloc[i+1]
        dtm = (nxt['ts_et'] - cur['ts_et']).total_seconds()/60.0
        if dtm >= gap_minutes:
            gap_high = cur['high']; gap_low = nxt['low']
            if gap_high < gap_low:
                out.append((f'gap_up_{gap_minutes}m', float(gap_low), nxt['ts_et']))
            elif gap_low > gap_high:
                out.append((f'gap_down_{gap_minutes}m', float(gap_high), nxt['ts_et']))
    return out

def session_name(ts: pd.Timestamp) -> str:
    t = ts.time()
    if ASIA_START <= t or t < ASIA_END:
        return "asia"
    elif LONDON_START <= t < LONDON_END:
        return "london"
    elif NY_OPEN <= t < NY_CLOSE:
        return "ny"
    else:
        return "after_hours"


# ----- Volume Profile helpers -----
def _build_price_grid(low: float, high: float, bins: int) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.linspace(low, low + 1e-6, num=max(2, bins+1))
    return np.linspace(low, high, num=bins+1)

def _distribute_bar_volume_to_bins(row, edges: np.ndarray, method: str) -> np.ndarray:
    vol = row['volume']
    if vol <= 0 or not np.isfinite(vol):
        return np.zeros(len(edges)-1, dtype=float)

    lo = float(row['low']); hi = float(row['high'])
    bins = len(edges)-1
    out = np.zeros(bins, dtype=float)

    if method == "spread":
        if not np.isfinite(lo) or not np.isfinite(hi):
            return out
        if hi <= lo:   # point-ish bar: allocate to close bin
            px = float(row['close'])
            idx = np.searchsorted(edges, px, side='right') - 1
            if 0 <= idx < bins:
                out[idx] += vol
            return out
        vol_density = vol / max(1e-9, (hi - lo))
        i0 = max(0, np.searchsorted(edges, lo, side='right') - 1)
        i1 = min(bins-1, np.searchsorted(edges, hi, side='left'))
        for i in range(i0, i1+1):
            a = max(lo, edges[i]); b = min(hi, edges[i+1])
            if b > a:
                out[i] += vol_density * (b - a)
        return out
    else:  # close-only fallback
        px = float(row['close'])
        idx = np.searchsorted(edges, px, side='right') - 1
        if 0 <= idx < bins:
            out[idx] += vol
        return out

def compute_volume_profile(df: pd.DataFrame, bins: int, method: str) -> tuple:
    if df.empty:
        return None, None
    lo = float(df['low'].min()); hi = float(df['high'].max())
    edges = _build_price_grid(lo, hi, bins)
    vol_bins = np.zeros(len(edges)-1, dtype=float)
    for _, row in df.iterrows():
        vol_bins += _distribute_bar_volume_to_bins(row, edges, method)
    return edges, vol_bins

def value_area_from_profile(edges: np.ndarray, vol_bins: np.ndarray, va_fraction: float) -> dict:
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
        left_val  = vol_bins[L] if L >= 0 else -1.0
        right_val = vol_bins[R] if R < bins else -1.0
        if right_val > left_val:
            used[R] = True; cum += right_val; R += 1
        else:
            used[L] = True; cum += left_val;  L -= 1

    poc = 0.5*(edges[poc_idx] + edges[poc_idx+1])
    used_idx = np.where(used)[0]
    if len(used_idx) == 0:
        return {}
    val = edges[used_idx.min()]
    vah = edges[used_idx.max() + 1]
    return {'poc': float(poc), 'val': float(val), 'vah': float(vah)}


# ------------- Level construction -------------
def build_levels_for_day(g_day: pd.DataFrame):
    """
    Build candidate levels for a single ET date (one symbol).
    Returns (levels_list, enriched_day_df, ny_df_with_vwap_bands)
    """
    levels: List[Dict] = []

    # Session references
    asia = g_day.loc[within_wrap(g_day['ts_et'], ASIA_START, ASIA_END)]
    if not asia.empty:
        levels.append({'type':'asia_high','level':float(asia['high'].max()),'side':'high_type',
                       'created_ts':asia.loc[asia['high'].idxmax(), 'ts_et'], 'origin_session':'asia'})
        levels.append({'type':'asia_low','level':float(asia['low'].min()),'side':'low_type',
                       'created_ts':asia.loc[asia['low'].idxmin(), 'ts_et'], 'origin_session':'asia'})

    london = g_day.loc[within(g_day['ts_et'], LONDON_START, LONDON_END)]
    if not london.empty:
        sh, sl = is_swing(london['high'], london['low'], SWING_K)
        if sh.any():
            lh = float(london.loc[sh, 'high'].max())
            levels.append({'type':'london_high','level':lh,'side':'high_type',
                           'created_ts': london.loc[london['high'].idxmax(), 'ts_et'], 'origin_session':'london'})
        if sl.any():
            ll = float(london.loc[sl, 'low'].min())
            levels.append({'type':'london_low','level':ll,'side':'low_type',
                           'created_ts': london.loc[london['low'].idxmin(), 'ts_et'], 'origin_session':'london'})

    # Initial Balance (default 9:30–10:30). If you trade only to 10:15, consider OR(30/45) instead.
    ib = g_day.loc[within(g_day['ts_et'], NY_OPEN, time(10,30))]
    if not ib.empty:
        levels.append({'type':'ib_high','level':float(ib['high'].max()),'side':'high_type',
                       'created_ts': ib.loc[ib['high'].idxmax(), 'ts_et'], 'origin_session':'ny'})
        levels.append({'type':'ib_low','level':float(ib['low'].min()),'side':'low_type',
                       'created_ts': ib.loc[ib['low'].idxmin(), 'ts_et'], 'origin_session':'ny'})
        levels.append({'type':'ib_mid','level': float((ib['high'].max()+ib['low'].min())/2.0),'side':'mid',
                       'created_ts': ib.iloc[0]['ts_et'], 'origin_session':'ny'})

    # Overnight Volume Profile (21:00–09:30 ET)
    on_mask = within_wrap(g_day['ts_et'], time(21,0), NY_OPEN)
    on_df = g_day.loc[on_mask].copy()
    if not on_df.empty:
        edges, vol_bins = compute_volume_profile(on_df, bins=VP_BINS, method=VP_METHOD)
        va = value_area_from_profile(edges, vol_bins, va_fraction=VP_VALUE_AREA)
        if va:
            levels.append({'type':'vp_on_poc','level': va['poc'],'side':'mid',
                           'created_ts': on_df['ts_et'].iloc[0], 'origin_session': 'overnight'})
            levels.append({'type':'vp_on_vah','level': va['vah'],'side':'high_type',
                           'created_ts': on_df['ts_et'].iloc[-1], 'origin_session': 'overnight'})
            levels.append({'type':'vp_on_val','level': va['val'],'side':'low_type',
                           'created_ts': on_df['ts_et'].iloc[-1], 'origin_session': 'overnight'})

    # VWAP + bands (dynamic)
    ny = g_day.loc[within(g_day['ts_et'], NY_OPEN, NY_CLOSE)].copy()
    if not ny.empty:
        tp = (ny['high']+ny['low']+ny['close'])/3.0
        pv = tp*ny['volume']; cum_pv = pv.cumsum(); cum_v = ny['volume'].cumsum().replace(0, np.nan)
        vwap = (cum_pv/cum_v).rename('vwap')
        roll_std = tp.rolling(VWAP_STDEV_LEN, min_periods=VWAP_STDEV_LEN).std(ddof=0)
        ny['vwap'] = vwap
        for mult in VWAP_BANDS:
            ny[f'vwap_u_{mult}'] = vwap + mult*roll_std
            ny[f'vwap_l_{mult}'] = vwap - mult*roll_std
            levels.append({'type': f'vwap+{mult}σ', 'level': float('nan'), 'side':'high_type',
                           'created_ts': ny['ts_et'].iloc[0], 'origin_session':'ny'})
            levels.append({'type': f'vwap-{mult}σ', 'level': float('nan'), 'side':'low_type',
                           'created_ts': ny['ts_et'].iloc[0], 'origin_session':'ny'})
    else:
        ny = pd.DataFrame()

    # Snapshot MAs / Bands at 9:30
    g_day = g_day.copy()
    g_day['sma50'] = g_day['close'].rolling(50, min_periods=50).mean()
    g_day['sma200'] = g_day['close'].rolling(200, min_periods=200).mean()
    g_day['bb_mid'] = g_day['close'].rolling(BB_LEN, min_periods=BB_LEN).mean()
    g_day['bb_std'] = g_day['close'].rolling(BB_LEN, min_periods=BB_LEN).std(ddof=0)
    g_day['bb_up']  = g_day['bb_mid'] + BB_STD*g_day['bb_std']
    g_day['bb_dn']  = g_day['bb_mid'] - BB_STD*g_day['bb_std']
    tr = np.maximum(g_day['high'] - g_day['low'],
                    np.maximum((g_day['high'] - g_day['close'].shift()).abs(),
                               (g_day['low'] - g_day['close'].shift()).abs()))
    g_day['ema20'] = g_day['close'].ewm(span=KELT_EMA, adjust=False).mean()
    g_day['atr10'] = tr.rolling(KELT_ATR, min_periods=KELT_ATR).mean()
    g_day['kel_up'] = g_day['ema20'] + KELT_MULT*g_day['atr10']
    g_day['kel_dn'] = g_day['ema20'] - KELT_MULT*g_day['atr10']

    open_row = g_day.loc[within(g_day['ts_et'], NY_OPEN, time(9,31))]
    if not open_row.empty:
        r0 = open_row.iloc[0]
        static_refs = [
            ('sma50', r0['sma50'], 'mid'),
            ('sma200', r0['sma200'], 'mid'),
            ('bb_up', r0['bb_up'], 'high_type'),
            ('bb_dn', r0['bb_dn'], 'low_type'),
            ('kel_up', r0['kel_up'], 'high_type'),
            ('kel_dn', r0['kel_dn'], 'low_type'),
        ]
        for nm, lvl, side in static_refs:
            if pd.notna(lvl):
                levels.append({'type': nm, 'level': float(lvl), 'side': side,
                               'created_ts': r0['ts_et'], 'origin_session':'ny'})

    # Time gaps
    for gm in [15, 60, 240]:
        for (gdir, lvl, ts_created) in find_time_gaps(g_day, gm):
            side = 'high_type' if 'down' in gdir else 'low_type'
            levels.append({'type': gdir, 'level': float(lvl), 'side': side,
                           'created_ts': ts_created, 'origin_session': session_name(ts_created)})

    # FVGs (1m/5m/15m)
    fvgs = []
    fvgs += find_fvgs(g_day)  # 1m
    g5  = resample_ohlc(g_day, '5min')
    g15 = resample_ohlc(g_day, '15min')
    fvgs += [(f'{t}_5m', lvl, ts) for (t,lvl,ts) in find_fvgs(g5)]
    fvgs += [(f'{t}_15m', lvl, ts) for (t,lvl,ts) in find_fvgs(g15)]
    for (t,lvl,ts) in fvgs:
        side = 'low_type' if t.startswith('bull') else 'high_type'
        levels.append({'type': t, 'level': float(lvl), 'side': side,
                       'created_ts': ts, 'origin_session': session_name(ts)})

    # ---- High-Timeframe levels (last completed bars) ----
    levels.extend(_last_completed_tf_levels(g_day))

    return levels, g_day, ny


def _last_completed_tf_levels(g_day: pd.DataFrame):
    """Create H1/H4/D1/W1 highs/lows from the last completed bar before 9:30 ET."""
    def tf_resample(rule):
        agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
        tf = g_day.resample(rule, on='ts_et').agg(agg).dropna()
        tf['start'] = tf.index
        return tf

    out = []
    h1 = tf_resample('60min'); h4 = tf_resample('240min')
    daily = tf_resample('1D'); weekly = tf_resample('1W')

    def add_tf(levels_name, tf_df):
        if tf_df.empty: return
        day_date = g_day['ts_et'].dt.date.iloc[0]
        # bars that started strictly before 9:30 of same date
        tz = g_day['ts_et'].iloc[0].tz
        thresh = pd.Timestamp.combine(day_date, NY_OPEN).tz_localize(tz)
        tf_day = tf_df[tf_df['start'] < thresh]
        if tf_day.empty: return
        row = tf_day.iloc[-1]
        out.extend([
            {'type': f'{levels_name}_high', 'level': float(row['high']), 'side':'high_type',
             'created_ts': row['start'], 'origin_session': levels_name},
            {'type': f'{levels_name}_low', 'level': float(row['low']), 'side':'low_type',
             'created_ts': row['start'], 'origin_session': levels_name},
        ])

    add_tf('h1', h1)
    add_tf('h4', h4)
    add_tf('d1', daily)
    add_tf('w1', weekly)
    return out


# ------------- Touch + reversal metrics -------------
def _window_label(ts: pd.Timestamp) -> Optional[str]:
    for ws, we, wn in WINDOWS:
        if ws <= ts.time() < we:
            return wn
    return None

def first_touch_and_reversal(day_df: pd.DataFrame,
                             series_df: pd.DataFrame,
                             level: Dict) -> Optional[Dict]:
    ny = day_df.loc[within(day_df['ts_et'], NY_OPEN, NY_CLOSE)].copy()
    if ny.empty:
        return None

    lvl = level['level']
    lvl_type = level['type']
    side = level['side']

    dynamic_vwap = isinstance(lvl, float) and np.isnan(lvl) and lvl_type.startswith('vwap')
    band_col = None
    if dynamic_vwap and not series_df.empty:
        if '+' in lvl_type:
            m = float(lvl_type.split('+')[-1].replace('σ',''))
            band_col = f'vwap_u_{m}'
        else:
            m = float(lvl_type.split('-')[-1].replace('σ',''))
            band_col = f'vwap_l_{m}'
        if band_col not in series_df.columns:
            return None

    touch_idx = None; touch_ts = None; touch_price = None; touch_window = None
    for idx, row in ny.iterrows():
        cur_ts = row['ts_et']
        win_label = _window_label(cur_ts)

        if dynamic_vwap:
            ser_row = series_df[series_df['ts_et'] == cur_ts]
            if ser_row.empty:
                continue
            lvl_now = float(ser_row.iloc[0][band_col])
        else:
            lvl_now = float(lvl)

        if side == 'high_type':
            hit = row['high'] >= lvl_now - 1e-9
        elif side == 'low_type':
            hit = row['low'] <= lvl_now + 1e-9
        else:
            hit = (row['low'] <= lvl_now) & (lvl_now <= row['high'])

        if hit:
            touch_idx = idx; touch_ts = cur_ts; touch_price = lvl_now; touch_window = win_label
            break

    if touch_idx is None:
        return None

    post = ny.iloc[(ny.index.get_loc(touch_idx)+1):]
    post = post[post['ts_et'] < touch_ts + timedelta(minutes=REV_WINDOW_MIN)]
    if post.empty:
        return None

    if 'atr14' in day_df.columns:
        atr_series = day_df['atr14'].loc[day_df['ts_et'] <= touch_ts]
        atr14 = atr_series.iloc[-1] if not atr_series.empty else np.nan
    else:
        atr14 = np.nan

    if side == 'high_type':
        mfe_opp = float(max(0.0, (touch_price - post['low'].min())))
        mae_thru = float(max(0.0, (post['high'].max() - touch_price)))
        retest = (post['high'] >= touch_price - 1e-9).any()
        peak_ts = post.loc[post['low'].idxmin(), 'ts_et']
    elif side == 'low_type':
        mfe_opp = float(max(0.0, (post['high'].max() - touch_price)))
        mae_thru = float(max(0.0, (touch_price - post['low'].min())))
        retest = (post['low'] <= touch_price + 1e-9).any()
        peak_ts = post.loc[post['high'].idxmax(), 'ts_et']
    else:
        up = float(max(0.0, post['high'].max() - touch_price))
        dn = float(max(0.0, touch_price - post['low'].min()))
        mfe_opp = max(up, dn); mae_thru = 0.0
        retest = ((post['low'] <= touch_price) & (touch_price <= post['high'])).any()
        peak_ts = (post.loc[post['high'].idxmax(), 'ts_et'] if up >= dn
                   else post.loc[post['low'].idxmin(), 'ts_et'])

    reversal_ratio = mfe_opp / (mae_thru + 1e-9)
    disp_atr = (mfe_opp / atr14) if (atr14 and not np.isnan(atr14) and atr14 > 0) else np.nan
    time_to_peak_min = (peak_ts - touch_ts).total_seconds()/60.0

    return {
        'level_type': lvl_type,
        'side': side,
        'origin_session': level.get('origin_session',''),
        'created_ts': level.get('created_ts'),
        'touch_ts': touch_ts,
        'touch_window': touch_window,
        'touch_price': touch_price,
        'mfe_opp': mfe_opp,
        'mae_thru': mae_thru,
        'reversal_ratio': reversal_ratio,
        'disp_atr': disp_atr,
        'time_to_peak_min': time_to_peak_min
    }


# ------------- Per-day processing -------------
def process_day(symbol: str, g: pd.DataFrame) -> List[Dict]:
    g = g.copy()
    g['atr14'] = compute_atr(g, ATR_LEN)

    levels, g_day, ny = build_levels_for_day(g)

    # Build a static-levels DataFrame for confluence tagging (exclude dynamic VWAP NaNs)
    df_levels = pd.DataFrame([
        {'type': lv['type'], 'level': lv['level']}
        for lv in levels
        if isinstance(lv.get('level'), float) and not np.isnan(lv['level'])
    ])

    open_row = g_day.loc[within(g_day['ts_et'], NY_OPEN, time(9,31))]
    if open_row.empty:
        return []
    open_px = float(open_row.iloc[0]['open'])

    events: List[Dict] = []

    def combo_tags(ev, prox):
        if df_levels.empty: return []
        # Nearby OTHER levels within prox points of the touch price
        near = df_levels.loc[
            (df_levels['type'] != ev['level_type']) &
            (df_levels['level'].sub(ev['touch_price']).abs() <= prox)
        ]
        # Build pair keys (primary + neighbor), sorted for stable identity
        tags = []
        for t in near['type'].unique():
            pair = "+".join(sorted([ev['level_type'], t]))
            tags.append(pair)
        return tags

    for lvl in levels:
        ev = first_touch_and_reversal(g_day, ny, lvl)
        if ev is None:
            continue
        ev['symbol'] = symbol
        ev['et_date'] = g_day['ts_et'].dt.date.iloc[0]
        # distance metric (static levels use their value; dynamic use touch price)
        if isinstance(lvl.get('level'), float) and not np.isnan(lvl['level']):
            dist = abs(lvl['level'] - open_px)
        else:
            dist = abs(ev['touch_price'] - open_px)
        ev['distance_from_open'] = float(dist)

        # Confluence combo tags for each proximity threshold
        for prox in CONF_THRESH_POINTS:
            tags = combo_tags(ev, prox)
            ev[f'combo_{prox}pt'] = "|".join(tags) if tags else ""

        events.append(ev)

    return events


# ------------- Aggregation / ranking -------------
def summarize_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    def pct(x): return 100.0 * x.mean() if len(x)>0 else np.nan
    def share_at_least(x, thr): return 100.0 * (x >= thr).mean() if len(x)>0 else np.nan

    grp = df.groupby('level_type')

    summary = pd.DataFrame({
        'N_touches': grp.size(),
        'median_dispATR': grp['disp_atr'].median(),
        'p_dispATR_ge_0.5': grp['disp_atr'].apply(lambda s: share_at_least(s, 0.5)),
        'p_dispATR_ge_1.0': grp['disp_atr'].apply(lambda s: share_at_least(s, 1.0)),
        'median_rev_ratio': grp['reversal_ratio'].median(),
        'median_time_to_peak_min': grp['time_to_peak_min'].median(),
        'median_distance_from_open': grp['distance_from_open'].median(),
        'p_first_window_9:30-9:45': grp['touch_window'].apply(lambda s: pct(s=='9:30-9:45')),
        'p_first_window_9:45-10:00': grp['touch_window'].apply(lambda s: pct(s=='9:45-10:00')),
        'p_first_window_10:00-10:15': grp['touch_window'].apply(lambda s: pct(s=='10:00-10:15')),
    }).reset_index()

    # Composite reversal power score (robust z-scores)
    def robust_z(series: pd.Series) -> pd.Series:
        med = series.median()
        mad = (series - med).abs().median()
        if mad == 0 or pd.isna(mad):
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - med) / (1.4826*mad)

    z_disp = robust_z(summary['median_dispATR'].fillna(0))
    z_p1   = robust_z(summary['p_dispATR_ge_1.0'].fillna(0))
    z_p05  = robust_z(summary['p_dispATR_ge_0.5'].fillna(0))
    z_rr   = robust_z(summary['median_rev_ratio'].replace([np.inf,-np.inf], np.nan).fillna(0))
    z_t    = robust_z(summary['median_time_to_peak_min'].fillna(summary['median_time_to_peak_min'].median()))
    z_d    = robust_z(summary['median_distance_from_open'].fillna(summary['median_distance_from_open'].median()))

    summary['rev_power_score'] = (
        z_disp + z_p1 + 0.5*z_p05 + z_rr - 0.5*z_t - 0.25*z_d
    )

    summary.sort_values(['rev_power_score','N_touches'], ascending=[False, False], inplace=True)
    return summary


def summarize_combos(df_ev: pd.DataFrame, prox: int) -> pd.DataFrame:
    col = f'combo_{prox}pt'
    if col not in df_ev.columns:
        return pd.DataFrame()
    dfc = df_ev.copy()
    dfc[col] = dfc[col].fillna("")
    dfc = dfc[dfc[col] != ""].copy()
    if dfc.empty:
        return pd.DataFrame()
    dfc = dfc.assign(combo=dfc[col].str.split("|")).explode('combo')

    # Define reversal success (tune if desired)
    dfc['rev_success_ge1ATR'] = (dfc['disp_atr'] >= 1.0).astype(int)

    g = dfc.groupby('combo')
    out = pd.DataFrame({
        'N_touches': g.size(),
        'success_ge1ATR_%': 100.0 * g['rev_success_ge1ATR'].mean(),
        'median_dispATR': g['disp_atr'].median(),
        'p_dispATR_ge_0.5': 100.0 * g['disp_atr'].apply(lambda s: (s>=0.5).mean()),
        'p_dispATR_ge_1.0': 100.0 * g['disp_atr'].apply(lambda s: (s>=1.0).mean()),
        'median_time_to_peak_min': g['time_to_peak_min'].median(),
        'median_distance_from_open': g['distance_from_open'].median(),
        'median_MFE_points': g['mfe_opp'].median(),
        'q75_MFE_points': g['mfe_opp'].quantile(0.75),
        'q90_MFE_points': g['mfe_opp'].quantile(0.90),
    }).reset_index().sort_values(['success_ge1ATR_%','N_touches'], ascending=[False, False])
    return out


# ------------- Main -------------
def main():
    tz_et = pytz.timezone("US/Eastern")
    all_events: List[Dict] = []
    residual = pd.DataFrame()

    usecols = ['ts_event','open','high','low','close','volume','symbol']
    dtypes  = {'open':'float64','high':'float64','low':'float64','close':'float64','volume':'float64','symbol':'string'}

    for chunk in pd.read_csv(CSV_FILE, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE):
        df = pd.concat([residual, chunk], ignore_index=True)
        df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True, errors='coerce')
        df.dropna(subset=['ts_event'], inplace=True)
        df['ts_et'] = df['ts_event'].dt.tz_convert(tz_et)
        df['et_date'] = df['ts_et'].dt.date

        df = df[df['symbol'].astype(str).str.match(SYMBOL_REGEX)]

        if WEEKDAYS_ONLY:
            df = df[df['ts_et'].dt.weekday <= 4]

        if df.empty:
            continue

        max_date = df['et_date'].max()
        to_proc = df[df['et_date'] < max_date]
        residual = df[df['et_date'] == max_date]

        for (sym, d), g in to_proc.groupby(['symbol','et_date']):
            g = g.sort_values('ts_et').reset_index(drop=True)
            events = process_day(sym, g)
            all_events.extend(events)

    if not residual.empty:
        for (sym, d), g in residual.groupby(['symbol','et_date']):
            g = g.sort_values('ts_et').reset_index(drop=True)
            events = process_day(sym, g)
            all_events.extend(events)

    if not all_events:
        print("No reversal events found.")
        return

    df_ev = pd.DataFrame(all_events)
    df_ev.to_csv(OUT_EVENTS, index=False)
    print(f"✅ Wrote per-touch reversal events to {OUT_EVENTS} ({len(df_ev)} rows)")

    df_sum = summarize_events(df_ev)
    df_sum.to_csv(OUT_SUMMARY, index=False)
    print(f"✅ Wrote reversal power summary to {OUT_SUMMARY} ({len(df_sum)} rows)")

    cols = ['level_type','N_touches','median_dispATR','p_dispATR_ge_1.0',
            'median_rev_ratio','median_time_to_peak_min','rev_power_score']
    print("\n=== Top 20 Level Types by Reversal Power ===")
    print(df_sum[cols].head(20).to_string(index=False))

    # Combo summaries
    for prox in CONF_THRESH_POINTS:
        combo_df = summarize_combos(df_ev, prox)
        if not combo_df.empty:
            fname = OUT_DIR / f"reversal_combo_{prox}pt.csv"
            combo_df.to_csv(str(fname), index=False)
            print(f"\n✅ Wrote combo summary @ ±{prox} pts → {fname} (rows={len(combo_df)})")
            print(combo_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()