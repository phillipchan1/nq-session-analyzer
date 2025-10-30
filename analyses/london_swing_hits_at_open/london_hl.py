#!/usr/bin/env python3
"""
Simple plug-and-play version:
Drop this file in the same directory as your DataBento 1-minute CSV (default name below)
and run:
    python nyopen_london_swing_hits.py

It will output summary_nyopen_london_swing_hits.csv
"""

import re, sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz

# -------------------- CONFIG --------------------
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20250926.ohlcv-1m.csv")
OUT_FILE = str(OUT_DIR / "summary_nyopen_london_swing_hits.csv")

LONDON_START = time(3, 0)
LONDON_END   = time(8, 0)
WINDOWS = [(time(9,30), time(9,45)),
           (time(9,45), time(10,0)),
           (time(10,0), time(10,15))]

SWING_K = 2           # bars on each side for swing point
EPSILON = 0.0         # overshoot requirement; use 0.25 for 1 tick
WEEKDAYS_ONLY = True  # skip weekends
CHUNKSIZE = 1_000_000
NQ_SINGLE_CONTRACT_RE = re.compile(r"^NQ[A-Z]\d{1,2}$")  # e.g., NQZ0, NQM24
# ------------------------------------------------

def within(ts, start_t, end_t):
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)

def is_swing(high, low, k):
    sh = pd.Series(True, index=high.index)
    sl = pd.Series(True, index=low.index)
    for off in range(1, k+1):
        sh &= (high > high.shift(+off)) & (high > high.shift(-off))
        sl &= (low  <  low.shift(+off)) & (low  <  low.shift(-off))
    sh.iloc[:k] = sh.iloc[-k:] = False
    sl.iloc[:k] = sl.iloc[-k:] = False
    return sh.fillna(False), sl.fillna(False)

def process_day(g):
    london = g.loc[within(g['ts_et'], LONDON_START, LONDON_END)]
    if london.empty: return None
    sh, sl = is_swing(london['high'], london['low'], SWING_K)
    idx_h = london['high'].idxmax()
    idx_l = london['low'].idxmin()
    london_high, london_low = london.at[idx_h,'high'], london.at[idx_l,'low']
    high_is_swing, low_is_swing = sh.get(idx_h, False), sl.get(idx_l, False)

    pre_open = g.loc[within(g['ts_et'], LONDON_END, time(9,30))]
    high_avail = low_avail = False
    if high_is_swing:
        high_avail = not (pre_open['high'] >= london_high - 1e-12 + EPSILON).any()
    if low_is_swing:
        low_avail = not (pre_open['low']  <= london_low  + 1e-12 - EPSILON).any()
    if not (high_avail or low_avail): return None

    def hit(win):
        w = g.loc[within(g['ts_et'], *win)]
        if w.empty: return False, False
        hh = ll = False
        if high_avail: hh = (w['high'] >= london_high - 1e-12 + EPSILON).any()
        if low_avail:  ll = (w['low']  <= london_low  + 1e-12 - EPSILON).any()
        return hh, ll

    res = {'avail_high': int(high_avail), 'avail_low': int(low_avail)}
    for i, win in enumerate(WINDOWS, 1):
        hh,ll = hit(win)
        res[f'w{i}_hit_high'], res[f'w{i}_hit_low'] = int(hh), int(ll)
    return res

def main():
    tz_et = pytz.timezone("US/Eastern")
    agg = {'avail_high_days':0,'avail_low_days':0,
           'w1_hit_high':0,'w1_hit_low':0,
           'w2_hit_high':0,'w2_hit_low':0,
           'w3_hit_high':0,'w3_hit_low':0}
    residual = pd.DataFrame()

    for chunk in pd.read_csv(CSV_FILE,
                             usecols=['ts_event','open','high','low','close','volume','symbol'],
                             dtype={'open':'float64','high':'float64','low':'float64',
                                    'close':'float64','volume':'int64','symbol':'string'},
                             chunksize=CHUNKSIZE):
        df = pd.concat([residual, chunk])
        df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True, errors='coerce')
        df.dropna(subset=['ts_event'], inplace=True)
        df['ts_et'] = df['ts_event'].dt.tz_convert(tz_et)
        df['et_date'] = df['ts_et'].dt.date
        df = df[df['symbol'].astype(str).str.match(NQ_SINGLE_CONTRACT_RE)]
        if WEEKDAYS_ONLY:
            df = df[df['ts_et'].dt.weekday <= 4]
        if df.empty: continue

        max_date = df['et_date'].max()
        to_proc = df[df['et_date'] < max_date]
        residual = df[df['et_date'] == max_date]
        for (sym, d), g in to_proc.groupby(['symbol','et_date']):
            g.sort_values('ts_et', inplace=True)
            res = process_day(g)
            if not res: continue
            for k,v in res.items(): agg[k + '_days' if k in ['avail_high', 'avail_low'] else k] += v

    # process final residual
    if not residual.empty:
        for (sym,d),g in residual.groupby(['symbol','et_date']):
            g.sort_values('ts_et', inplace=True)
            res = process_day(g)
            if not res: continue
            for k,v in res.items(): agg[k + '_days' if k in ['avail_high', 'avail_low'] else k] += v

    def pct(n,d): return round(100*n/d,2) if d>0 else 0.0
    out = {
        'available_high_days': agg['avail_high_days'],
        'available_low_days':  agg['avail_low_days'],
        '9:30-9:45 % hit HIGH': pct(agg['w1_hit_high'], agg['avail_high_days']),
        '9:30-9:45 % hit LOW':  pct(agg['w1_hit_low'],  agg['avail_low_days']),
        '9:45-10:00 % hit HIGH': pct(agg['w2_hit_high'], agg['avail_high_days']),
        '9:45-10:00 % hit LOW':  pct(agg['w2_hit_low'],  agg['avail_low_days']),
        '10:00-10:15 % hit HIGH': pct(agg['w3_hit_high'], agg['avail_high_days']),
        '10:00-10:15 % hit LOW':  pct(agg['w3_hit_low'],  agg['avail_low_days']),
        'First 45min % hit HIGH': pct(agg['w1_hit_high'] + agg['w2_hit_high'], agg['avail_high_days']),
        'First 45min % hit LOW':  pct(agg['w1_hit_low'] + agg['w2_hit_low'],  agg['avail_low_days']),
    }
    pd.DataFrame([out]).to_csv(OUT_FILE, index=False)
    print(f"✅ Wrote {OUT_FILE}")
    print(pd.DataFrame([out]).T.to_string(header=False))

if __name__ == "__main__":
    main()
