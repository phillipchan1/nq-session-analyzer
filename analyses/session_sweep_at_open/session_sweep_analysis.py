#!/usr/bin/env python3
"""
Session Swing H/L Sweep Analysis
---------------------------------
Checks whether Asia and London session swing H/L levels that survive
premarket get swept in the first 45 minutes of NY trading (9:30-10:15 ET),
broken down by 15-min macro windows.

Swing H/L = the session extreme bar must be confirmed as a swing point
(k bars on each side), not just a random edge-of-session extreme.

Swept = strict takeout, price must exceed the level by at least 1 NQ tick
(0.25 pts).

Usage:
    python session_sweep_analysis.py

Outputs:
    session_sweep_summary.csv   – aggregate probabilities by session/side
    session_sweep_detailed.csv  – per-day rows for manual backtesting
"""

import re
from pathlib import Path
from datetime import time, timedelta

import pandas as pd
import numpy as np
import pytz

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20260221.ohlcv-1m.csv")
SUMMARY_OUT = str(OUT_DIR / "session_sweep_summary.csv")
DETAILED_OUT = str(OUT_DIR / "session_sweep_detailed.csv")

ASIA_START = time(19, 0)     # 7:00 PM ET
ASIA_END = time(2, 0)        # 2:00 AM ET (wraps midnight)
LONDON_START = time(2, 0)    # 2:00 AM ET
LONDON_END = time(8, 0)      # 8:00 AM ET
NY_OPEN = time(9, 30)

WINDOWS = [
    ("9:30-9:45", time(9, 30), time(9, 45)),
    ("9:45-10:00", time(9, 45), time(10, 0)),
    ("10:00-10:15", time(10, 0), time(10, 15)),
]

SWING_K = 2       # bars each side for swing confirmation
NQ_TICK = 0.25    # strict sweep = exceeds level by >= 1 tick
CHUNKSIZE = 1_000_000
WEEKDAYS_ONLY = True
NQ_RE = re.compile(r"^NQ[A-Z]\d{1,2}$")
# ==============================================


# --------------- helpers ---------------

def _within(ts, start_t, end_t):
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)


def _within_wrap(ts, start_t, end_t):
    """Intervals that wrap midnight, e.g. 19:00 -> 02:00."""
    t = ts.dt.time
    return (t >= start_t) | (t < end_t)


def _is_swing(high, low, k):
    sh = pd.Series(True, index=high.index)
    sl = pd.Series(True, index=low.index)
    for off in range(1, k + 1):
        sh &= (high > high.shift(off)) & (high > high.shift(-off))
        sl &= (low < low.shift(off)) & (low < low.shift(-off))
    sh.iloc[:k] = sh.iloc[-k:] = False
    sl.iloc[:k] = sl.iloc[-k:] = False
    return sh.fillna(False), sl.fillna(False)


def _check_sweep(bars, level, is_high):
    """Strict sweep: price must exceed level by >= 1 NQ tick.

    Returns (swept, first_time, points_beyond).
    """
    if bars.empty:
        return False, None, 0.0
    if is_high:
        mask = bars["high"] > level + NQ_TICK
    else:
        mask = bars["low"] < level - NQ_TICK
    if mask.any():
        idx = bars[mask].index[0]
        t = bars.at[idx, "ts_et"]
        pts = (bars.at[idx, "high"] - level) if is_high else (level - bars.at[idx, "low"])
        return True, t, pts
    return False, None, 0.0


def _front_month(df):
    """Keep only the front-month contract per trading_date (max vol 9:30-9:45)."""
    rth = df.loc[_within(df["ts_et"], NY_OPEN, time(9, 45))]
    if rth.empty:
        return df
    vol = (
        rth.groupby(["trading_date", "symbol"])
        .agg(volume=("volume", "sum"))
        .reset_index()
    )
    best = vol.loc[
        vol.groupby("trading_date")["volume"].idxmax(),
        ["trading_date", "symbol"],
    ].rename(columns={"symbol": "front"})
    out = df.merge(best, on="trading_date", how="left")
    return out.loc[out["symbol"] == out["front"]].drop(columns=["front"])


# --------------- per-day processing ---------------

def _process_day(g):
    """Analyse one (symbol, trading_date) group.

    Returns list[dict] – one row per available level (up to 4: Asia H/L,
    London H/L).
    """
    rows = []

    ny = g.loc[_within(g["ts_et"], NY_OPEN, time(10, 15))]
    if ny.empty:
        return rows
    ny_open = ny.iloc[0]["open"]

    sessions = [
        ("Asia", _within_wrap(g["ts_et"], ASIA_START, ASIA_END), ASIA_END),
        ("London", _within(g["ts_et"], LONDON_START, LONDON_END), LONDON_END),
    ]

    for sess_name, sess_mask, pre_start in sessions:
        sess = g.loc[sess_mask]
        if len(sess) < SWING_K * 2 + 1:
            continue

        sh, sl = _is_swing(sess["high"], sess["low"], SWING_K)

        idx_h = sess["high"].idxmax()
        idx_l = sess["low"].idxmin()
        s_high = sess.at[idx_h, "high"]
        s_low = sess.at[idx_l, "low"]
        h_swing = sh.get(idx_h, False)
        l_swing = sl.get(idx_l, False)

        pre = g.loc[_within(g["ts_et"], pre_start, NY_OPEN)]

        for side, level, is_high, is_sw in [
            ("High", s_high, True, h_swing),
            ("Low", s_low, False, l_swing),
        ]:
            if not is_sw:
                continue

            swept_pre, _, _ = _check_sweep(pre, level, is_high)
            if swept_pre:
                continue

            rec = {
                "session": sess_name,
                "side": side,
                "level_price": level,
                "ny_open_price": ny_open,
                "distance_from_open": round(level - ny_open, 2),
            }

            first_time = first_win = None
            for wname, ws, we in WINDOWS:
                w = g.loc[_within(g["ts_et"], ws, we)]
                swept, st, _ = _check_sweep(w, level, is_high)
                rec[f"swept_{wname}"] = int(swept)
                if swept and first_time is None:
                    first_time = st
                    first_win = wname

            rec["first_sweep_time"] = first_time
            rec["first_sweep_window"] = first_win
            rec["swept_first_45min"] = int(first_win is not None)

            rows.append(rec)

    return rows


# --------------- main ---------------

def main():
    tz = pytz.timezone("US/Eastern")
    all_detail = []
    residual = pd.DataFrame()
    chunk_n = 0

    print("Loading data …")
    for chunk in pd.read_csv(
        CSV_FILE,
        usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
            "symbol": "string",
        },
        chunksize=CHUNKSIZE,
    ):
        chunk_n += 1
        df = pd.concat([residual, chunk], ignore_index=True)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz)
        df = df[df["symbol"].astype(str).str.match(NQ_RE)]

        # Trading date: evening bars (>=18:00) belong to next calendar day's session
        cal = df["ts_et"].dt.normalize()
        evening = df["ts_et"].dt.time >= time(18, 0)
        td = cal.copy()
        td.loc[evening] = cal.loc[evening] + pd.Timedelta(days=1)
        df["trading_date"] = td.dt.date

        if WEEKDAYS_ONLY:
            td_weekday = df["trading_date"].apply(lambda d: d.weekday())
            df = df[td_weekday <= 4]
        if df.empty:
            residual = pd.DataFrame()
            continue

        max_td = df["trading_date"].max()
        to_proc = df[df["trading_date"] < max_td]
        residual = df[df["trading_date"] == max_td]

        if to_proc.empty:
            continue

        to_proc = _front_month(to_proc)
        n_before = len(all_detail)
        for (sym, td_val), g in to_proc.groupby(["symbol", "trading_date"]):
            g = g.sort_values("ts_et")
            for rec in _process_day(g):
                rec["date"] = td_val
                rec["symbol"] = sym
                all_detail.append(rec)
        print(f"  chunk {chunk_n}: +{len(all_detail) - n_before} rows  (total {len(all_detail)})")

    # final residual
    if not residual.empty:
        residual = _front_month(residual)
        for (sym, td_val), g in residual.groupby(["symbol", "trading_date"]):
            g = g.sort_values("ts_et")
            for rec in _process_day(g):
                rec["date"] = td_val
                rec["symbol"] = sym
                all_detail.append(rec)

    # ---- build outputs ----
    if not all_detail:
        print("No qualifying levels found.")
        return

    detail = pd.DataFrame(all_detail)
    col_order = [
        "date", "symbol", "session", "side", "level_price",
        "ny_open_price", "distance_from_open",
        "swept_9:30-9:45", "swept_9:45-10:00", "swept_10:00-10:15",
        "swept_first_45min", "first_sweep_time", "first_sweep_window",
    ]
    detail = detail[col_order].sort_values(["date", "session", "side"])
    detail.to_csv(DETAILED_OUT, index=False)
    print(f"\n✅ Wrote {DETAILED_OUT}  ({len(detail)} rows)")

    # ---- summary ----
    summary_rows = []
    for (sess, side), grp in detail.groupby(["session", "side"]):
        n = len(grp)
        fw = grp["first_sweep_window"].value_counts()

        w1 = int(fw.get("9:30-9:45", 0))
        w2 = int(fw.get("9:45-10:00", 0))
        w3 = int(fw.get("10:00-10:15", 0))
        total = w1 + w2 + w3

        pct = lambda v: round(100 * v / n, 1) if n else 0.0

        summary_rows.append({
            "session": sess,
            "side": side,
            "available_days": n,
            # Incremental: which window had the FIRST sweep
            "first_in_9:30-9:45": w1,
            "first_in_9:30-9:45_pct": pct(w1),
            "first_in_9:45-10:00": w2,
            "first_in_9:45-10:00_pct": pct(w2),
            "first_in_10:00-10:15": w3,
            "first_in_10:00-10:15_pct": pct(w3),
            # Cumulative: swept by end of window
            "swept_by_9:45": w1,
            "swept_by_9:45_pct": pct(w1),
            "swept_by_10:00": w1 + w2,
            "swept_by_10:00_pct": pct(w1 + w2),
            "swept_by_10:15": total,
            "swept_by_10:15_pct": pct(total),
            "not_swept_45min": n - total,
            "not_swept_45min_pct": pct(n - total),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"✅ Wrote {SUMMARY_OUT}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
