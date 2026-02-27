#!/usr/bin/env python3
"""
6am 4H Candle Post-Formation Sweep -- Enhanced with Predictive Factors
-----------------------------------------------------------------------
At 10:00 AM ET the 6am 4H candle (6:00-10:00) is fully formed.

Core question: if the 10:00 close is at least MIN_DISTANCE points from
one side of the candle, how often does price sweep (strict takeout, >= 1
NQ tick beyond) at least one side in the 10:00-10:15 macro?

Enhanced with predictive factors across three tiers:

  Tier 1 (structural):
    - Close position in candle range
    - Direction of last 15m candle (9:45-10:00)
    - When the candle extreme was formed (pre-market vs RTH)
    - Unswept Asia/London swing level near the 4H boundary
    - High-impact economic event near 10:00 AM

  Tier 2 (context):
    - RTH portion of candle range (9:30-10:00 vs 6:00-10:00)
    - Pre-RTH range (overnight activity on this calendar date)

  Tier 3 (regime):
    - VIXY previous-day close (volatility regime)
    - Day of week

Usage:
    python 4h_6am_post_formation_sweep.py

Outputs:
    4h_6am_sweep_summary.csv            – overall sweep probabilities
    4h_6am_sweep_distance_buckets.csv   – sweep probability by distance bucket
    4h_6am_sweep_detailed.csv           – per-day rows for manual backtesting
    4h_6am_sweep_factor_analysis.csv    – sweep rates bucketed by each factor
"""

import re
from pathlib import Path
from datetime import time, date as date_type

import pandas as pd
import numpy as np
import pytz

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ANALYSES_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20260221.ohlcv-1m.csv")

SUMMARY_OUT = str(OUT_DIR / "4h_6am_sweep_summary.csv")
BUCKETS_OUT = str(OUT_DIR / "4h_6am_sweep_distance_buckets.csv")
DETAILED_OUT = str(OUT_DIR / "4h_6am_sweep_detailed.csv")
FACTOR_OUT = str(OUT_DIR / "4h_6am_sweep_factor_analysis.csv")

# External data for factor enrichment
EVENTS_FILE = str(DATA_DIR / "us_high_impact_events_2020_to_2025.csv")
SESSION_SWEEP_FILE = str(
    ANALYSES_DIR / "session_sweep_at_open" / "session_sweep_detailed.csv"
)
VIX_METRICS_FILE = str(
    ANALYSES_DIR / "vix_range_prediction" / "vix_daily_metrics.csv"
)

CANDLE_START = time(6, 0)
CANDLE_END = time(10, 0)
SWEEP_START = time(10, 0)
SWEEP_END = time(10, 15)
NY_OPEN = time(9, 30)
LAST_15M_START = time(9, 45)

MIN_DISTANCE = 20
NQ_TICK = 0.25
BUCKET_EDGES = [20, 40, 60, 80, 100]
UNSWEPT_NEAR_THRESHOLD = 15  # points -- "near" the 4H boundary

CHUNKSIZE = 1_000_000
WEEKDAYS_ONLY = True
NQ_RE = re.compile(r"^NQ[A-Z]\d{1,2}$")
# ==============================================


# --------------- helpers ---------------

def _within(ts, start_t, end_t):
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)


def _front_month(df):
    """Keep only the front-month contract per et_date (max vol 9:30-9:45)."""
    rth = df.loc[_within(df["ts_et"], NY_OPEN, time(9, 45))]
    if rth.empty:
        return df
    vol = (
        rth.groupby(["et_date", "symbol"])
        .agg(volume=("volume", "sum"))
        .reset_index()
    )
    best = vol.loc[
        vol.groupby("et_date")["volume"].idxmax(),
        ["et_date", "symbol"],
    ].rename(columns={"symbol": "front"})
    out = df.merge(best, on="et_date", how="left")
    return out.loc[out["symbol"] == out["front"]].drop(columns=["front"])


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


def _distance_bucket(dist):
    for i, edge in enumerate(BUCKET_EDGES):
        if dist < edge:
            lo = BUCKET_EDGES[i - 1] if i > 0 else 0
            return f"{lo}-{edge}"
    return f"{BUCKET_EDGES[-1]}+"


# --------------- external data loaders ---------------

def _load_events():
    """Returns dict[date] -> list of (time_str, event_name)."""
    try:
        ev = pd.read_csv(EVENTS_FILE)
        ev["date"] = pd.to_datetime(ev["date"], errors="coerce").dt.date
        ev.dropna(subset=["date"], inplace=True)
        lookup = {}
        for _, row in ev.iterrows():
            d = row["date"]
            lookup.setdefault(d, []).append((str(row["time_et"]), str(row["event_name"])))
        return lookup
    except Exception as e:
        print(f"  Warning: could not load events ({e})")
        return {}


def _load_session_levels():
    """Returns dict[date] -> list of level_price values still unswept at 10:00 AM.

    A level is "still available at 10:00" if it was available at 9:30 and
    was NOT swept in the 9:30-9:45 or 9:45-10:00 windows.
    """
    try:
        sl = pd.read_csv(SESSION_SWEEP_FILE)
        sl["date"] = pd.to_datetime(sl["date"], errors="coerce").dt.date
        sl.dropna(subset=["date"], inplace=True)
        available_at_10 = sl[
            (sl["swept_9:30-9:45"] == 0) & (sl["swept_9:45-10:00"] == 0)
        ]
        lookup = {}
        for _, row in available_at_10.iterrows():
            lookup.setdefault(row["date"], []).append(row["level_price"])
        return lookup
    except Exception as e:
        print(f"  Warning: could not load session sweep data ({e})")
        return {}


def _load_vixy():
    """Returns dict[date] -> VIXY previous-day close from pre-computed VIX metrics."""
    try:
        vm = pd.read_csv(VIX_METRICS_FILE)
        vm["date"] = pd.to_datetime(vm["date"], errors="coerce").dt.date
        vm.dropna(subset=["date"], inplace=True)
        return dict(zip(vm["date"], vm["vix_prev_day_close"]))
    except Exception as e:
        print(f"  Warning: could not load VIX data ({e})")
        return {}


# --------------- per-day processing ---------------

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _process_day(g, d, events_lk, levels_lk, vixy_lk):
    """Analyse one (symbol, et_date) group.

    Returns a dict with candle stats, sweep results, and all factor
    columns, or None if the day doesn't qualify.
    """
    candle_bars = g.loc[_within(g["ts_et"], CANDLE_START, CANDLE_END)]
    if candle_bars.empty:
        return None

    candle_high = candle_bars["high"].max()
    candle_low = candle_bars["low"].min()
    candle_open = candle_bars.iloc[0]["open"]
    candle_close = candle_bars.iloc[-1]["close"]
    candle_range = round(candle_high - candle_low, 2)

    if candle_range <= 0:
        return None

    dist_to_high = round(candle_high - candle_close, 2)
    dist_to_low = round(candle_close - candle_low, 2)

    if max(dist_to_high, dist_to_low) < MIN_DISTANCE:
        return None

    # -- sweep check --
    sweep_bars = g.loc[_within(g["ts_et"], SWEEP_START, SWEEP_END)]
    h_swept, h_time, _ = _check_sweep(sweep_bars, candle_high, is_high=True)
    l_swept, l_time, _ = _check_sweep(sweep_bars, candle_low, is_high=False)

    rec = {
        "candle_high": candle_high,
        "candle_low": candle_low,
        "candle_open": candle_open,
        "candle_close": candle_close,
        "candle_range": candle_range,
        "dist_to_high": dist_to_high,
        "dist_to_low": dist_to_low,
        "high_swept": int(h_swept),
        "low_swept": int(l_swept),
        "either_swept": int(h_swept or l_swept),
        "both_swept": int(h_swept and l_swept),
        "high_sweep_time": h_time,
        "low_sweep_time": l_time,
    }

    # ======== TIER 1 FACTORS ========

    # 1. Close position in candle (0 = at low, 1 = at high)
    rec["close_position"] = round((candle_close - candle_low) / candle_range, 3)

    # 2. Last 15m candle (9:45-10:00) direction & strength
    last_15m = g.loc[_within(g["ts_et"], LAST_15M_START, CANDLE_END)]
    if not last_15m.empty:
        l15_open = last_15m.iloc[0]["open"]
        l15_close = last_15m.iloc[-1]["close"]
        l15_high = last_15m["high"].max()
        l15_low = last_15m["low"].min()
        l15_range = round(l15_high - l15_low, 2)
        l15_body = abs(l15_close - l15_open)
        rec["last_15m_direction"] = "bullish" if l15_close > l15_open else "bearish"
        rec["last_15m_range"] = l15_range
        rec["last_15m_body_pct"] = round(100 * l15_body / l15_range, 1) if l15_range > 0 else 0.0
    else:
        rec["last_15m_direction"] = None
        rec["last_15m_range"] = np.nan
        rec["last_15m_body_pct"] = np.nan

    # 3. When the candle high/low were formed
    high_idx = candle_bars["high"].idxmax()
    low_idx = candle_bars["low"].idxmin()
    high_set_ts = candle_bars.at[high_idx, "ts_et"]
    low_set_ts = candle_bars.at[low_idx, "ts_et"]
    rec["high_set_time"] = high_set_ts
    rec["low_set_time"] = low_set_ts
    rec["high_in_rth"] = int(high_set_ts.time() >= NY_OPEN)
    rec["low_in_rth"] = int(low_set_ts.time() >= NY_OPEN)

    # 4. Nearest unswept Asia/London level to each boundary
    levels = levels_lk.get(d, [])
    if levels:
        dists_h = [abs(lv - candle_high) for lv in levels]
        dists_l = [abs(lv - candle_low) for lv in levels]
        rec["nearest_unswept_to_high"] = round(min(dists_h), 2)
        rec["nearest_unswept_to_low"] = round(min(dists_l), 2)
        rec["unswept_near_high"] = int(min(dists_h) <= UNSWEPT_NEAR_THRESHOLD)
        rec["unswept_near_low"] = int(min(dists_l) <= UNSWEPT_NEAR_THRESHOLD)
    else:
        rec["nearest_unswept_to_high"] = np.nan
        rec["nearest_unswept_to_low"] = np.nan
        rec["unswept_near_high"] = 0
        rec["unswept_near_low"] = 0

    # 5. Economic events
    day_events = events_lk.get(d, [])
    rec["has_10am_event"] = int(any(t == "10:00" for t, _ in day_events))
    rec["has_830am_event"] = int(any(t == "08:30" for t, _ in day_events))
    rec["event_names"] = ", ".join(n for _, n in day_events) if day_events else ""

    # ======== TIER 2 FACTORS ========

    # 6. RTH range as % of 4H candle range
    rth_bars = g.loc[_within(g["ts_et"], NY_OPEN, CANDLE_END)]
    if not rth_bars.empty:
        rth_range = rth_bars["high"].max() - rth_bars["low"].min()
        rec["rth_range"] = round(rth_range, 2)
        rec["rth_range_pct"] = round(100 * rth_range / candle_range, 1)
    else:
        rec["rth_range"] = np.nan
        rec["rth_range_pct"] = np.nan

    # 7. Pre-RTH range (earliest bar on this date through 9:30)
    pre_rth = g.loc[g["ts_et"].dt.time < NY_OPEN]
    if not pre_rth.empty:
        rec["pre_rth_range"] = round(pre_rth["high"].max() - pre_rth["low"].min(), 2)
    else:
        rec["pre_rth_range"] = np.nan

    # ======== TIER 3 FACTORS ========

    # 8. VIXY previous-day close
    rec["vixy_prev_close"] = vixy_lk.get(d, np.nan)

    # 9. Day of week
    rec["day_of_week"] = DOW_NAMES[d.weekday()] if isinstance(d, date_type) else ""

    return rec


# --------------- factor analysis ---------------

def _bucket_analysis(detail, factor_col, buckets, swept_cols):
    """Compute sweep rates for a factor, grouped by bucket labels.

    buckets: list of (label, mask_series) tuples, OR 'categorical' to use
             unique values of factor_col.
    """
    rows = []
    if buckets == "categorical":
        for val in sorted(detail[factor_col].dropna().unique()):
            mask = detail[factor_col] == val
            sub = detail[mask]
            n = len(sub)
            if n == 0:
                continue
            row = {"factor": factor_col, "category": str(val), "days": n}
            for sc in swept_cols:
                s = int(sub[sc].sum())
                row[f"{sc}_count"] = s
                row[f"{sc}_pct"] = round(100 * s / n, 1)
            rows.append(row)
    else:
        for label, lo, hi in buckets:
            mask = (detail[factor_col] >= lo) & (detail[factor_col] < hi)
            sub = detail[mask]
            n = len(sub)
            if n == 0:
                continue
            row = {"factor": factor_col, "category": label, "days": n}
            for sc in swept_cols:
                s = int(sub[sc].sum())
                row[f"{sc}_count"] = s
                row[f"{sc}_pct"] = round(100 * s / n, 1)
            rows.append(row)
    return rows


def _build_factor_analysis(detail):
    swept_cols = ["high_swept", "low_swept", "either_swept"]
    all_rows = []

    # Close position (0=at low, 1=at high)
    all_rows += _bucket_analysis(detail, "close_position", [
        ("0-0.25 (near low)", 0, 0.25),
        ("0.25-0.50", 0.25, 0.50),
        ("0.50-0.75", 0.50, 0.75),
        ("0.75-1.0 (near high)", 0.75, 1.01),
    ], swept_cols)

    # Last 15m direction
    all_rows += _bucket_analysis(detail, "last_15m_direction", "categorical", swept_cols)

    # Last 15m body %
    all_rows += _bucket_analysis(detail, "last_15m_body_pct", [
        ("0-25 (doji-like)", 0, 25),
        ("25-50", 25, 50),
        ("50-75", 50, 75),
        ("75-100 (strong body)", 75, 101),
    ], swept_cols)

    # High set in RTH
    all_rows += _bucket_analysis(detail, "high_in_rth", "categorical", swept_cols)

    # Low set in RTH
    all_rows += _bucket_analysis(detail, "low_in_rth", "categorical", swept_cols)

    # Unswept near high
    all_rows += _bucket_analysis(detail, "unswept_near_high", "categorical", swept_cols)

    # Unswept near low
    all_rows += _bucket_analysis(detail, "unswept_near_low", "categorical", swept_cols)

    # 10am event
    all_rows += _bucket_analysis(detail, "has_10am_event", "categorical", swept_cols)

    # 8:30am event
    all_rows += _bucket_analysis(detail, "has_830am_event", "categorical", swept_cols)

    # Candle range
    all_rows += _bucket_analysis(detail, "candle_range", [
        ("0-40", 0, 40),
        ("40-80", 40, 80),
        ("80-120", 80, 120),
        ("120-160", 120, 160),
        ("160+", 160, 9999),
    ], swept_cols)

    # RTH range %
    all_rows += _bucket_analysis(detail, "rth_range_pct", [
        ("0-25%", 0, 25),
        ("25-50%", 25, 50),
        ("50-75%", 50, 75),
        ("75-100%", 75, 101),
    ], swept_cols)

    # Pre-RTH range
    all_rows += _bucket_analysis(detail, "pre_rth_range", [
        ("0-40", 0, 40),
        ("40-80", 40, 80),
        ("80-120", 80, 120),
        ("120+", 120, 9999),
    ], swept_cols)

    # VIXY prev close
    vixy_valid = detail.dropna(subset=["vixy_prev_close"])
    if not vixy_valid.empty:
        all_rows += _bucket_analysis(vixy_valid, "vixy_prev_close", [
            ("0-15 (low vol)", 0, 15),
            ("15-20", 15, 20),
            ("20-25", 20, 25),
            ("25-30", 25, 30),
            ("30+ (high vol)", 30, 9999),
        ], swept_cols)

    # Day of week
    all_rows += _bucket_analysis(detail, "day_of_week", "categorical", swept_cols)

    return pd.DataFrame(all_rows)


# --------------- main ---------------

def main():
    tz = pytz.timezone("US/Eastern")

    print("Loading external lookups …")
    events_lk = _load_events()
    levels_lk = _load_session_levels()
    vixy_lk = _load_vixy()
    print(f"  Events: {len(events_lk)} dates, "
          f"Session levels: {len(levels_lk)} dates, "
          f"VIXY: {len(vixy_lk)} dates")

    all_detail = []
    residual = pd.DataFrame()
    chunk_n = 0

    print("Loading NQ data …")
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
        df["et_date"] = df["ts_et"].dt.date
        df = df[df["symbol"].astype(str).str.match(NQ_RE)]

        if WEEKDAYS_ONLY:
            df = df[df["ts_et"].dt.weekday <= 4]
        if df.empty:
            residual = pd.DataFrame()
            continue

        max_date = df["et_date"].max()
        to_proc = df[df["et_date"] < max_date]
        residual = df[df["et_date"] == max_date]

        if to_proc.empty:
            continue

        to_proc = _front_month(to_proc)
        n_before = len(all_detail)
        for (sym, d), g in to_proc.groupby(["symbol", "et_date"]):
            g = g.sort_values("ts_et")
            rec = _process_day(g, d, events_lk, levels_lk, vixy_lk)
            if rec is not None:
                rec["date"] = d
                rec["symbol"] = sym
                all_detail.append(rec)
        print(f"  chunk {chunk_n}: +{len(all_detail) - n_before} rows  (total {len(all_detail)})")

    if not residual.empty:
        residual = _front_month(residual)
        for (sym, d), g in residual.groupby(["symbol", "et_date"]):
            g = g.sort_values("ts_et")
            rec = _process_day(g, d, events_lk, levels_lk, vixy_lk)
            if rec is not None:
                rec["date"] = d
                rec["symbol"] = sym
                all_detail.append(rec)

    if not all_detail:
        print("No qualifying days found.")
        return

    # ---- detailed CSV ----
    detail = pd.DataFrame(all_detail)
    col_order = [
        # core
        "date", "symbol", "candle_high", "candle_low", "candle_open",
        "candle_close", "candle_range", "dist_to_high", "dist_to_low",
        "high_swept", "low_swept", "either_swept", "both_swept",
        "high_sweep_time", "low_sweep_time",
        # tier 1
        "close_position", "last_15m_direction", "last_15m_range",
        "last_15m_body_pct", "high_set_time", "low_set_time",
        "high_in_rth", "low_in_rth",
        "nearest_unswept_to_high", "nearest_unswept_to_low",
        "unswept_near_high", "unswept_near_low",
        "has_10am_event", "has_830am_event", "event_names",
        # tier 2
        "rth_range", "rth_range_pct", "pre_rth_range",
        # tier 3
        "vixy_prev_close", "day_of_week",
    ]
    detail = detail[[c for c in col_order if c in detail.columns]].sort_values("date")
    detail.to_csv(DETAILED_OUT, index=False)
    print(f"\n✅ Wrote {DETAILED_OUT}  ({len(detail)} rows)")

    # ---- summary ----
    n = len(detail)
    pct = lambda v: round(100 * v / n, 1) if n else 0.0

    h_swept = int(detail["high_swept"].sum())
    l_swept = int(detail["low_swept"].sum())
    either = int(detail["either_swept"].sum())
    both = int(detail["both_swept"].sum())
    neither = n - either

    summary = pd.DataFrame([{
        "qualifying_days": n,
        "high_swept": h_swept,
        "high_swept_pct": pct(h_swept),
        "low_swept": l_swept,
        "low_swept_pct": pct(l_swept),
        "either_swept": either,
        "either_swept_pct": pct(either),
        "both_swept": both,
        "both_swept_pct": pct(both),
        "neither_swept": neither,
        "neither_swept_pct": pct(neither),
    }])
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"✅ Wrote {SUMMARY_OUT}")
    print()
    print(summary.T.to_string(header=False))

    # ---- distance bucket analysis ----
    bucket_rows = []
    for side, dist_col, swept_col in [
        ("High", "dist_to_high", "high_swept"),
        ("Low", "dist_to_low", "low_swept"),
    ]:
        side_df = detail[detail[dist_col] >= MIN_DISTANCE].copy()
        if side_df.empty:
            continue
        side_df["bucket"] = side_df[dist_col].apply(_distance_bucket)

        for bucket, grp in side_df.groupby("bucket", sort=False):
            cnt = len(grp)
            swept = int(grp[swept_col].sum())
            bucket_rows.append({
                "side": side,
                "distance_bucket": bucket,
                "days": cnt,
                "swept": swept,
                "swept_pct": round(100 * swept / cnt, 1) if cnt else 0.0,
            })

    if bucket_rows:
        buckets = pd.DataFrame(bucket_rows)
        bucket_order = [f"{BUCKET_EDGES[i-1] if i else 0}-{e}" for i, e in enumerate(BUCKET_EDGES)]
        bucket_order.append(f"{BUCKET_EDGES[-1]}+")
        buckets["_sort"] = buckets["distance_bucket"].apply(
            lambda b: bucket_order.index(b) if b in bucket_order else 999
        )
        buckets = buckets.sort_values(["side", "_sort"]).drop(columns=["_sort"])
        buckets.to_csv(BUCKETS_OUT, index=False)
        print(f"✅ Wrote {BUCKETS_OUT}")
        print()
        print(buckets.to_string(index=False))

    # ---- factor analysis ----
    factor_df = _build_factor_analysis(detail)
    if not factor_df.empty:
        factor_df.to_csv(FACTOR_OUT, index=False)
        print(f"\n✅ Wrote {FACTOR_OUT}")
        print()
        print(factor_df.to_string(index=False))


if __name__ == "__main__":
    main()
