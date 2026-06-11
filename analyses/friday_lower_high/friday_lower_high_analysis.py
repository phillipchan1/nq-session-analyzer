#!/usr/bin/env python3
"""
Friday Lower High Setup: Monday Visit Probability of Friday's Low
-----------------------------------------------------------------
Backtest claim: When Friday's RTH high is LOWER than Thursday's high,
Friday's low will be visited on Monday with "overwhelmingly high" odds.

Since we trade only the first 45 minutes (9:30-10:15), the critical
question is WHEN that visit occurs.

Outputs:
    friday_lower_high_summary.csv
    friday_lower_high_timing.csv
    friday_lower_high_cumulative.csv
    friday_lower_high_detailed.csv
    friday_lower_high_distance_buckets.csv
    friday_lower_high_baseline.csv
    friday_lower_high_symmetric.csv
"""

from pathlib import Path
from datetime import time, timedelta, date
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import pytz
import re

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20260221.ohlcv-1m.csv")

RTH_START = time(9, 30)
RTH_END = time(16, 0)
FIRST_45_END = time(10, 15)
NQ_TICK = 0.25
TOUCH_EPSILON = 0.1

CUMULATIVE_WINDOWS = [
    ("10:15", time(10, 15)),
    ("10:30", time(10, 30)),
    ("11:00", time(11, 0)),
    ("12:00", time(12, 0)),
    ("16:00", time(16, 0)),
]

TIMING_BUCKETS = [
    ("0-5", 0, 5),
    ("5-10", 5, 10),
    ("10-15", 10, 15),
    ("15-20", 15, 20),
    ("20-25", 20, 25),
    ("25-30", 25, 30),
    ("30-45", 30, 45),
    ("after_45", 45, 9999),
]

DISTANCE_BUCKETS = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 9999)]

CHUNKSIZE = 1_000_000
NQ_RE = re.compile(r"^NQ[A-Z]\d{1,2}$")
# ==============================================


def _within(ts, start_t, end_t):
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)


def _get_us_market_holidays(start_year: int = 2020, end_year: int = 2027) -> set:
    """US market holidays (NYSE/NASDAQ closures)."""
    holidays = set()
    for year in range(start_year, end_year + 1):
        # New Year's Day
        ny = date(year, 1, 1)
        if ny.weekday() == 5:
            holidays.add(date(year - 1, 12, 31))
        elif ny.weekday() == 6:
            holidays.add(date(year, 1, 2))
        else:
            holidays.add(ny)
        # MLK Day
        jan_mondays = [d for d in [date(year, 1, d) for d in range(1, 32)] if d.weekday() == 0]
        if len(jan_mondays) >= 3:
            holidays.add(jan_mondays[2])
        # Presidents Day
        pres_mondays = [d for d in [date(year, 2, d) for d in range(1, 29)] if d.weekday() == 0]
        holidays.add(pres_mondays[2] if len(pres_mondays) > 2 else pres_mondays[-1])
        # Good Friday
        a, b, c = year % 19, year // 100, year % 100
        d, e, f = b // 4, b % 4, (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = c // 4, c % 4
        l_ = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l_) // 451
        month = (h + l_ - 7 * m + 114) // 31
        day = ((h + l_ - 7 * m + 114) % 31) + 1
        easter = date(year, month, day)
        holidays.add(easter - timedelta(days=2))
        # Memorial Day
        may_mondays = [d for d in [date(year, 5, d) for d in range(1, 32)] if d.weekday() == 0]
        holidays.add(may_mondays[-1])
        # Juneteenth
        if year >= 2021:
            jun = date(year, 6, 19)
            holidays.add(jun if jun.weekday() < 5 else (date(year, 6, 18) if jun.weekday() == 5 else date(year, 6, 20)))
        # Independence Day
        july4 = date(year, 7, 4)
        holidays.add(july4 if july4.weekday() < 5 else (date(year, 7, 3) if july4.weekday() == 5 else date(year, 7, 5)))
        # Labor Day
        sep_mondays = [d for d in [date(year, 9, d) for d in range(1, 31)] if d.weekday() == 0]
        holidays.add(sep_mondays[0])
        # Thanksgiving
        nov_thurs = [d for d in [date(year, 11, d) for d in range(1, 31)] if d.weekday() == 3]
        holidays.add(nov_thurs[3] if len(nov_thurs) > 3 else nov_thurs[-1])
        # Christmas
        xmas = date(year, 12, 25)
        if xmas.weekday() == 5:
            holidays.add(date(year, 12, 24))
        elif xmas.weekday() == 6:
            holidays.add(date(year, 12, 26))
        else:
            holidays.add(xmas)
    return holidays


def _next_trading_day(d: date, holidays: set) -> Optional[date]:
    """Next trading day after d, skipping weekends and holidays."""
    cand = d + timedelta(days=1)
    while cand.weekday() >= 5 or cand in holidays:
        cand += timedelta(days=1)
    return cand


def _front_month(df: pd.DataFrame) -> pd.DataFrame:
    rth = df.loc[_within(df["ts_et"], RTH_START, time(9, 45))]
    if rth.empty:
        return df
    vol = rth.groupby(["et_date", "symbol"]).agg(volume=("volume", "sum")).reset_index()
    best = vol.loc[vol.groupby("et_date")["volume"].idxmax(), ["et_date", "symbol"]].rename(columns={"symbol": "front"})
    out = df.merge(best, on="et_date", how="left")
    return out.loc[out["symbol"] == out["front"]].drop(columns=["front"])


def _check_sweep(bars: pd.DataFrame, level: float, is_high: bool) -> Tuple[bool, Optional[pd.Timestamp]]:
    """Strict sweep: price exceeds level by >= 1 NQ tick. Returns (swept, first_time)."""
    if bars.empty:
        return False, None
    if is_high:
        mask = bars["high"] > level + NQ_TICK
    else:
        mask = bars["low"] < level - NQ_TICK
    if mask.any():
        idx = bars[mask].index[0]
        return True, bars.at[idx, "ts_et"]
    return False, None


def _check_touch(bars: pd.DataFrame, level: float, is_high: bool) -> Tuple[bool, Optional[pd.Timestamp]]:
    """Touch: any hit within epsilon. Returns (touched, first_time)."""
    if bars.empty:
        return False, None
    if is_high:
        mask = bars["high"] >= level - TOUCH_EPSILON
    else:
        mask = bars["low"] <= level + TOUCH_EPSILON
    if mask.any():
        idx = bars[mask].index[0]
        return True, bars.at[idx, "ts_et"]
    return False, None


def _minutes_from_open(ts: pd.Timestamp, open_time: time, day_date: date, tz) -> float:
    """Minutes from 9:30 open to ts."""
    open_dt = tz.localize(pd.Timestamp.combine(day_date, open_time))
    hit_ts = pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts
    return (hit_ts - open_dt).total_seconds() / 60


def _distance_bucket(dist: float) -> str:
    d = abs(dist)
    for lo, hi in DISTANCE_BUCKETS:
        if d < hi:
            return f"{lo}-{hi}" if hi < 9999 else f"{lo}+"
    return f"{DISTANCE_BUCKETS[-1][0]}+"


def main():
    tz = pytz.timezone("US/Eastern")
    holidays = _get_us_market_holidays()

    # Build daily RTH high/low lookup
    print("Building daily RTH high/low...")
    daily_hl: Dict[date, Dict] = {}
    residual = pd.DataFrame()

    for chunk in pd.read_csv(
        CSV_FILE,
        usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
        dtype={"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "int64", "symbol": "string"},
        chunksize=CHUNKSIZE,
    ):
        df = pd.concat([residual, chunk], ignore_index=True)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz)
        df["et_date"] = df["ts_et"].dt.date
        df = df[df["symbol"].astype(str).str.match(NQ_RE)]
        df = df[df["ts_et"].dt.weekday <= 4]

        if df.empty:
            residual = pd.DataFrame()
            continue

        max_date = df["et_date"].max()
        to_proc = df[df["et_date"] < max_date]
        residual = df[df["et_date"] == max_date]

        to_proc = _front_month(to_proc)
        for (_, d), g in to_proc.groupby(["symbol", "et_date"]):
            rth = g.loc[_within(g["ts_et"], RTH_START, RTH_END)]
            if rth.empty:
                continue
            open_930 = g.loc[_within(g["ts_et"], RTH_START, time(9, 31))]
            open_price = float(open_930.iloc[0]["open"]) if not open_930.empty else np.nan
            daily_hl[d] = {
                "rth_high": float(rth["high"].max()),
                "rth_low": float(rth["low"].min()),
                "rth_close": float(rth.iloc[-1]["close"]),
                "open_930": open_price,
                "symbol": g["symbol"].iloc[0],
            }

    if not residual.empty:
        residual = _front_month(residual)
        for (_, d), g in residual.groupby(["symbol", "et_date"]):
            rth = g.loc[_within(g["ts_et"], RTH_START, RTH_END)]
            if rth.empty:
                continue
            open_930 = g.loc[_within(g["ts_et"], RTH_START, time(9, 31))]
            open_price = float(open_930.iloc[0]["open"]) if not open_930.empty else np.nan
            daily_hl[d] = {
                "rth_high": float(rth["high"].max()),
                "rth_low": float(rth["low"].min()),
                "rth_close": float(rth.iloc[-1]["close"]),
                "open_930": open_price,
                "symbol": g["symbol"].iloc[0],
            }

    print(f"  Loaded {len(daily_hl)} trading days")

    # Identify Fridays with Thu/Fri data
    fridays = [d for d in sorted(daily_hl.keys()) if d.weekday() == 4]
    thursday = {d: d - timedelta(days=1) for d in fridays}

    # Qualifying Fridays: Friday high < Thursday high
    qualifying = []
    for fri in fridays:
        thu = thursday[fri]
        if thu not in daily_hl or fri not in daily_hl:
            continue
        if fri in holidays or thu in holidays:
            continue
        mon = _next_trading_day(fri, holidays)
        if mon not in daily_hl:
            continue
        fh = daily_hl[fri]["rth_high"]
        fl = daily_hl[fri]["rth_low"]
        th = daily_hl[thu]["rth_high"]
        tl = daily_hl[thu]["rth_low"]
        if fh < th:
            qualifying.append({
                "friday": fri,
                "thursday": thu,
                "monday": mon,
                "friday_high": fh,
                "friday_low": fl,
                "thursday_high": th,
                "thursday_low": tl,
                "monday_open": daily_hl[mon]["open_930"],
                "distance_open_to_friday_low": daily_hl[mon]["open_930"] - fl,
            })

    print(f"Qualifying Fridays (Fri high < Thu high): {len(qualifying)}")

    # All Fridays with valid Thu/Fri/Mon for baseline (need Monday bars for all)
    all_friday_mondays = set()
    for fri in fridays:
        thu = fri - timedelta(days=1)
        if thu not in daily_hl or fri not in daily_hl or fri in holidays or thu in holidays:
            continue
        mon = _next_trading_day(fri, holidays)
        if mon in daily_hl:
            all_friday_mondays.add(mon)
    monday_dates = sorted(all_friday_mondays)

    # Re-load 1m data for Monday bars to check visit
    print("Loading Monday 1m bars for visit checks...")
    monday_bars: Dict[date, pd.DataFrame] = {}
    residual = pd.DataFrame()

    for chunk in pd.read_csv(
        CSV_FILE,
        usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
        dtype={"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "int64", "symbol": "string"},
        chunksize=CHUNKSIZE,
    ):
        df = pd.concat([residual, chunk], ignore_index=True)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz)
        df["et_date"] = df["ts_et"].dt.date
        df = df[df["symbol"].astype(str).str.match(NQ_RE)]
        df = df[df["ts_et"].dt.weekday <= 4]

        if df.empty:
            residual = pd.DataFrame()
            continue

        max_date = df["et_date"].max()
        to_proc = df[df["et_date"] < max_date]
        residual = df[df["et_date"] == max_date]

        to_proc = _front_month(to_proc)
        for d in monday_dates:
            if d in to_proc["et_date"].values and d not in monday_bars:
                g = to_proc[to_proc["et_date"] == d]
                monday_bars[d] = g.sort_values("ts_et").reset_index(drop=True)

    if not residual.empty:
        residual = _front_month(residual)
        for d in monday_dates:
            if d not in monday_bars and d in residual["et_date"].values:
                g = residual[residual["et_date"] == d]
                monday_bars[d] = g.sort_values("ts_et").reset_index(drop=True)

    # Process each qualifying Friday
    detail_rows = []
    for q in qualifying:
        mon = q["monday"]
        friday_low = q["friday_low"]
        monday_open = q["monday_open"]
        dist = q["distance_open_to_friday_low"]

        if mon not in monday_bars:
            continue

        bars = monday_bars[mon]
        rth_bars = bars.loc[_within(bars["ts_et"], RTH_START, RTH_END)]

        # Sweep and touch within first 45 min
        first_45 = bars.loc[_within(bars["ts_et"], RTH_START, FIRST_45_END)]
        sweep_45, sweep_time = _check_sweep(first_45, friday_low, is_high=False)
        touch_45, touch_time = _check_touch(first_45, friday_low, is_high=False)

        # Sweep/touch by EOD
        sweep_eod, sweep_eod_time = _check_sweep(rth_bars, friday_low, is_high=False)
        touch_eod, touch_eod_time = _check_touch(rth_bars, friday_low, is_high=False)

        first_sweep_time = sweep_time if sweep_45 else sweep_eod_time
        first_touch_time = touch_time if touch_45 else touch_eod_time

        minutes_sweep = None
        minutes_touch = None
        if first_sweep_time is not None:
            minutes_sweep = _minutes_from_open(first_sweep_time, RTH_START, mon, tz)
        if first_touch_time is not None:
            minutes_touch = _minutes_from_open(first_touch_time, RTH_START, mon, tz)

        # Cumulative by window
        cum_sweep = {}
        cum_touch = {}
        for lbl, end_t in CUMULATIVE_WINDOWS:
            wbars = bars.loc[_within(bars["ts_et"], RTH_START, end_t)]
            cs, _ = _check_sweep(wbars, friday_low, is_high=False)
            ct, _ = _check_touch(wbars, friday_low, is_high=False)
            cum_sweep[lbl] = int(cs)
            cum_touch[lbl] = int(ct)

        # Friday close position
        friday_close = daily_hl[q["friday"]]["rth_close"]
        friday_range = q["friday_high"] - q["friday_low"]
        close_position = (friday_close - q["friday_low"]) / friday_range if friday_range > 0 else np.nan

        detail_rows.append({
            "friday": q["friday"],
            "thursday": q["thursday"],
            "monday": mon,
            "friday_high": q["friday_high"],
            "friday_low": q["friday_low"],
            "thursday_high": q["thursday_high"],
            "monday_open": monday_open,
            "distance_open_to_friday_low": round(dist, 2),
            "distance_bucket": _distance_bucket(dist),
            "sweep_45min": int(sweep_45),
            "touch_45min": int(touch_45),
            "sweep_eod": int(sweep_eod),
            "touch_eod": int(touch_eod),
            "first_sweep_time": first_sweep_time,
            "first_touch_time": first_touch_time,
            "minutes_to_sweep": round(minutes_sweep, 1) if minutes_sweep is not None else None,
            "minutes_to_touch": round(minutes_touch, 1) if minutes_touch is not None else None,
            **{f"sweep_by_{lbl.replace(':', '')}": cum_sweep[lbl] for lbl, _ in CUMULATIVE_WINDOWS},
            **{f"touch_by_{lbl.replace(':', '')}": cum_touch[lbl] for lbl, _ in CUMULATIVE_WINDOWS},
            "friday_close_position": round(close_position, 3) if not np.isnan(close_position) else np.nan,
        })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        print("No qualifying Fridays with Monday data. Exiting.")
        return

    # ---- Summary ----
    n = len(detail)
    summary = pd.DataFrame([{
        "qualifying_fridays": n,
        "sweep_45min": int(detail["sweep_45min"].sum()),
        "sweep_45min_pct": round(100 * detail["sweep_45min"].sum() / n, 1),
        "touch_45min": int(detail["touch_45min"].sum()),
        "touch_45min_pct": round(100 * detail["touch_45min"].sum() / n, 1),
        "sweep_eod": int(detail["sweep_eod"].sum()),
        "sweep_eod_pct": round(100 * detail["sweep_eod"].sum() / n, 1),
        "touch_eod": int(detail["touch_eod"].sum()),
        "touch_eod_pct": round(100 * detail["touch_eod"].sum() / n, 1),
    }])
    summary.to_csv(OUT_DIR / "friday_lower_high_summary.csv", index=False)
    print("\nWrote friday_lower_high_summary.csv")
    print(summary.T.to_string(header=False))

    # ---- Timing ----
    sweep_times = detail["minutes_to_sweep"].dropna()
    timing_rows = []
    for lbl, lo, hi in TIMING_BUCKETS:
        if hi < 9999:
            cnt = ((sweep_times >= lo) & (sweep_times < hi)).sum()
        else:
            cnt = (sweep_times >= lo).sum()
        timing_rows.append({"minutes_bucket": lbl, "count": int(cnt), "pct": round(100 * cnt / len(sweep_times), 1) if len(sweep_times) else 0})
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(OUT_DIR / "friday_lower_high_timing.csv", index=False)
    print("\nWrote friday_lower_high_timing.csv")

    # ---- Cumulative ----
    cum_rows = []
    for lbl, _ in CUMULATIVE_WINDOWS:
        col = f"sweep_by_{lbl.replace(':', '')}"
        cnt = int(detail[col].sum())
        cum_rows.append({"window_end": lbl, "swept": cnt, "swept_pct": round(100 * cnt / n, 1)})
    cum_df = pd.DataFrame(cum_rows)
    cum_df.to_csv(OUT_DIR / "friday_lower_high_cumulative.csv", index=False)
    print("\nWrote friday_lower_high_cumulative.csv")

    # ---- Distance buckets ----
    dist_rows = []
    for bucket, grp in detail.groupby("distance_bucket"):
        cnt = len(grp)
        swept_45 = int(grp["sweep_45min"].sum())
        swept_eod = int(grp["sweep_eod"].sum())
        dist_rows.append({
            "distance_bucket": bucket,
            "days": cnt,
            "sweep_45min": swept_45,
            "sweep_45min_pct": round(100 * swept_45 / cnt, 1) if cnt else 0,
            "sweep_eod": swept_eod,
            "sweep_eod_pct": round(100 * swept_eod / cnt, 1) if cnt else 0,
        })
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUT_DIR / "friday_lower_high_distance_buckets.csv", index=False)
    print("\nWrote friday_lower_high_distance_buckets.csv")

    # ---- Detailed ----
    detail_out = detail.copy()
    detail_out["friday"] = detail_out["friday"].astype(str)
    detail_out["thursday"] = detail_out["thursday"].astype(str)
    detail_out["monday"] = detail_out["monday"].astype(str)
    detail_out.to_csv(OUT_DIR / "friday_lower_high_detailed.csv", index=False)
    print("\nWrote friday_lower_high_detailed.csv (" + str(len(detail)) + " rows)")

    # ---- Baseline: all Fridays (Fri high >= Thu high or any Fri)
    baseline_rows = []
    for fri in fridays:
        thu = fri - timedelta(days=1)
        if thu not in daily_hl or fri not in daily_hl:
            continue
        if fri in holidays or thu in holidays:
            continue
        mon = _next_trading_day(fri, holidays)
        if mon not in daily_hl or mon not in monday_bars:
            continue
        fh, fl = daily_hl[fri]["rth_high"], daily_hl[fri]["rth_low"]
        th = daily_hl[thu]["rth_high"]
        bars = monday_bars[mon]
        first_45 = bars.loc[_within(bars["ts_et"], RTH_START, FIRST_45_END)]
        rth_bars = bars.loc[_within(bars["ts_et"], RTH_START, RTH_END)]
        sweep_45, _ = _check_sweep(first_45, fl, is_high=False)
        sweep_eod, _ = _check_sweep(rth_bars, fl, is_high=False)
        qualifies = fh < th
        baseline_rows.append({
            "friday": fri,
            "qualifies": qualifies,
            "sweep_45min": int(sweep_45),
            "sweep_eod": int(sweep_eod),
        })

    baseline = pd.DataFrame(baseline_rows)
    if not baseline.empty:
        qual = baseline[baseline["qualifies"]]
        non_qual = baseline[~baseline["qualifies"]]
        baseline_summary = pd.DataFrame([
            {"group": "qualifying", "days": len(qual), "sweep_45min": int(qual["sweep_45min"].sum()), "sweep_45min_pct": round(100 * qual["sweep_45min"].sum() / len(qual), 1) if len(qual) else 0, "sweep_eod": int(qual["sweep_eod"].sum()), "sweep_eod_pct": round(100 * qual["sweep_eod"].sum() / len(qual), 1) if len(qual) else 0},
            {"group": "non_qualifying", "days": len(non_qual), "sweep_45min": int(non_qual["sweep_45min"].sum()), "sweep_45min_pct": round(100 * non_qual["sweep_45min"].sum() / len(non_qual), 1) if len(non_qual) else 0, "sweep_eod": int(non_qual["sweep_eod"].sum()), "sweep_eod_pct": round(100 * non_qual["sweep_eod"].sum() / len(non_qual), 1) if len(non_qual) else 0},
            {"group": "all_fridays", "days": len(baseline), "sweep_45min": int(baseline["sweep_45min"].sum()), "sweep_45min_pct": round(100 * baseline["sweep_45min"].sum() / len(baseline), 1), "sweep_eod": int(baseline["sweep_eod"].sum()), "sweep_eod_pct": round(100 * baseline["sweep_eod"].sum() / len(baseline), 1)},
        ])
        baseline_summary.to_csv(OUT_DIR / "friday_lower_high_baseline.csv", index=False)
        print("\nWrote friday_lower_high_baseline.csv")

    # ---- Symmetric: Friday low > Thursday low -> Friday high visited on Monday?
    sym_rows = []
    for fri in fridays:
        thu = fri - timedelta(days=1)
        if thu not in daily_hl or fri not in daily_hl:
            continue
        if fri in holidays or thu in holidays:
            continue
        mon = _next_trading_day(fri, holidays)
        if mon not in monday_bars:
            continue
        fh, fl = daily_hl[fri]["rth_high"], daily_hl[fri]["rth_low"]
        tl = daily_hl[thu]["rth_low"]
        if fl <= tl:
            continue
        bars = monday_bars[mon]
        first_45 = bars.loc[_within(bars["ts_et"], RTH_START, FIRST_45_END)]
        rth_bars = bars.loc[_within(bars["ts_et"], RTH_START, RTH_END)]
        sweep_45, _ = _check_sweep(first_45, fh, is_high=True)
        sweep_eod, _ = _check_sweep(rth_bars, fh, is_high=True)
        sym_rows.append({"friday": fri, "sweep_45min": int(sweep_45), "sweep_eod": int(sweep_eod)})

    if sym_rows:
        sym_df = pd.DataFrame(sym_rows)
        sym_summary = pd.DataFrame([{
            "qualifying_fridays": len(sym_df),
            "sweep_45min": int(sym_df["sweep_45min"].sum()),
            "sweep_45min_pct": round(100 * sym_df["sweep_45min"].sum() / len(sym_df), 1),
            "sweep_eod": int(sym_df["sweep_eod"].sum()),
            "sweep_eod_pct": round(100 * sym_df["sweep_eod"].sum() / len(sym_df), 1),
        }])
        sym_summary.to_csv(OUT_DIR / "friday_lower_high_symmetric.csv", index=False)
        print("\nWrote friday_lower_high_symmetric.csv (Fri low > Thu low -> Fri high visited)")

    print("\nDone.")


if __name__ == "__main__":
    main()
