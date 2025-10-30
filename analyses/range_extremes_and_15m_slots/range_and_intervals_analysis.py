#!/usr/bin/env python3
"""
Range Extremes and NY Session 15-minute Interval Analysis
---------------------------------------------------------

Outputs:
- daily_session_metrics.csv: per-day features and ranges (overnight, Asia, London, RTH, gaps)
- extreme_range_correlations.csv: conditional extreme-day rates by feature bins
- ny_15m_interval_summary.csv: summary stats per 15-minute slot 9:30–16:00
- ny_15m_interval_by_day.csv: per-day 15-minute ranges (for ad-hoc deep dives)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import time, timedelta
import pytz


# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20250926.ohlcv-1m.csv")
EVENTS_FILE = str(DATA_DIR / "us_high_impact_events_2020_to_2025.csv")

# Threshold for an "extreme" RTH day range (points)
EXTREME_RTH_RANGE = 200.0

# Session time boundaries (ET)
ASIA_START, ASIA_END = time(21, 0), time(3, 0)    # wraps midnight
LONDON_START, LONDON_END = time(3, 0), time(8, 0)
RTH_START, RTH_END = time(9, 30), time(16, 0)

# Chunking
CHUNKSIZE = 1_000_000


def _within(ts: pd.Series, start_t: time, end_t: time) -> pd.Series:
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)


def _within_wrap(ts: pd.Series, start_t: time, end_t: time) -> pd.Series:
    """Handles wrapping intervals like 21:00–03:00."""
    t = ts.dt.time
    return (t >= start_t) | (t < end_t)


def _load_events() -> pd.DataFrame:
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
    """Filter to front-month per date by max range in 9:30–10:15 ET window (liquidity proxy)."""
    mask = _within(df["ts_et"], RTH_START, time(10, 15))
    rth = df.loc[mask]
    if rth.empty:
        return df
    agg = rth.groupby(["et_date", "symbol"]).agg(high=("high", "max"), low=("low", "min")).reset_index()
    agg["range"] = agg["high"] - agg["low"]
    idx = agg.groupby("et_date")["range"].idxmax()
    fm = agg.loc[idx][["et_date", "symbol"]].rename(columns={"symbol": "front_symbol"})
    out = df.merge(fm, on="et_date", how="left")
    out = out.loc[out["symbol"] == out["front_symbol"]].drop(columns=["front_symbol"])
    return out


def _session_range(df: pd.DataFrame, start_t: time, end_t: time, wrap: bool = False) -> Tuple[float, float, float]:
    if df.empty:
        return 0.0, np.nan, np.nan
    mask = _within_wrap(df["ts_et"], start_t, end_t) if wrap else _within(df["ts_et"], start_t, end_t)
    seg = df.loc[mask]
    if seg.empty:
        return 0.0, np.nan, np.nan
    return float(seg["high"].max() - seg["low"].min()), float(seg.iloc[0]["open"]), float(seg.iloc[-1]["close"])


def _build_15m_slots() -> List[Tuple[time, time, str]]:
    slots: List[Tuple[time, time, str]] = []
    start = pd.Timestamp.combine(pd.to_datetime("2000-01-01").date(), RTH_START)
    end = pd.Timestamp.combine(pd.to_datetime("2000-01-01").date(), RTH_END)
    cur = start
    while cur < end:
        nxt = cur + pd.Timedelta(minutes=15)
        label = f"{cur.time().strftime('%H:%M')}-{nxt.time().strftime('%H:%M')}"
        slots.append((cur.time(), nxt.time(), label))
        cur = nxt
    return slots


def compute_daily_and_intervals() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tz_et = pytz.timezone("US/Eastern")
    residual = pd.DataFrame()
    events = _load_events()

    daily_rows: List[Dict] = []
    intervals_rows: List[Dict] = []
    slots = _build_15m_slots()

    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    dtypes = {"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64", "symbol": "string"}

    for chunk in pd.read_csv(CSV_FILE, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE):
        df = pd.concat([residual, chunk], ignore_index=True)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz_et)
        df["et_date"] = df["ts_et"].dt.date

        # Weekdays only
        df = df[df["ts_et"].dt.weekday <= 4]
        if df.empty:
            continue

        # Keep only outright NQ contracts (avoid spreads)
        df = df[df["symbol"].astype(str).str.startswith("NQ") & ~df["symbol"].astype(str).str.contains("-")]

        # Split off the final day to complete next chunk
        max_date = df["et_date"].max()
        to_proc = df[df["et_date"] < max_date]
        residual = df[df["et_date"] == max_date]

        # Filter to front-month per date
        if not to_proc.empty:
            to_proc = _determine_front_month(to_proc)

        for (sym, d), g in to_proc.groupby(["symbol", "et_date"]):
            g = g.sort_values("ts_et").reset_index(drop=True)

            # Overnight session: 18:00 prev day to 9:30 current day
            start_ts = (pd.Timestamp(d) - pd.Timedelta(days=1)).replace(hour=18, minute=0)
            start_ts = tz_et.localize(start_ts)
            end_ts = tz_et.localize(pd.Timestamp.combine(d, RTH_START))
            on_mask = (g["ts_et"] >= start_ts) & (g["ts_et"] < end_ts)
            overnight = g.loc[on_mask]
            if not overnight.empty:
                on_range = float(overnight["high"].max() - overnight["low"].min())
                on_open = float(overnight.iloc[0]["open"])
                on_close = float(overnight.iloc[-1]["close"])
                on_dir = "Up" if on_close > on_open else "Down"
            else:
                on_range = 0.0; on_dir = "None"

            # Asia (wrap) and London ranges
            asia_range, _, _ = _session_range(g, ASIA_START, ASIA_END, wrap=True)
            london_range, _, _ = _session_range(g, LONDON_START, LONDON_END, wrap=False)

            # Previous close (approx: last bar before 16:00) and 9:30 open for gap
            prev_date = d - timedelta(days=1)
            prev_rth_mask = (g["ts_et"].dt.date == prev_date) & _within(g["ts_et"], RTH_START, RTH_END)
            prev_rth = g.loc[prev_rth_mask]
            if not prev_rth.empty:
                prev_close = float(prev_rth.iloc[-1]["close"])
            else:
                prev_close = np.nan
            open_bar = g.loc[_within(g["ts_et"], RTH_START, time(9, 31))]
            cur_open = float(open_bar.iloc[0]["open"]) if not open_bar.empty else np.nan
            pre_open_gap = float(cur_open - prev_close) if (pd.notna(cur_open) and pd.notna(prev_close)) else np.nan

            # RTH range (full 9:30–16:00)
            rth_mask = _within(g["ts_et"], RTH_START, RTH_END)
            rth = g.loc[rth_mask]
            rth_range = float(rth["high"].max() - rth["low"].min()) if not rth.empty else 0.0
            is_extreme = int(rth_range >= EXTREME_RTH_RANGE)

            daily_rows.append({
                "date": d,
                "symbol": sym,
                "overnight_range": on_range,
                "overnight_direction": on_dir,
                "asia_range": asia_range,
                "london_range": london_range,
                "pre_open_gap": pre_open_gap,
                "rth_range": rth_range,
                "extreme_rth": is_extreme,
            })

            # 15-minute slots across RTH
            slot_row = {"date": d, "symbol": sym}
            for s, e, lbl in slots:
                seg = rth.loc[(rth["ts_et"].dt.time >= s) & (rth["ts_et"].dt.time < e)]
                rng = float(seg["high"].max() - seg["low"].min()) if not seg.empty else 0.0
                slot_row[f"rng_{lbl}"] = rng
            intervals_rows.append(slot_row)

    # Final residual day
    if not residual.empty:
        df = residual.copy()
        tz_et = pytz.timezone("US/Eastern")
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz_et)
        df["et_date"] = df["ts_et"].dt.date
        df = df[df["ts_et"].dt.weekday <= 4]
        df = df[df["symbol"].astype(str).str.startswith("NQ") & ~df["symbol"].astype(str).str.contains("-")]
        df = _determine_front_month(df)

        for (sym, d), g in df.groupby(["symbol", "et_date"]):
            g = g.sort_values("ts_et").reset_index(drop=True)

            start_ts = (pd.Timestamp(d) - pd.Timedelta(days=1)).replace(hour=18, minute=0)
            start_ts = tz_et.localize(start_ts)
            end_ts = tz_et.localize(pd.Timestamp.combine(d, RTH_START))
            on_mask = (g["ts_et"] >= start_ts) & (g["ts_et"] < end_ts)
            overnight = g.loc[on_mask]
            if not overnight.empty:
                on_range = float(overnight["high"].max() - overnight["low"].min())
                on_open = float(overnight.iloc[0]["open"])
                on_close = float(overnight.iloc[-1]["close"])
                on_dir = "Up" if on_close > on_open else "Down"
            else:
                on_range = 0.0; on_dir = "None"

            asia_range, _, _ = _session_range(g, ASIA_START, ASIA_END, wrap=True)
            london_range, _, _ = _session_range(g, LONDON_START, LONDON_END, wrap=False)

            prev_date = d - timedelta(days=1)
            prev_rth_mask = (g["ts_et"].dt.date == prev_date) & _within(g["ts_et"], RTH_START, RTH_END)
            prev_rth = g.loc[prev_rth_mask]
            if not prev_rth.empty:
                prev_close = float(prev_rth.iloc[-1]["close"])
            else:
                prev_close = np.nan
            open_bar = g.loc[_within(g["ts_et"], RTH_START, time(9, 31))]
            cur_open = float(open_bar.iloc[0]["open"]) if not open_bar.empty else np.nan
            pre_open_gap = float(cur_open - prev_close) if (pd.notna(cur_open) and pd.notna(prev_close)) else np.nan

            rth_mask = _within(g["ts_et"], RTH_START, RTH_END)
            rth = g.loc[rth_mask]
            rth_range = float(rth["high"].max() - rth["low"].min()) if not rth.empty else 0.0
            is_extreme = int(rth_range >= EXTREME_RTH_RANGE)

            daily_rows.append({
                "date": d,
                "symbol": sym,
                "overnight_range": on_range,
                "overnight_direction": on_dir,
                "asia_range": asia_range,
                "london_range": london_range,
                "pre_open_gap": pre_open_gap,
                "rth_range": rth_range,
                "extreme_rth": is_extreme,
            })

            slot_row = {"date": d, "symbol": sym}
            for s, e, lbl in slots:
                seg = rth.loc[(rth["ts_et"].dt.time >= s) & (rth["ts_et"].dt.time < e)]
                rng = float(seg["high"].max() - seg["low"].min()) if not seg.empty else 0.0
                slot_row[f"rng_{lbl}"] = rng
            intervals_rows.append(slot_row)

    daily = pd.DataFrame(daily_rows)
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])  # normalize dtype
    intervals = pd.DataFrame(intervals_rows)
    if not intervals.empty:
        intervals["date"] = pd.to_datetime(intervals["date"])  # normalize dtype

    # Merge events onto daily
    if not daily.empty:
        ev = events.copy()
        if not ev.empty:
            ev["date"] = pd.to_datetime(ev["date"])  # ensure datetime for merge
            daily = daily.merge(ev, left_on="date", right_on="date", how="left")
            for c in ["event_names", "event_types", "event_sessions", "event_times"]:
                if c in daily.columns:
                    daily[c] = daily[c].fillna("None")

            # Event timing flags
            daily["has_any_event"] = (daily["event_names"] != "None").astype(int)
            daily["has_morning_event"] = daily["event_times"].str.contains("09:3|10:0|08:", na=False).astype(int)

    return daily, intervals


def summarize_extreme_correlations(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()

    out_rows: List[Dict] = []

    def add_bin_stats(feature: str, series: pd.Series, bins: int | List[float] = 4):
        nonlocal out_rows
        s = series.copy()
        mask = s.notna()
        if mask.sum() == 0:
            return
        try:
            if isinstance(bins, list):
                b = pd.cut(s[mask], bins=bins, include_lowest=True, duplicates='drop')
            else:
                b = pd.qcut(s[mask], q=bins, duplicates='drop')
        except Exception:
            # fallback to numeric cut
            b = pd.cut(s[mask], bins=4, include_lowest=True, duplicates='drop')
        grp = daily.loc[mask].groupby(b)
        stats = grp["extreme_rth"].agg(["mean", "count"]).rename(columns={"mean": "extreme_rate"})
        stats["mean_rth_range"] = grp["rth_range"].mean()
        for idx, row in stats.reset_index().iterrows():
            out_rows.append({
                "feature": feature,
                "bin": str(row.iloc[0]),
                "extreme_rate": float(row["extreme_rate"]),
                "count": int(row["count"]),
                "mean_rth_range": float(row["mean_rth_range"]),
            })

    add_bin_stats("overnight_range", daily["overnight_range"], bins=4)
    add_bin_stats("asia_range", daily["asia_range"], bins=4)
    add_bin_stats("london_range", daily["london_range"], bins=4)
    # Gap bins in points
    valid_gap = daily["pre_open_gap"].abs()
    add_bin_stats("abs_pre_open_gap", valid_gap, bins=[-0.01, 5, 10, 20, 40, 80, 160, 1e9])

    # Direction split
    if "overnight_direction" in daily.columns:
        dir_stats = daily.groupby("overnight_direction")["extreme_rth"].agg(["mean", "count"]).rename(columns={"mean": "extreme_rate"})
        dir_stats["mean_rth_range"] = daily.groupby("overnight_direction")["rth_range"].mean()
        for k, row in dir_stats.reset_index().iterrows():
            out_rows.append({
                "feature": "overnight_direction",
                "bin": row["overnight_direction"],
                "extreme_rate": float(row["extreme_rate"]),
                "count": int(row["count"]),
                "mean_rth_range": float(row["mean_rth_range"]),
            })

    # Event presence
    if "has_any_event" in daily.columns:
        ev_stats = daily.groupby("has_any_event")["extreme_rth"].agg(["mean", "count"]).rename(columns={"mean": "extreme_rate"})
        ev_stats["mean_rth_range"] = daily.groupby("has_any_event")["rth_range"].mean()
        for k, row in ev_stats.reset_index().iterrows():
            out_rows.append({
                "feature": "has_any_event",
                "bin": "Yes" if row["has_any_event"] else "No",
                "extreme_rate": float(row["extreme_rate"]),
                "count": int(row["count"]),
                "mean_rth_range": float(row["mean_rth_range"]),
            })

    return pd.DataFrame(out_rows)


def summarize_15m(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    # Identify all rng_* columns
    rng_cols = [c for c in intervals.columns if c.startswith("rng_")]
    if not rng_cols:
        return pd.DataFrame()

    # Which slot is max-per-day?
    per_day = intervals[["date", "symbol"] + rng_cols].copy()
    per_day["max_slot"] = per_day[rng_cols].idxmax(axis=1)

    rows: List[Dict] = []
    for c in rng_cols:
        s = intervals[c].astype(float)
        label = c.replace("rng_", "")
        mask = s.notna()
        if mask.sum() == 0:
            continue
        stats = {
            "slot": label,
            "mean": float(s[mask].mean()),
            "median": float(s[mask].median()),
            "std": float(s[mask].std(ddof=0)),
            "p90": float(s[mask].quantile(0.90)),
            "p95": float(s[mask].quantile(0.95)),
            "p99": float(s[mask].quantile(0.99)),
            "max": float(s[mask].max()),
            "count_days": int(mask.sum()),
            "share_max_of_day_%": 100.0 * float((per_day["max_slot"] == c).mean()),
        }
        rows.append(stats)

    out = pd.DataFrame(rows).sort_values(["mean"], ascending=False)
    return out


def main() -> None:
    print("Computing daily metrics and 15-minute interval ranges...")
    daily, intervals = compute_daily_and_intervals()

    if daily.empty:
        print("No daily data produced. Check input CSV path.")
        return

    # Save raw outputs
    daily.to_csv(str(OUT_DIR / "daily_session_metrics.csv"), index=False)
    intervals.to_csv(str(OUT_DIR / "ny_15m_interval_by_day.csv"), index=False)
    print("✅ Wrote analyses/range_extremes_and_15m_slots/daily_session_metrics.csv and ny_15m_interval_by_day.csv")

    # Extreme correlations
    corr = summarize_extreme_correlations(daily)
    if not corr.empty:
        corr.to_csv(str(OUT_DIR / "extreme_range_correlations.csv"), index=False)
        print("✅ Wrote analyses/range_extremes_and_15m_slots/extreme_range_correlations.csv")
        print("Top drivers by extreme_rate (first 12 rows):")
        print(corr.sort_values(["extreme_rate", "count"], ascending=[False, False]).head(12).to_string(index=False))

    # 15-minute summary across full RTH session
    slot_summary = summarize_15m(intervals)
    if not slot_summary.empty:
        slot_summary.to_csv(str(OUT_DIR / "ny_15m_interval_summary.csv"), index=False)
        print("✅ Wrote analyses/range_extremes_and_15m_slots/ny_15m_interval_summary.csv")
        print("\nTop 10 15m slots by mean range:")
        print(slot_summary[["slot", "mean", "median", "p95", "share_max_of_day_%"]].head(10).to_string(index=False))

    # Quick answer snippets
    # Q1: List extreme days and their overnight ranges
    extreme_days = daily.loc[daily["extreme_rth"] == 1, ["date", "rth_range", "overnight_range", "overnight_direction", "pre_open_gap", "asia_range", "london_range"]]
    if not extreme_days.empty:
        extreme_days.sort_values("rth_range", ascending=False).head(15).to_csv(str(OUT_DIR / "top_extreme_days.csv"), index=False)
        print("\n✅ Wrote analyses/range_extremes_and_15m_slots/top_extreme_days.csv (top 15 by RTH range)")


if __name__ == "__main__":
    main()


