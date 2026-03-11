#!/usr/bin/env python3
"""
NY Open Liquidity Sweep Analysis
--------------------------------
Analyzes Asia H/L, London H/L, and Previous Day NY H/L as liquidity targets
for the first 45 minutes of NY (9:30-10:15 ET).

Tracks:
- Availability: % of days with ≥1 level available (not swept pre-open)
- Distance as measuring variable: sweep probability by distance bucket
- Level clustering: when 2+ levels within 50 pts ("super magnet")
- First 45 min range and red folder days for correlation

Usage:
    python ny_open_liquidity_analysis.py

Outputs:
    availability_summary.csv
    sweep_by_distance.csv
    clustering_magnet.csv
    sweep_by_range_redfolder.csv
    daily_detail.csv
"""

import re
from pathlib import Path
from datetime import time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import pytz

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20260221.ohlcv-1m.csv")
EVENTS_FILE = str(DATA_DIR / "us_high_impact_events_2020_to_2025.csv")

ASIA_START = time(19, 0)   # 7:00 PM ET
ASIA_END = time(2, 0)      # 2:00 AM ET (wraps)
LONDON_START = time(2, 0)
LONDON_END = time(8, 0)
RTH_START = time(9, 30)
RTH_END = time(16, 0)
NY_OPEN = time(9, 30)
FIRST_45_END = time(10, 15)

# Distance buckets (pts) - finer near open
DISTANCE_BUCKETS = [0, 25, 50, 75, 100, 150, 200, 300]
# Availability thresholds
AVAIL_THRESHOLDS = [50, 100, 150, 200]
# Clustering: levels within X pts = "magnet"
CLUSTER_THRESHOLD = 50

NQ_TICK = 0.25
CHUNKSIZE = 1_000_000
WEEKDAYS_ONLY = True
NQ_RE = re.compile(r"^NQ[A-Z]\d{1,2}$")
# ==============================================


def _within(ts, start_t, end_t):
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)


def _within_wrap(ts, start_t, end_t):
    t = ts.dt.time
    return (t >= start_t) | (t < end_t)


def _check_sweep(bars: pd.DataFrame, level: float, is_high: bool) -> bool:
    if bars.empty:
        return False
    if is_high:
        return (bars["high"] > level + NQ_TICK).any()
    return (bars["low"] < level - NQ_TICK).any()


def _front_month(df: pd.DataFrame, date_col: str = "trading_date") -> pd.DataFrame:
    rth = df.loc[_within(df["ts_et"], NY_OPEN, time(9, 45))]
    if rth.empty:
        return df
    vol = rth.groupby([date_col, "symbol"]).agg(volume=("volume", "sum")).reset_index()
    best = vol.loc[vol.groupby(date_col)["volume"].idxmax(), [date_col, "symbol"]].rename(columns={"symbol": "front"})
    out = df.merge(best, on=date_col, how="left")
    return out.loc[out["symbol"] == out["front"]].drop(columns=["front"])


def _load_events() -> pd.DataFrame:
    try:
        e = pd.read_csv(EVENTS_FILE)
    except Exception:
        return pd.DataFrame()
    e["date"] = pd.to_datetime(e["date"], errors="coerce").dt.date
    return e[["date"]].drop_duplicates()


def _get_distance_bucket(distance: float) -> str:
    d = abs(distance)
    for i in range(len(DISTANCE_BUCKETS) - 1):
        if d <= DISTANCE_BUCKETS[i + 1]:
            return f"{DISTANCE_BUCKETS[i]}-{DISTANCE_BUCKETS[i+1]}"
    return f"{DISTANCE_BUCKETS[-1]}+"


def _find_clusters(levels: List[Dict], threshold: float) -> List[Dict]:
    """Find zones where 2+ levels are within threshold pts of each other."""
    if len(levels) < 2:
        return []
    out = []
    used = set()
    for i, a in enumerate(levels):
        if i in used:
            continue
        zone = [a]
        for j, b in enumerate(levels):
            if j <= i or j in used:
                continue
            if abs(a["level"] - b["level"]) <= threshold:
                zone.append(b)
                used.add(j)
        if len(zone) >= 2:
            used.add(i)
            lvls = [x["level"] for x in zone]
            out.append({
                "zone_center": np.mean(lvls),
                "level_types": [x["level_type"] for x in zone],
                "confluence_strength": len(zone),
            })
    return out


def process_day(g: pd.DataFrame, prev_day: Optional[pd.DataFrame], trading_date) -> Tuple[List[Dict], float, float]:
    """
    Process one trading day. Returns (level_rows, range_45m, open_930).
    """
    ny = g.loc[_within(g["ts_et"], NY_OPEN, FIRST_45_END)]
    if ny.empty:
        return [], 0.0, 0.0
    open_930 = ny.iloc[0]["open"]
    range_45m = float(ny["high"].max() - ny["low"].min())

    levels = []

    # 1. Asia H/L (session range 19:00-02:00)
    asia_mask = _within_wrap(g["ts_et"], ASIA_START, ASIA_END)
    asia = g.loc[asia_mask]
    if len(asia) >= 3:
        asia_high = asia["high"].max()
        asia_low = asia["low"].min()
        pre_asia = g.loc[_within(g["ts_et"], ASIA_END, NY_OPEN)]
        if not _check_sweep(pre_asia, asia_high, True):
            levels.append({"level_type": "asia_high", "level": asia_high, "is_high": True})
        if not _check_sweep(pre_asia, asia_low, False):
            levels.append({"level_type": "asia_low", "level": asia_low, "is_high": False})

    # 2. London H/L (session range 02:00-08:00)
    london = g.loc[_within(g["ts_et"], LONDON_START, LONDON_END)]
    if len(london) >= 3:
        london_high = london["high"].max()
        london_low = london["low"].min()
        pre_london = g.loc[_within(g["ts_et"], LONDON_END, NY_OPEN)]
        if not _check_sweep(pre_london, london_high, True):
            levels.append({"level_type": "london_high", "level": london_high, "is_high": True})
        if not _check_sweep(pre_london, london_low, False):
            levels.append({"level_type": "london_low", "level": london_low, "is_high": False})

    # 3. Previous Day NY (RTH) H/L
    if prev_day is not None and not prev_day.empty:
        prev_rth = prev_day.loc[_within(prev_day["ts_et"], RTH_START, RTH_END)]
        if not prev_rth.empty:
            pd_high = prev_rth["high"].max()
            pd_low = prev_rth["low"].min()
            levels.append({"level_type": "prev_day_high", "level": pd_high, "is_high": True})
            levels.append({"level_type": "prev_day_low", "level": pd_low, "is_high": False})

    # Build rows with distance, sweep check
    rows = []
    for lev in levels:
        dist = lev["level"] - open_930
        dist_abs = abs(dist)
        bucket = _get_distance_bucket(dist)

        first_45 = g.loc[_within(g["ts_et"], NY_OPEN, FIRST_45_END)]
        swept = _check_sweep(first_45, lev["level"], lev["is_high"])

        rows.append({
            "level_type": lev["level_type"],
            "level": lev["level"],
            "distance_from_open": round(dist, 2),
            "distance_abs": dist_abs,
            "distance_bucket": bucket,
            "swept_first_45min": int(swept),
        })

    return rows, range_45m, open_930


def main():
    tz = pytz.timezone("US/Eastern")
    events_df = _load_events()
    red_folder_dates = set(events_df["date"].tolist()) if not events_df.empty else set()

    all_detail = []
    daily_range_redfolder = []
    all_dates = set()
    prev_day_cache: Dict = {}
    residual = pd.DataFrame()
    chunk_n = 0

    print("Loading data...")
    for chunk in pd.read_csv(
        CSV_FILE,
        usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
        dtype={"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "int64", "symbol": "string"},
        chunksize=CHUNKSIZE,
    ):
        chunk_n += 1
        df = pd.concat([residual, chunk], ignore_index=True)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz)
        df = df[df["symbol"].astype(str).str.match(NQ_RE)]

        cal = df["ts_et"].dt.normalize()
        evening = df["ts_et"].dt.time >= time(18, 0)
        td = cal.copy()
        td.loc[evening] = cal.loc[evening] + pd.Timedelta(days=1)
        df["trading_date"] = td.dt.date

        if WEEKDAYS_ONLY:
            df = df[df["trading_date"].apply(lambda d: d.weekday() <= 4)]
        if df.empty:
            residual = pd.DataFrame()
            continue

        max_td = df["trading_date"].max()
        to_proc = df[df["trading_date"] < max_td]
        residual = df[df["trading_date"] == max_td]

        if to_proc.empty:
            continue

        to_proc = _front_month(to_proc)
        dates = sorted(to_proc["trading_date"].unique())

        for d in dates:
            all_dates.add(d)
            g = to_proc[to_proc["trading_date"] == d].sort_values("ts_et")
            prev_date = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()
            prev_day = prev_day_cache.get(prev_date)

            rows, range_45m, open_930 = process_day(g, prev_day, d)
            is_red = d in red_folder_dates

            for r in rows:
                r["date"] = d
                r["ny_open_price"] = open_930
                r["range_45m"] = range_45m
                r["is_red_folder_day"] = int(is_red)
                all_detail.append(r)

            if rows:
                n_avail = len(rows)
                n_200 = sum(1 for r in rows if r["distance_abs"] <= 200)
                any_swept = int(any(r["swept_first_45min"] for r in rows))
                daily_range_redfolder.append({
                    "date": d,
                    "range_45m": range_45m,
                    "is_red_folder_day": int(is_red),
                    "n_levels_available": n_avail,
                    "n_within_200": n_200,
                    "any_swept_45min": any_swept,
                })

            prev_day_cache[d] = g.copy()

        print(f"  chunk {chunk_n}: +{len([x for x in all_detail if x['date'] in dates])} rows")

    # Final residual
    if not residual.empty:
        residual = _front_month(residual)
        for d in residual["trading_date"].unique():
            all_dates.add(d)
            g = residual[residual["trading_date"] == d].sort_values("ts_et")
            prev_date = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()
            prev_day = prev_day_cache.get(prev_date)

            rows, range_45m, open_930 = process_day(g, prev_day, d)
            is_red = d in red_folder_dates

            for r in rows:
                r["date"] = d
                r["ny_open_price"] = open_930
                r["range_45m"] = range_45m
                r["is_red_folder_day"] = int(is_red)
                all_detail.append(r)

            if rows:
                n_avail = len(rows)
                n_200 = sum(1 for r in rows if r["distance_abs"] <= 200)
                any_swept = int(any(r["swept_first_45min"] for r in rows))
                daily_range_redfolder.append({
                    "date": d,
                    "range_45m": range_45m,
                    "is_red_folder_day": int(is_red),
                    "n_levels_available": n_avail,
                    "n_within_200": n_200,
                    "any_swept_45min": any_swept,
                })

    if not all_detail:
        print("No qualifying levels found.")
        return

    detail = pd.DataFrame(all_detail)

    # ---- daily_detail.csv ----
    detail_cols = ["date", "level_type", "level", "ny_open_price", "distance_from_open", "distance_bucket",
                   "range_45m", "is_red_folder_day", "swept_first_45min"]
    detail[detail_cols].to_csv(OUT_DIR / "daily_detail.csv", index=False)
    print(f"✅ Wrote daily_detail.csv ({len(detail)} rows)")

    # ---- availability_summary.csv ----
    total_days = len(all_dates) if all_dates else detail["date"].nunique()
    avail_rows = []
    for thresh in AVAIL_THRESHOLDS:
        within = detail[detail["distance_abs"] <= thresh].groupby("date").size()
        n_days = len(within)
        pct = round(100 * n_days / total_days, 1) if total_days else 0
        avail_rows.append({"threshold_pts": thresh, "days_with_level_within": n_days, "pct_of_days": pct})
    pd.DataFrame(avail_rows).to_csv(OUT_DIR / "availability_summary.csv", index=False)
    print(f"✅ Wrote availability_summary.csv")

    # ---- sweep_by_distance.csv ----
    sweep_dist = []
    for (level_type, bucket), grp in detail.groupby(["level_type", "distance_bucket"]):
        n = len(grp)
        swept = grp["swept_first_45min"].sum()
        sweep_dist.append({
            "level_type": level_type,
            "distance_bucket": bucket,
            "total_levels": n,
            "swept_45min": int(swept),
            "sweep_rate_pct": round(100 * swept / n, 1) if n else 0,
        })
    pd.DataFrame(sweep_dist).to_csv(OUT_DIR / "sweep_by_distance.csv", index=False)
    print(f"✅ Wrote sweep_by_distance.csv")

    # ---- clustering_magnet.csv ----
    cluster_rows = []
    for date, grp in detail.groupby("date"):
        levels = [{"level_type": r["level_type"], "level": r["level"], "is_high": "high" in r["level_type"]}
                 for _, r in grp.iterrows()]
        clusters = _find_clusters(levels, CLUSTER_THRESHOLD)
        open_price = grp["ny_open_price"].iloc[0]
        for c in clusters:
            dist = abs(c["zone_center"] - open_price)
            if dist > 200:
                continue
            cluster_levels = set(c["level_types"])
            sub_cluster = grp[grp["level_type"].isin(cluster_levels)]
            swept = sub_cluster["swept_first_45min"].max() if not sub_cluster.empty else 0
            cluster_rows.append({
                "date": date,
                "confluence_strength": c["confluence_strength"],
                "level_types": "|".join(c["level_types"]),
                "zone_center": c["zone_center"],
                "distance_from_open": round(dist, 2),
                "swept_45min": int(swept),
            })
    if cluster_rows:
        cluster_df = pd.DataFrame(cluster_rows)
        cluster_summary = cluster_df.groupby("confluence_strength").agg(
            total_zones=("date", "count"),
            swept_45min=("swept_45min", "sum"),
        ).reset_index()
        cluster_summary["sweep_rate_pct"] = (100 * cluster_summary["swept_45min"] / cluster_summary["total_zones"]).round(1)
        cluster_summary.to_csv(OUT_DIR / "clustering_magnet.csv", index=False)
        print(f"✅ Wrote clustering_magnet.csv ({len(cluster_rows)} zones)")
    else:
        print("⚠ No clustering zones found")

    # ---- level_clustering_2025.csv: 2025 days with which levels cluster (within 50 pts) and available ----
    detail["date"] = pd.to_datetime(detail["date"])
    detail_2025 = detail[detail["date"].dt.year == 2025]
    clustering_2025 = []
    for date, grp in detail_2025.groupby("date"):
        levels = [{"level_type": r["level_type"], "level": r["level"]} for _, r in grp.iterrows()]
        clusters = _find_clusters(levels, CLUSTER_THRESHOLD)
        open_price = grp["ny_open_price"].iloc[0]
        for c in clusters:
            cluster_levels = set(c["level_types"])
            sub = grp[grp["level_type"].isin(cluster_levels)]
            lvls = sub["level"].tolist()
            pts_between = max(lvls) - min(lvls) if len(lvls) >= 2 else 0
            dist_open = abs(c["zone_center"] - open_price)
            swept = sub["swept_first_45min"].max() if not sub.empty else 0
            clustering_2025.append({
                "date": date.strftime("%Y-%m-%d"),
                "level_types": " | ".join(sorted(c["level_types"])),
                "level_prices": " | ".join(f"{x:.2f}" for x in sorted(lvls)),
                "pts_between_levels": round(pts_between, 2),
                "confluence_strength": c["confluence_strength"],
                "distance_from_open_pts": round(dist_open, 2),
                "swept_first_45min": int(swept),
                "ny_open": round(open_price, 2),
            })
    if clustering_2025:
        pd.DataFrame(clustering_2025).to_csv(OUT_DIR / "level_clustering_2025.csv", index=False)
        print(f"✅ Wrote level_clustering_2025.csv ({len(clustering_2025)} clusters)")
    else:
        print("⚠ No 2025 clustering data")

    # ---- sweep_by_range_redfolder.csv ----
    if daily_range_redfolder:
        drf = pd.DataFrame(daily_range_redfolder)
        # Range quartiles
        try:
            drf["range_quartile"] = pd.qcut(drf["range_45m"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except Exception:
            drf["range_quartile"] = "all"
        corr_rows = []

        for red in [0, 1]:
            sub = drf[drf["is_red_folder_day"] == red]
            if sub.empty:
                continue
            label = "red_folder" if red else "neutral"
            n = len(sub)
            any_swept = sub["any_swept_45min"].sum()
            corr_rows.append({
                "day_type": label,
                "n_days": n,
                "days_with_any_sweep": int(any_swept),
                "sweep_rate_pct": round(100 * any_swept / n, 1) if n else 0,
                "avg_range_45m": round(sub["range_45m"].mean(), 2),
            })

        for q in drf["range_quartile"].dropna().unique():
            sub = drf[drf["range_quartile"] == q]
            if sub.empty:
                continue
            n = len(sub)
            any_swept = sub["any_swept_45min"].sum()
            corr_rows.append({
                "day_type": f"range_{q}",
                "n_days": n,
                "days_with_any_sweep": int(any_swept),
                "sweep_rate_pct": round(100 * any_swept / n, 1) if n else 0,
                "avg_range_45m": round(sub["range_45m"].mean(), 2),
            })

        pd.DataFrame(corr_rows).to_csv(OUT_DIR / "sweep_by_range_redfolder.csv", index=False)
        print(f"✅ Wrote sweep_by_range_redfolder.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
