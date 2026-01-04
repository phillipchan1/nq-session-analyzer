#!/usr/bin/env python3
"""
VIX Correlation and NY Session Range Prediction Analysis
---------------------------------------------------------

Analyzes how VIX (VIXY ETF) correlates with NY session range for both:
- First 45 minutes (9:30-10:15 AM ET)
- Full session (9:30 AM-4:00 PM ET)

Builds predictive models using VIX metrics and overnight range to predict upcoming session ranges.

Outputs:
- vix_daily_metrics.csv: Per-day metrics with all VIX values and ranges
- vix_correlations.csv: Correlation analysis between VIX metrics and ranges
- vix_bin_analysis.csv: Binning analysis grouped by VIX levels and overnight range
- vix_regression_results.csv: Regression model results
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import time, timedelta
import pytz
from scipy import stats
from tqdm import tqdm

# =================== CONFIG ===================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
NQ_CSV_FILE = str(DATA_DIR / "glbx-mdp3-20200927-20250926.ohlcv-1m.csv")
VIX_CSV_FILE = str(DATA_DIR / "xnas-itch-20201203-20251203.ohlcv-1m.csv")

# Session time boundaries (ET)
RTH_START, RTH_END = time(9, 30), time(16, 0)
FIRST_45_END = time(10, 15)
OVERNIGHT_START_HOUR = 18  # 6 PM previous day

# Chunking
CHUNKSIZE = 1_000_000


def _within(ts: pd.Series, start_t: time, end_t: time) -> pd.Series:
    """Check if timestamps fall within time range."""
    t = ts.dt.time
    return (t >= start_t) & (t < end_t)


def _determine_front_month(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to front-month per date by max range in 9:30–10:15 ET window (liquidity proxy)."""
    mask = _within(df["ts_et"], RTH_START, FIRST_45_END)
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


def load_and_process_nq_data() -> pd.DataFrame:
    """Load NQ data and calculate session ranges."""
    print("Loading NQ data...")
    tz_et = pytz.timezone("US/Eastern")
    residual = pd.DataFrame()
    daily_rows: List[Dict] = []

    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    dtypes = {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "symbol": "string"
    }

    chunk_count = 0
    for chunk in pd.read_csv(NQ_CSV_FILE, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE):
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"  Processed {chunk_count} chunks...")
        
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
            start_ts = (pd.Timestamp(d) - pd.Timedelta(days=1)).replace(hour=OVERNIGHT_START_HOUR, minute=0)
            start_ts = tz_et.localize(start_ts)
            end_ts = tz_et.localize(pd.Timestamp.combine(d, RTH_START))
            on_mask = (g["ts_et"] >= start_ts) & (g["ts_et"] < end_ts)
            overnight = g.loc[on_mask]
            
            if not overnight.empty:
                on_range = float(overnight["high"].max() - overnight["low"].min())
            else:
                on_range = 0.0

            # First 45 minutes range (9:30-10:15)
            first_45_mask = _within(g["ts_et"], RTH_START, FIRST_45_END)
            first_45 = g.loc[first_45_mask]
            if not first_45.empty:
                first_45_range = float(first_45["high"].max() - first_45["low"].min())
            else:
                first_45_range = 0.0

            # Full session range (9:30-16:00)
            rth_mask = _within(g["ts_et"], RTH_START, RTH_END)
            rth = g.loc[rth_mask]
            if not rth.empty:
                full_session_range = float(rth["high"].max() - rth["low"].min())
            else:
                full_session_range = 0.0

            daily_rows.append({
                "date": d,
                "symbol": sym,
                "overnight_range": on_range,
                "first_45m_range": first_45_range,
                "full_session_range": full_session_range,
            })

    # Process final residual day
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

            start_ts = (pd.Timestamp(d) - pd.Timedelta(days=1)).replace(hour=OVERNIGHT_START_HOUR, minute=0)
            start_ts = tz_et.localize(start_ts)
            end_ts = tz_et.localize(pd.Timestamp.combine(d, RTH_START))
            on_mask = (g["ts_et"] >= start_ts) & (g["ts_et"] < end_ts)
            overnight = g.loc[on_mask]
            
            if not overnight.empty:
                on_range = float(overnight["high"].max() - overnight["low"].min())
            else:
                on_range = 0.0

            first_45_mask = _within(g["ts_et"], RTH_START, FIRST_45_END)
            first_45 = g.loc[first_45_mask]
            if not first_45.empty:
                first_45_range = float(first_45["high"].max() - first_45["low"].min())
            else:
                first_45_range = 0.0

            rth_mask = _within(g["ts_et"], RTH_START, RTH_END)
            rth = g.loc[rth_mask]
            if not rth.empty:
                full_session_range = float(rth["high"].max() - rth["low"].min())
            else:
                full_session_range = 0.0

            daily_rows.append({
                "date": d,
                "symbol": sym,
                "overnight_range": on_range,
                "first_45m_range": first_45_range,
                "full_session_range": full_session_range,
            })

    nq_df = pd.DataFrame(daily_rows)
    if not nq_df.empty:
        nq_df["date"] = pd.to_datetime(nq_df["date"])
    print(f"  Processed {len(nq_df)} trading days")
    return nq_df


def load_and_process_vix_data() -> pd.DataFrame:
    """Load VIXY data and calculate VIX metrics available before session starts."""
    print("Loading VIXY data...")
    tz_et = pytz.timezone("US/Eastern")
    residual = pd.DataFrame()
    daily_rows: List[Dict] = []

    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    dtypes = {
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "symbol": "string"
    }

    chunk_count = 0
    for chunk in pd.read_csv(VIX_CSV_FILE, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE):
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"  Processed {chunk_count} chunks...")
        
        df = pd.concat([residual, chunk], ignore_index=True)
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz_et)
        df["et_date"] = df["ts_et"].dt.date

        # Weekdays only
        df = df[df["ts_et"].dt.weekday <= 4]
        if df.empty:
            continue

        # Filter to VIXY symbol
        df = df[df["symbol"].astype(str) == "VIXY"]

        # Split off the final day to complete next chunk
        max_date = df["et_date"].max()
        to_proc = df[df["et_date"] < max_date]
        residual = df[df["et_date"] == max_date]

        for d, g in to_proc.groupby("et_date"):
            g = g.sort_values("ts_et").reset_index(drop=True)

            # Previous day close (last bar before 16:00 on previous day)
            prev_date = d - timedelta(days=1)
            prev_rth_mask = (g["ts_et"].dt.date == prev_date) & _within(g["ts_et"], RTH_START, RTH_END)
            prev_rth = g.loc[prev_rth_mask]
            vix_prev_day_close = float(prev_rth.iloc[-1]["close"]) if not prev_rth.empty else np.nan

            # Overnight session: 18:00 prev day to 9:30 current day
            start_ts = (pd.Timestamp(d) - pd.Timedelta(days=1)).replace(hour=OVERNIGHT_START_HOUR, minute=0)
            start_ts = tz_et.localize(start_ts)
            end_ts = tz_et.localize(pd.Timestamp.combine(d, RTH_START))
            on_mask = (g["ts_et"] >= start_ts) & (g["ts_et"] < end_ts)
            overnight = g.loc[on_mask]

            if not overnight.empty:
                vix_overnight_open = float(overnight.iloc[0]["open"])
                vix_overnight_close = float(overnight.iloc[-1]["close"])
                vix_overnight_high = float(overnight["high"].max())
                vix_overnight_low = float(overnight["low"].min())
                vix_overnight_range = vix_overnight_high - vix_overnight_low
                vix_overnight_change = vix_overnight_close - vix_overnight_open
            else:
                vix_overnight_open = np.nan
                vix_overnight_close = np.nan
                vix_overnight_range = np.nan
                vix_overnight_change = np.nan

            # VIX at 9:30 AM (or closest bar)
            open_930_mask = _within(g["ts_et"], RTH_START, time(9, 31))
            open_930 = g.loc[open_930_mask]
            if not open_930.empty:
                vix_930_open = float(open_930.iloc[0]["open"])
            else:
                # Try to get closest bar before 9:30
                before_930 = g[g["ts_et"] < tz_et.localize(pd.Timestamp.combine(d, RTH_START))]
                if not before_930.empty:
                    vix_930_open = float(before_930.iloc[-1]["close"])
                else:
                    vix_930_open = np.nan

            daily_rows.append({
                "date": d,
                "vix_prev_day_close": vix_prev_day_close,
                "vix_overnight_open": vix_overnight_open,
                "vix_overnight_close": vix_overnight_close,
                "vix_overnight_range": vix_overnight_range,
                "vix_overnight_change": vix_overnight_change,
                "vix_930_open": vix_930_open,
            })

    # Process final residual day
    if not residual.empty:
        df = residual.copy()
        tz_et = pytz.timezone("US/Eastern")
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df.dropna(subset=["ts_event"], inplace=True)
        df["ts_et"] = df["ts_event"].dt.tz_convert(tz_et)
        df["et_date"] = df["ts_et"].dt.date
        df = df[df["ts_et"].dt.weekday <= 4]
        df = df[df["symbol"].astype(str) == "VIXY"]

        for d, g in df.groupby("et_date"):
            g = g.sort_values("ts_et").reset_index(drop=True)

            prev_date = d - timedelta(days=1)
            prev_rth_mask = (g["ts_et"].dt.date == prev_date) & _within(g["ts_et"], RTH_START, RTH_END)
            prev_rth = g.loc[prev_rth_mask]
            vix_prev_day_close = float(prev_rth.iloc[-1]["close"]) if not prev_rth.empty else np.nan

            start_ts = (pd.Timestamp(d) - pd.Timedelta(days=1)).replace(hour=OVERNIGHT_START_HOUR, minute=0)
            start_ts = tz_et.localize(start_ts)
            end_ts = tz_et.localize(pd.Timestamp.combine(d, RTH_START))
            on_mask = (g["ts_et"] >= start_ts) & (g["ts_et"] < end_ts)
            overnight = g.loc[on_mask]

            if not overnight.empty:
                vix_overnight_open = float(overnight.iloc[0]["open"])
                vix_overnight_close = float(overnight.iloc[-1]["close"])
                vix_overnight_high = float(overnight["high"].max())
                vix_overnight_low = float(overnight["low"].min())
                vix_overnight_range = vix_overnight_high - vix_overnight_low
                vix_overnight_change = vix_overnight_close - vix_overnight_open
            else:
                vix_overnight_open = np.nan
                vix_overnight_close = np.nan
                vix_overnight_range = np.nan
                vix_overnight_change = np.nan

            open_930_mask = _within(g["ts_et"], RTH_START, time(9, 31))
            open_930 = g.loc[open_930_mask]
            if not open_930.empty:
                vix_930_open = float(open_930.iloc[0]["open"])
            else:
                before_930 = g[g["ts_et"] < tz_et.localize(pd.Timestamp.combine(d, RTH_START))]
                if not before_930.empty:
                    vix_930_open = float(before_930.iloc[-1]["close"])
                else:
                    vix_930_open = np.nan

            daily_rows.append({
                "date": d,
                "vix_prev_day_close": vix_prev_day_close,
                "vix_overnight_open": vix_overnight_open,
                "vix_overnight_close": vix_overnight_close,
                "vix_overnight_range": vix_overnight_range,
                "vix_overnight_change": vix_overnight_change,
                "vix_930_open": vix_930_open,
            })

    vix_df = pd.DataFrame(daily_rows)
    if not vix_df.empty:
        vix_df["date"] = pd.to_datetime(vix_df["date"])
    print(f"  Processed {len(vix_df)} trading days")
    return vix_df


def merge_nq_and_vix(nq_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Merge NQ ranges with VIX metrics by date."""
    print("Merging NQ and VIX data...")
    merged = pd.merge(nq_df, vix_df, on="date", how="inner")
    print(f"  Merged data: {len(merged)} days with both NQ and VIX data")
    return merged


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlations."""
    print("Calculating correlations...")
    
    vix_features = [
        "vix_930_open",
        "vix_overnight_close",
        "vix_prev_day_close",
        "vix_overnight_range",
        "vix_overnight_change",
    ]
    
    range_targets = ["first_45m_range", "full_session_range"]
    
    correlation_rows = []
    
    for target in range_targets:
        for feature in vix_features + ["overnight_range"]:
            # Remove NaN values for correlation
            valid_mask = df[[target, feature]].notna().all(axis=1)
            if valid_mask.sum() < 10:  # Need at least 10 data points
                continue
            
            x = df.loc[valid_mask, feature]
            y = df.loc[valid_mask, target]
            
            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)
            
            correlation_rows.append({
                "target": target,
                "feature": feature,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n_samples": valid_mask.sum(),
            })
    
    # Also calculate correlations for combined features
    df_clean = df.dropna(subset=["vix_930_open", "overnight_range", "first_45m_range", "full_session_range"])
    if len(df_clean) >= 10:
        # VIX * overnight_range interaction
        df_clean["vix_x_overnight"] = df_clean["vix_930_open"] * df_clean["overnight_range"]
        
        for target in range_targets:
            pearson_r, pearson_p = stats.pearsonr(df_clean["vix_x_overnight"], df_clean[target])
            spearman_r, spearman_p = stats.spearmanr(df_clean["vix_x_overnight"], df_clean[target])
            
            correlation_rows.append({
                "target": target,
                "feature": "vix_930_open_x_overnight_range",
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n_samples": len(df_clean),
            })
    
    corr_df = pd.DataFrame(correlation_rows)
    print(f"  Calculated {len(corr_df)} correlations")
    return corr_df


def binning_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Create binning analysis grouped by VIX levels and overnight range bins."""
    print("Performing binning analysis...")
    
    # Create bins
    df_clean = df.dropna(subset=["vix_930_open", "overnight_range", "first_45m_range", "full_session_range"])
    
    # VIX bins
    vix_bins = [0, 15, 20, 25, 30, 50, 1000]
    df_clean["vix_bin"] = pd.cut(df_clean["vix_930_open"], bins=vix_bins, labels=["<15", "15-20", "20-25", "25-30", "30-50", "50+"])
    
    # Overnight range bins (in points)
    on_bins = [0, 50, 100, 150, 200, 1000]
    df_clean["overnight_range_bin"] = pd.cut(df_clean["overnight_range"], bins=on_bins, labels=["<50", "50-100", "100-150", "150-200", "200+"])
    
    bin_rows = []
    
    # By VIX bins
    for vix_bin in df_clean["vix_bin"].cat.categories:
        subset = df_clean[df_clean["vix_bin"] == vix_bin]
        if len(subset) == 0:
            continue
        
        bin_rows.append({
            "bin_type": "vix_level",
            "bin_label": str(vix_bin),
            "n_samples": len(subset),
            "first_45m_range_mean": subset["first_45m_range"].mean(),
            "first_45m_range_median": subset["first_45m_range"].median(),
            "first_45m_range_std": subset["first_45m_range"].std(),
            "full_session_range_mean": subset["full_session_range"].mean(),
            "full_session_range_median": subset["full_session_range"].median(),
            "full_session_range_std": subset["full_session_range"].std(),
        })
    
    # By overnight range bins
    for on_bin in df_clean["overnight_range_bin"].cat.categories:
        subset = df_clean[df_clean["overnight_range_bin"] == on_bin]
        if len(subset) == 0:
            continue
        
        bin_rows.append({
            "bin_type": "overnight_range",
            "bin_label": str(on_bin),
            "n_samples": len(subset),
            "first_45m_range_mean": subset["first_45m_range"].mean(),
            "first_45m_range_median": subset["first_45m_range"].median(),
            "first_45m_range_std": subset["first_45m_range"].std(),
            "full_session_range_mean": subset["full_session_range"].mean(),
            "full_session_range_median": subset["full_session_range"].median(),
            "full_session_range_std": subset["full_session_range"].std(),
        })
    
    bin_df = pd.DataFrame(bin_rows)
    print(f"  Created {len(bin_df)} bin groups")
    return bin_df


def regression_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Build regression models to predict session ranges using multiple linear regression."""
    print("Performing regression analysis...")
    
    df_clean = df.dropna(subset=["vix_930_open", "overnight_range", "first_45m_range", "full_session_range"])
    
    if len(df_clean) < 20:
        print("  Warning: Not enough data for regression analysis")
        return pd.DataFrame()
    
    regression_rows = []
    
    # Helper function to calculate R² and RMSE manually
    def calculate_r2_rmse(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return r2, rmse
    
    # Model 1: first_45m_range ~ vix_930_open + overnight_range
    # Using numpy for multiple linear regression
    X1 = np.column_stack([np.ones(len(df_clean)), df_clean["vix_930_open"].values, df_clean["overnight_range"].values])
    y1 = df_clean["first_45m_range"].values
    
    coeffs1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    y1_pred = X1 @ coeffs1
    r2_1, rmse_1 = calculate_r2_rmse(y1, y1_pred)
    
    regression_rows.append({
        "model": "first_45m_range ~ vix_930_open + overnight_range",
        "target": "first_45m_range",
        "features": "vix_930_open, overnight_range",
        "r2": r2_1,
        "rmse": rmse_1,
        "n_samples": len(df_clean),
        "intercept": coeffs1[0],
        "coef_vix_930_open": coeffs1[1],
        "coef_overnight_range": coeffs1[2],
    })
    
    # Model 2: full_session_range ~ vix_930_open + overnight_range
    X2 = np.column_stack([np.ones(len(df_clean)), df_clean["vix_930_open"].values, df_clean["overnight_range"].values])
    y2 = df_clean["full_session_range"].values
    
    coeffs2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
    y2_pred = X2 @ coeffs2
    r2_2, rmse_2 = calculate_r2_rmse(y2, y2_pred)
    
    regression_rows.append({
        "model": "full_session_range ~ vix_930_open + overnight_range",
        "target": "full_session_range",
        "features": "vix_930_open, overnight_range",
        "r2": r2_2,
        "rmse": rmse_2,
        "n_samples": len(df_clean),
        "intercept": coeffs2[0],
        "coef_vix_930_open": coeffs2[1],
        "coef_overnight_range": coeffs2[2],
    })
    
    # Model 3: Include all VIX metrics (if available)
    vix_features_all = [
        "vix_930_open",
        "vix_overnight_close",
        "vix_prev_day_close",
        "vix_overnight_range",
        "vix_overnight_change",
        "overnight_range",
    ]
    
    df_model3 = df_clean.dropna(subset=vix_features_all + ["first_45m_range", "full_session_range"])
    
    if len(df_model3) >= 20:
        for target in ["first_45m_range", "full_session_range"]:
            X3 = np.column_stack([np.ones(len(df_model3))] + [df_model3[feat].values for feat in vix_features_all])
            y3 = df_model3[target].values
            
            coeffs3 = np.linalg.lstsq(X3, y3, rcond=None)[0]
            y3_pred = X3 @ coeffs3
            r2_3, rmse_3 = calculate_r2_rmse(y3, y3_pred)
            
            coef_dict = {f"coef_{feat}": coef for feat, coef in zip(vix_features_all, coeffs3[1:])}
            
            regression_rows.append({
                "model": f"{target} ~ all_vix_features",
                "target": target,
                "features": ", ".join(vix_features_all),
                "r2": r2_3,
                "rmse": rmse_3,
                "n_samples": len(df_model3),
                "intercept": coeffs3[0],
                **coef_dict,
            })
    
    reg_df = pd.DataFrame(regression_rows)
    print(f"  Built {len(reg_df)} regression models")
    return reg_df


def main():
    """Main analysis function."""
    print("=" * 60)
    print("VIX Correlation and NY Session Range Prediction Analysis")
    print("=" * 60)
    print()
    
    # Load and process data
    nq_df = load_and_process_nq_data()
    vix_df = load_and_process_vix_data()
    
    # Merge data
    merged_df = merge_nq_and_vix(nq_df, vix_df)
    
    if merged_df.empty:
        print("ERROR: No overlapping data between NQ and VIX!")
        return
    
    # Save daily metrics
    print("Saving daily metrics...")
    merged_df.to_csv(OUT_DIR / "vix_daily_metrics.csv", index=False)
    print(f"  Saved to {OUT_DIR / 'vix_daily_metrics.csv'}")
    
    # Correlation analysis
    corr_df = calculate_correlations(merged_df)
    corr_df.to_csv(OUT_DIR / "vix_correlations.csv", index=False)
    print(f"  Saved to {OUT_DIR / 'vix_correlations.csv'}")
    
    # Binning analysis
    bin_df = binning_analysis(merged_df)
    bin_df.to_csv(OUT_DIR / "vix_bin_analysis.csv", index=False)
    print(f"  Saved to {OUT_DIR / 'vix_bin_analysis.csv'}")
    
    # Regression analysis
    reg_df = regression_analysis(merged_df)
    if not reg_df.empty:
        reg_df.to_csv(OUT_DIR / "vix_regression_results.csv", index=False)
        print(f"  Saved to {OUT_DIR / 'vix_regression_results.csv'}")
    
    print()
    print("Analysis complete!")
    print(f"Total days analyzed: {len(merged_df)}")


if __name__ == "__main__":
    main()

