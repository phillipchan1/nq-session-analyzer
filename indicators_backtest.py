import numpy as np
import pandas as pd
from tqdm import tqdm

# ========= USER SETTINGS =========
DATA_PATH      = "glbx-mdp3-20200927-20250926.ohlcv-1m.csv"  # Databento 1m OHLCV CSV
SYMBOL_PREFIX  = "NQ"                  # filter outright NQ (no spreads)
TZ             = "America/New_York"

# Labeling / test sweep
HORIZON_MIN    = [20, 30]              # minutes to scan forward (try both)
TP_POINTS      = [10.0, 15.0, 20.0, 30.0]  # targets in index points
SL_MULTIPLIERS = [0.5, 0.75, 1.0]      # stop = SL_MULT * TP
N_BINS         = 10                    # deciles

# Indicator parameter grids (compact, extensible)
EMA_WINDOWS    = [9, 20, 50]
RSI_WINDOWS    = [6, 14, 21]
ATR_WINDOWS    = [7, 14, 21]
BB_WINDOWS     = [(20, 2.0), (20, 3.0)]
STOCH_KD       = [(14, 3, 3)]          # (k, d, smooth)

# =================================

def load_data(path):
    df = pd.read_csv(path)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df[df["symbol"].str.startswith(SYMBOL_PREFIX) & ~df["symbol"].str.contains("-")]
    df = df.sort_values("ts_event").reset_index(drop=True)
    df["ts_ny"] = df["ts_event"].dt.tz_convert(TZ)
    df["date"]  = df["ts_ny"].dt.date
    df["time"]  = df["ts_ny"].dt.time
    return df

# -------- indicators (from 1m OHLCV only) --------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    up_ema = up.ewm(alpha=1/n, adjust=False).mean()
    dn_ema = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = up_ema / dn_ema.replace(0, np.nan)
    out = 100 - (100/(1+rs))
    return out.fillna(50)

def atr(df, n=14):
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def bb_pctb(close, n=20, k=2.0):
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std()
    up = ma + k*sd; lo = ma - k*sd
    return (close - lo) / (up - lo)

def stoch_kd(df, k=14, d=3, smooth=3):
    low_k  = df["low"].rolling(k, min_periods=k).min()
    high_k = df["high"].rolling(k, min_periods=k).max()
    k_fast = 100 * (df["close"] - low_k) / (high_k - low_k)
    k_fast = k_fast.rolling(smooth, min_periods=smooth).mean()
    d_slow = k_fast.rolling(d, min_periods=d).mean()
    return k_fast, d_slow

def body_share(df):
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    bd  = (df["close"] - df["open"]).abs()
    return (bd / rng).fillna(0)

def opening5_features(df):
    t = df["time"]
    open5 = (t >= pd.to_datetime("09:30").time()) & (t < pd.to_datetime("09:35").time())
    hi5 = df.where(open5).groupby("date")["high"].transform("max")
    lo5 = df.where(open5).groupby("date")["low"].transform("min")
    df["open5_range"] = (hi5 - lo5)
    return df

def build_feature_frame(df):
    feats = pd.DataFrame(index=df.index)

    for n in EMA_WINDOWS:
        e = ema(df["close"], n)
        feats[f"ema_{n}"] = e
        feats[f"dist_ema_{n}"] = df["close"] - e

    for n in RSI_WINDOWS:
        feats[f"rsi_{n}"] = rsi(df["close"], n)

    for n in ATR_WINDOWS:
        feats[f"atr_{n}"] = atr(df, n)
        feats[f"range_norm_{n}"] = (df["high"] - df["low"]) / (feats[f"atr_{n}"] + 1e-9)

    for (n, k) in BB_WINDOWS:
        feats[f"bb_pctb_{n}_{k}"] = bb_pctb(df["close"], n, k)

    for (k, d, sm) in STP:
        pass  # kept for future extension

    kf, ds = stoch_kd(df, 14, 3, 3)
    feats["stochK_14_3"] = kf
    feats["stochD_14_3"] = ds

    feats["body_share"] = body_share(df)
    feats["rng_3"] = (df["high"].rolling(3).max() - df["low"].rolling(3).min())
    feats["rng_5"] = (df["high"].rolling(5).max() - df["low"].rolling(5).min())

    # Opening-5 range (session context)
    df2 = opening5_features(df.copy())
    feats["open5_range"] = df2["open5_range"]

    # Drop rows with all-NaN features
    return feats

# ----- path-aware label & excursions -----
def first_passage_outcome(df, i, horizon, tp_pts, sl_pts):
    """
    Simulate from bar i forward up to horizon minutes:
      Long side: +tp before -sl?   Short side: -tp before +sl?
    Also compute MAE/MFE (worst adverse / best favorable excursion vs entry) over the path.
    Returns dict with long_win, short_win (0/1), mae, mfe, end_move.
    """
    n = len(df)
    if i >= n-1: 
        return dict(long=0, short=0, mae=np.nan, mfe=np.nan, end_move=np.nan)

    jmax = min(n-1, i + horizon)
    entry = df.at[i, "close"]
    tpL, slL = entry + tp_pts, entry - sl_pts
    tpS, slS = entry - tp_pts, entry + sl_pts

    # track MAE/MFE relative to entry (in points), irrespective of side
    mae = 0.0
    mfe = 0.0

    long_state = 0  # 0=alive, +1=TP, -1=SL
    short_state = 0

    prev_max = entry
    prev_min = entry

    for j in range(i+1, jmax+1):
        h = df.at[j, "high"]; l = df.at[j, "low"]
        # excursions
        prev_max = max(prev_max, h)
        prev_min = min(prev_min, l)
        mfe = max(mfe, prev_max - entry)
        mae = min(mae, prev_min - entry)  # negative number

        # long path
        if long_state == 0:
            if l <= slL: long_state = -1
            elif h >= tpL: long_state = +1

        # short path
        if short_state == 0:
            if h >= slS: short_state = -1
            elif l <= tpS: short_state = +1

        if long_state != 0 and short_state != 0:
            break

    end_move = df.at[jmax, "close"] - entry
    return dict(
        long = 1 if long_state==+1 else 0,
        short= 1 if short_state==+1 else 0,
        mae  = mae,   # negative or 0
        mfe  = mfe,   # positive or 0
        end_move = end_move
    )

def compute_labels_and_excursions(df, horizon, tp_pts, sl_pts):
    out = {"long":[], "short":[], "mae":[], "mfe":[], "end_move":[]}
    for i in range(len(df)):
        r = first_passage_outcome(df, i, horizon, tp_pts, sl_pts)
        for k in out.keys(): out[k].append(r[k])
    return pd.DataFrame(out, index=df.index)

# ---------- bin stats ----------
def bin_stats(feature_series, outcome_df, n_bins=10):
    """
    feature_series: 1D Series for one indicator (at time t)
    outcome_df: DataFrame with columns long, short, mae, mfe, end_move (aligned index)
    Returns aggregated stats per quantile bin.
    """
    df = pd.concat([feature_series.rename("f"), outcome_df], axis=1).dropna()
    if df.empty: 
        return pd.DataFrame()

    # quantile bins
    try:
        bins = pd.qcut(df["f"], q=n_bins, labels=False, duplicates="drop")
    except Exception:
        # constant feature, etc.
        return pd.DataFrame()

    g = df.groupby(bins).agg(
        count=("f","size"),
        long_win_rate=("long","mean"),
        short_win_rate=("short","mean"),
        avg_end_pts=("end_move","mean"),
        med_end_pts=("end_move","median"),
        avg_mae=("mae","mean"),
        p50_mae=("mae",lambda x: np.percentile(x,50)),
        p95_mae=("mae",lambda x: np.percentile(x,95)),
        avg_mfe=("mfe","mean"),
        p50_mfe=("mfe",lambda x: np.percentile(x,50)),
        p95_mfe=("mfe",lambda x: np.percentile(x,95)),
    )
    g.index.name = "bin"
    g = g.reset_index()
    return g

# -------------- main runner --------------
def main():
    raw = load_data(DATA_PATH)
    feats = build_feature_frame(raw)
    # Trim the leading NA warmup across features
    mask = ~feats.isna().any(axis=1)
    raw2 = raw.loc[mask].reset_index(drop=True)
    feats = feats.loc[mask].reset_index(drop=True)

    all_rows = []
    for H in HORIZON_MIN:
        for TP in TP_POINTS:
            for sl_mult in SL_MULTIPLIERS:
                SL = TP * sl_mult
                print(f"Computing outcomes: H={H}m, TP={TP}, SL={SL} …")
                outcomes = compute_labels_and_excursions(raw2, H, TP, SL)

                # For each feature, compute decile table
                for f_name in feats.columns:
                    tbl = bin_stats(feats[f_name], outcomes, n_bins=N_BINS)
                    if tbl.empty: 
                        continue
                    tbl.insert(0, "feature", f_name)
                    tbl.insert(1, "horizon_min", H)
                    tbl.insert(2, "tp_pts", TP)
                    tbl.insert(3, "sl_pts", SL)
                    all_rows.append(tbl)

    if not all_rows:
        print("No stats generated (check inputs).")
        return

    result = pd.concat(all_rows, axis=0, ignore_index=True)

    # Convenience scores: best of either direction, and net directional edge
    result["max_win_rate"] = result[["long_win_rate","short_win_rate"]].max(axis=1)
    result["dir_edge"]     = result["long_win_rate"] - result["short_win_rate"]  # positive -> long skew, negative -> short skew

    # Save all stats
    result.to_csv("indicator_bin_stats.csv", index=False)

    # Pull top candidates (high precision + decent count)
    MIN_COUNT = 200  # tune: avoid spurious tiny bins
    tops = (result[result["count"]>=MIN_COUNT]
              .sort_values(["max_win_rate","count"], ascending=[False, False])
              .head(200))
    tops.to_csv("indicator_bin_top_candidates.csv", index=False)

    # Also summarize “20-pt default” slice if present
    if 20.0 in TP_POINTS and (20.0*np.array(SL_MULTIPLIERS)).size>0:
        slice20 = result[(result["tp_pts"]==20.0)]
        slice20.to_csv("indicator_bin_stats_20pt_only.csv", index=False)

    print("\n✅ Done. Files written:")
    print("- indicator_bin_stats.csv (ALL features × bins × TP/SL × horizon)")
    print("- indicator_bin_top_candidates.csv (top 200 by max_win_rate, count-filtered)")
    if 20.0 in TP_POINTS:
        print("- indicator_bin_stats_20pt_only.csv")

if __name__ == "__main__":
    # small alias to avoid a stray var above
    STP = []
    main()
