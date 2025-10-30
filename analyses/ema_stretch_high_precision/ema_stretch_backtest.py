import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

# ======================
# Config (edit here)
# ======================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = Path(__file__).resolve().parent
INPUT_CSV = str(DATA_DIR / "glbx-mdp3-20240926-20250925.ohlcv-1m.csv.zst")  # your Databento 1m OHLCV export
INPUT_COMPRESSION = "zstd"

# RTH filter (NY time). Set to None to disable.
RTH_START = "09:30"
RTH_END   = "16:00"

# Rolling window for deciles (approx 20 trading days of 1m bars)
DECILE_WINDOW = 8000   # must be big enough for stable deciles

# “High-precision” candidates to test (from your table)
# Each tuple: (feature, bin, direction, tp_pts, sl_pts, horizon_min, label)
# feature ∈ {"dist_ema_9","dist_ema_20","dist_ema_50"}
# direction ∈ {"long","short"}
CANDIDATES = [
    # “extremely below EMA” → LONG
    ("dist_ema_50", 0, "long", 30.0, 30.0, 30, "d50_bin0_L_30/30_h30"),
    ("dist_ema_50", 0, "long", 30.0, 30.0, 20, "d50_bin0_L_30/30_h20"),
    ("dist_ema_20", 0, "long", 30.0, 30.0, 30, "d20_bin0_L_30/30_h30"),
    ("dist_ema_20", 0, "long", 30.0, 30.0, 20, "d20_bin0_L_30/30_h20"),
    ("dist_ema_9",  1, "long", 30.0, 30.0, 30, "d9_bin1_L_30/30_h30"),
    # “extremely above EMA” → SHORT
    ("dist_ema_50", 9, "short", 30.0, 30.0, 30, "d50_bin9_S_30/30_h30"),
    ("dist_ema_20", 9, "short", 30.0, 30.0, 30, "d20_bin9_S_30/30_h30"),
    ("dist_ema_9",  9, "short", 30.0, 30.0, 30, "d9_bin9_S_30/30_h30"),
]

OUTPUT_TRADES_CSV = str(OUT_DIR / "trades_high_precision.csv")

# ======================
# Helpers
# ======================
def to_ny(idx_utc) -> pd.DatetimeIndex:
    if isinstance(idx_utc, pd.Series):
        return idx_utc.dt.tz_convert("America/New_York")
    else:
        return idx_utc.tz_convert("America/New_York")

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def assign_decile(series: pd.Series, q_edges: pd.DataFrame) -> pd.Series:
    """
    For each row t, place series[t] into decile using quantile edges computed from PRIOR window (edges shifted by 1).
    q_edges has 9 columns: q10..q90, already shifted by 1 row to avoid leakage.
    """
    bins = pd.Series(np.nan, index=series.index, dtype="float64")

    # Order of comparisons: <= q10 => bin0, <= q20 => bin1, ..., > q90 => bin9
    q_cols = [f"q{q}" for q in (10,20,30,40,50,60,70,80,90)]

    # Vectorized approach: create a 2D array comparisons
    vals = series.values
    edges = q_edges[q_cols].values  # shape (n,9)

    # For rows with any NaN edge, leave NaN bin (insufficient history)
    valid = np.all(np.isfinite(edges), axis=1) & np.isfinite(vals)
    out = np.full(len(series), np.nan)

    # For valid rows, compute bin index
    v = vals[valid][:, None]
    e = edges[valid]
    # compare v <= each edge; sum True gives first bin index; values above all edges => bin 9
    cmp = v <= e
    first_le = cmp.argmax(axis=1)  # if all False, returns 0, but we’ll detect that case
    # Where all False, bin = 9; else = first_le
    none = ~cmp.any(axis=1)
    bin_idx = first_le.astype(float)
    bin_idx[none] = 9.0

    out[valid] = bin_idx
    return pd.Series(out, index=series.index)

def build_quantile_edges(series: pd.Series, window: int) -> pd.DataFrame:
    """
    Rolling quantiles (q10..q90) using only PAST data:
    compute rolling on full series, then shift by 1 to ensure edges at t come from t-1..t-window.
    """
    qs = {}
    for q in (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9):
        qs[f"q{int(q*100)}"] = series.rolling(window, min_periods=max(1000, window//4)).quantile(q)
    edges = pd.DataFrame(qs)
    return edges.shift(1)

def simulate(bar_slice: pd.DataFrame, entry_price: float, direction: str,
             tp_pts: float, sl_pts: float, horizon_min: int):
    """
    Walk forward minute-by-minute; stop when TP/SL first touched or horizon time exceeded.
    Returns (exit_price, outcome, exit_ts_utc, exit_ts_ny).
    """
    if direction == "long":
        tp = entry_price + tp_pts
        sl = entry_price - sl_pts
    else:
        tp = entry_price - tp_pts
        sl = entry_price + sl_pts

    outcome = "TIME"
    exit_price = bar_slice.iloc[-1]["close"]
    exit_ts_utc = bar_slice.iloc[-1]["ts_event"]
    exit_ts_ny  = bar_slice.iloc[-1]["ts_ny"]

    for _, r in bar_slice.iterrows():
        hi = r["high"]; lo = r["low"]
        t_utc = r["ts_event"]; t_ny = r["ts_ny"]

        hit_tp = (hi >= tp) if direction=="long" else (lo <= tp)
        hit_sl = (lo <= sl) if direction=="long" else (hi >= sl)

        if hit_tp and hit_sl:
            # neutral: whichever is closer to entry
            d_tp = abs(tp - entry_price)
            d_sl = abs(entry_price - sl)
            if d_tp <= d_sl:
                outcome, exit_price = "TP", tp
            else:
                outcome, exit_price = "SL", sl
            exit_ts_utc, exit_ts_ny = t_utc, t_ny
            break
        elif hit_tp:
            outcome, exit_price = "TP", tp
            exit_ts_utc, exit_ts_ny = t_utc, t_ny
            break
        elif hit_sl:
            outcome, exit_price = "SL", sl
            exit_ts_utc, exit_ts_ny = t_utc, t_ny
            break

    return exit_price, outcome, exit_ts_utc, exit_ts_ny

# ======================
# Load & prep
# ======================
print("Loading…")
df = pd.read_csv(INPUT_CSV, compression=INPUT_COMPRESSION)

# Keep outright NQ contracts only
df = df[df["symbol"].str.startswith("NQ") & ~df["symbol"].str.contains("-")].copy()

df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
df["ts_ny"] = to_ny(df["ts_event"])
df["date_ny"] = df["ts_ny"].dt.date

# Pick front-month per day by max RTH range 09:30–10:15 (simple, consistent with your earlier runs)
def in_window(ts_ny, s="09:30", e="10:15"):
    t = ts_ny.dt.time
    return (t >= pd.to_datetime(s).time()) & (t <= pd.to_datetime(e).time())

rth_fm = df.loc[in_window(df["ts_ny"])].copy()
fm = (rth_fm.groupby(["date_ny","symbol"])
              .agg(h=("high","max"), l=("low","min"))
              .assign(r=lambda x: x.h - x.l)
              .reset_index())
fm = fm.loc[fm.groupby("date_ny")["r"].idxmax(), ["date_ny","symbol"]].rename(columns={"symbol":"fm_sym"})
df = df.merge(fm, on="date_ny", how="left")
df = df[df["symbol"] == df["fm_sym"]].drop(columns=["fm_sym"]).sort_values("ts_event").reset_index(drop=True)

# RTH filter if set
if RTH_START and RTH_END:
    t = df["ts_ny"].dt.time
    rth_mask = (t >= pd.to_datetime(RTH_START).time()) & (t <= pd.to_datetime(RTH_END).time())
    df = df.loc[rth_mask].copy()

# Indicators
print("Indicators…")
df["ema9"]  = ema(df["close"], 9)
df["ema20"] = ema(df["close"], 20)
df["ema50"] = ema(df["close"], 50)

df["dist_ema_9"]  = df["close"] - df["ema9"]
df["dist_ema_20"] = df["close"] - df["ema20"]
df["dist_ema_50"] = df["close"] - df["ema50"]

# Rolling decile edges (PAST only)
print("Rolling deciles (no look-ahead)…")
edges9  = build_quantile_edges(df["dist_ema_9"],  DECILE_WINDOW)
edges20 = build_quantile_edges(df["dist_ema_20"], DECILE_WINDOW)
edges50 = build_quantile_edges(df["dist_ema_50"], DECILE_WINDOW)

df["bin_9"]  = assign_decile(df["dist_ema_9"],  edges9)
df["bin_20"] = assign_decile(df["dist_ema_20"], edges20)
df["bin_50"] = assign_decile(df["dist_ema_50"], edges50)

# ======================
# Generate signals & simulate
# ======================
print("Simulating candidates…")
records = []

# Precompute for speed
df_vals = df[["ts_event","ts_ny","open","high","low","close","bin_9","bin_20","bin_50"]].copy()

for feature, target_bin, direction, tp_pts, sl_pts, horizon_min, label in CANDIDATES:
    if feature == "dist_ema_9":
        bins = df_vals["bin_9"]
    elif feature == "dist_ema_20":
        bins = df_vals["bin_20"]
    else:
        bins = df_vals["bin_50"]

    sig_mask = (bins == float(target_bin))
    idxs = np.where(sig_mask.values)[0]

    for i in idxs:
        # entry at close of bar i
        row = df_vals.iloc[i]
        entry_ts = row["ts_event"]
        entry_ts_ny = row["ts_ny"]
        entry = float(row["close"])

        # horizon slice = next N minutes (exclusive of entry bar)
        # we’ll just take the next X rows whose timestamps < entry_ts + horizon
        horizon_end = entry_ts + timedelta(minutes=horizon_min)
        j = i + 1
        # guard
        if j >= len(df_vals):
            continue
        look = df_vals.iloc[j:][df_vals.iloc[j:]["ts_event"] <= horizon_end]
        if look.empty:
            continue

        exit_price, outcome, exit_ts, exit_ts_ny = simulate(look, entry, direction, tp_pts, sl_pts, horizon_min)
        points = (exit_price - entry) if direction=="long" else (entry - exit_price)

        records.append({
            "label": label,
            "feature": feature,
            "bin": target_bin,
            "direction": direction,
            "tp_pts": tp_pts,
            "sl_pts": sl_pts,
            "horizon_min": horizon_min,
            "ts_entry_utc": entry_ts,
            "ts_entry_ny": entry_ts_ny,
            "entry": entry,
            "ts_exit_utc": exit_ts,
            "ts_exit_ny": exit_ts_ny,
            "exit": exit_price,
            "outcome": outcome,
            "points": points
        })

trades = pd.DataFrame.from_records(records).sort_values("ts_entry_utc")

if trades.empty:
    print("No trades generated — likely not enough history for deciles. Try lowering DECILE_WINDOW.")
else:
    # Summary by label
    def summarize(g):
        n = len(g)
        wr = (g["outcome"] == "TP").mean() if n else 0.0
        return pd.Series({
            "n": n,
            "win_rate": wr,
            "avg_pts": g["points"].mean(),
            "med_pts": g["points"].median()
        })

    print("\n=== Summary by setup ===")
    print(trades.groupby("label").apply(summarize).sort_values(["win_rate","n"], ascending=[False, False]))

    # Overall
    n_all = len(trades)
    wr_all = (trades["outcome"] == "TP").mean() if n_all else 0.0
    print("\n=== Overall ===")
    print(f"n_trades: {n_all}")
    print(f"win_rate: {wr_all:.3f}")
    print(f"avg_points: {trades['points'].mean():.2f}")
    print(f"median_points: {trades['points'].median():.2f}")

    trades.to_csv(OUTPUT_TRADES_CSV, index=False)
    print(f"\nSaved {OUTPUT_TRADES_CSV}")
