import pandas as pd
from pathlib import Path

# --- Load your full reversal data ---
OUT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(str(OUT_DIR / "reversal_events.csv"), parse_dates=["touch_ts", "created_ts", "et_date"])

# --- Define your top 10 reversal types (singles or confluences) ---
top_signals = [
    "sma200+vp_on_poc",  # strongest combo
    "sma200+sma50", 
    "ib_high", "ib_low",
    "d1_high", "d1_low",
    "h4_high", "h4_low",
    "vp_on_vah", "vp_on_val",
    "london_high", "london_low",
    "bb_up+d1_high",
    "d1_low+london_low",
]

# --- Find matches either by level_type or combo name ---
mask = (
    df["level_type"].isin(
        [lvl for lvl in top_signals if "+" not in lvl]  # singles
    )
    | df["combo_10pt"].apply(lambda x: any(k in str(x) for k in top_signals))
)

# --- Optional: filter to meaningful reactions (e.g. ≥ 1×ATR move) ---
mask_strong = df["disp_atr"] >= 1.0

# Combine filters
subset = df[mask & mask_strong].copy()

# --- Extract the trading day and sessions for manual review ---
cols = [
    "et_date", "level_type", "side", "origin_session", "touch_ts",
    "touch_window", "touch_price", "mfe_opp", "mae_thru", "disp_atr", "combo_10pt"
]
subset = subset[cols].sort_values(["et_date", "touch_ts"])

# --- Summarize how many top events per day ---
day_summary = subset.groupby("et_date")["level_type"].count().reset_index()
day_summary.columns = ["date", "top_event_count"]

# --- Save both ---
subset.to_csv(str(OUT_DIR / "reversal_top_events_detailed.csv"), index=False)
day_summary.to_csv(str(OUT_DIR / "reversal_top_events_by_day.csv"), index=False)

print(f"✅ Found {len(subset)} top reversal touches across {len(day_summary)} trading days.")
print("Files saved: reversal_top_events_detailed.csv and reversal_top_events_by_day.csv")
