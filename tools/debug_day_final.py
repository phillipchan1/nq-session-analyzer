import pandas as pd

# This is a final, detailed debug script to inspect the raw data for a specific day.

# --- Configuration ---
TARGET_DATE = "2025-09-05"
INPUT_FILE = "glbx-mdp3-20240926-20250925.ohlcv-1m.csv.zst"

try:
    print(f"--- Starting Final Debug for {TARGET_DATE} ---")

    # Load and prep the data
    df = pd.read_csv(INPUT_FILE, compression="zstd")
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["ts_event"].dt.date
    
    # Filter for the target date
    df = df[df['date'] == pd.to_datetime(TARGET_DATE).date()]
    
    if df.empty:
        raise ValueError(f"No data found for {TARGET_DATE}")

    # Isolate the RTH session window
    rth_mask = (df["ts_event"].dt.time >= pd.to_datetime("09:30").time()) & \
               (df["ts_event"].dt.time < pd.to_datetime("10:15").time())
    rth_df = df[rth_mask]

    if rth_df.empty:
        raise ValueError(f"No data found in RTH window for {TARGET_DATE}")

    # --- Key Analysis: Calculate range for ALL contracts ---
    rth_agg = rth_df.groupby(["date", "symbol"]).agg(
        high=("high", "max"),
        low=("low", "min"),
        volume=("volume", "sum")
    ).reset_index()

    rth_agg["range"] = rth_agg["high"] - rth_agg["low"]
    
    print("\n--- Calculated RTH Session Data for ALL Contracts ---")
    print("This table shows what the script sees in the raw data file.")
    print(rth_agg[['symbol', 'high', 'low', 'range', 'volume']])
    print("----------------------------------------------------")

    # --- Show the Decision ---
    if not rth_agg.empty:
        winner_idx = rth_agg["range"].idxmax()
        winner_symbol = rth_agg.loc[winner_idx]['symbol']
        winner_range = rth_agg.loc[winner_idx]['range']
        
        print(f"\nDecision: The contract with the largest range is '{winner_symbol}' with a range of {winner_range:.2f}.")
    else:
        print("\nNo contracts found to make a decision.")


except FileNotFoundError:
    print(f"Error: The file {INPUT_FILE} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

