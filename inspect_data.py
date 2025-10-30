import pandas as pd

# Define the input file path
input_file = 'glbx-mdp3-20240926-20250925.ohlcv-1m.csv'

try:
    print(f"Reading {input_file}...")
    # Read the data, we only need the timestamp and symbol columns
    df = pd.read_csv(input_file, usecols=['ts_event', 'symbol'])

    print("Checking for duplicate timestamps with different symbols...")

    # Clean up symbols to only include outrights
    df = df[df["symbol"].str.startswith("NQ") & ~df["symbol"].str.contains("-")]

    # Count the number of unique symbols for each timestamp
    duplicate_counts = df.groupby('ts_event')['symbol'].nunique()

    # Filter for timestamps that have more than one unique symbol
    timestamps_with_duplicates = duplicate_counts[duplicate_counts > 1]

    if not timestamps_with_duplicates.empty:
        print("\nFound timestamps with multiple NQ contracts:")
        print(timestamps_with_duplicates.head())
        print(f"\nConclusion: The file contains data for multiple contracts at the same time.")
        print(f"This is likely the cause of the abnormally large ranges.")
        print(f"Total number of 1-minute intervals with overlapping contracts: {len(timestamps_with_duplicates)}")
    else:
        print("\nConclusion: No duplicate timestamps found for different symbols.")
        print("Each 1-minute interval corresponds to a single NQ contract.")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found. Please ensure you have decompressed it first.")
except Exception as e:
    print(f"An error occurred: {e}")

