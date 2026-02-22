#!/usr/bin/env python3
"""
Merge OHLCV CSV Files
---------------------
Merges multiple OHLCV CSV files into a single file, sorted by timestamp.
Handles duplicates and ensures proper ordering.
"""

import sys
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def merge_ohlcv_files(input_files, output_file, chunk_size=1_000_000):
    """
    Merge multiple OHLCV CSV files into a single sorted file.
    
    Args:
        input_files: List of input CSV file paths
        output_file: Output CSV file path
        chunk_size: Chunk size for reading large files
    """
    print(f"Merging {len(input_files)} files into {output_file.name}...")
    print("=" * 60)
    
    all_data = []
    total_rows = 0
    
    # Read all files
    for i, input_file in enumerate(input_files, 1):
        print(f"\n[{i}/{len(input_files)}] Reading {input_file.name}...")
        
        if not input_file.exists():
            print(f"  ERROR: File not found: {input_file}")
            continue
        
        file_rows = 0
        chunk_count = 0
        
        try:
            for chunk in pd.read_csv(
                input_file,
                usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
                dtype={
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                    "symbol": "string"
                },
                chunksize=chunk_size
            ):
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"    Processed {chunk_count} chunks...")
                
                # Convert timestamp
                chunk["ts_event"] = pd.to_datetime(chunk["ts_event"], utc=True, errors="coerce")
                chunk = chunk.dropna(subset=["ts_event"])
                
                if not chunk.empty:
                    all_data.append(chunk)
                    file_rows += len(chunk)
            
            print(f"  ✓ Read {file_rows:,} rows from {input_file.name}")
            total_rows += file_rows
            
        except Exception as e:
            print(f"  ERROR reading {input_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_data:
        print("\nERROR: No data to merge!")
        return
    
    print(f"\nCombining {len(all_data)} chunks...")
    df = pd.concat(all_data, ignore_index=True)
    
    print(f"Total rows before deduplication: {len(df):,}")
    
    # Remove duplicates based on timestamp and symbol
    print("Removing duplicates...")
    df = df.drop_duplicates(subset=["ts_event", "symbol"], keep="first")
    
    print(f"Total rows after deduplication: {len(df):,}")
    duplicates_removed = total_rows - len(df)
    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed:,} duplicate rows")
    
    # Sort by timestamp
    print("Sorting by timestamp...")
    df = df.sort_values("ts_event").reset_index(drop=True)
    
    # Get date range
    min_date = df["ts_event"].min()
    max_date = df["ts_event"].max()
    
    print(f"\nDate range: {min_date} to {max_date}")
    print(f"Total unique rows: {len(df):,}")
    
    # Save to output file
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    file_size_mb = output_file.stat().st_size / (1024*1024)
    
    print(f"\n✓ Successfully merged files!")
    print(f"  Output file: {output_file}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Date range: {min_date.date()} to {max_date.date()}")
    print(f"  File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_ohlcv_data.py <output_file> <input_file1> [input_file2] ...")
        print("\nExample:")
        print(f"  python merge_ohlcv_data.py \\")
        print(f"    {DATA_DIR}/glbx-mdp3-20200927-20260221.ohlcv-1m.csv \\")
        print(f"    {DATA_DIR}/glbx-mdp3-20200927-20250926.ohlcv-1m.csv \\")
        print(f"    {DATA_DIR}/glbx-mdp3-20250821-20260221.ohlcv-1m.csv")
        sys.exit(1)
    
    output_file = Path(sys.argv[1])
    input_files = [Path(f) for f in sys.argv[2:]]
    
    merge_ohlcv_files(input_files, output_file)
