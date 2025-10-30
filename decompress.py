import pandas as pd

# Define the input and output file paths
input_file = 'glbx-mdp3-20240926-20250925.ohlcv-1m.csv.zst'
output_file = 'glbx-mdp3-20240926-20250925.ohlcv-1m.csv'

try:
    # Read the compressed CSV file into a pandas DataFrame
    print(f"Decompressing {input_file}...")
    df = pd.read_csv(input_file, compression='zstd')

    # Write the DataFrame to a new, uncompressed CSV file
    df.to_csv(output_file, index=False)

    print(f"✅ Successfully created {output_file}")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
