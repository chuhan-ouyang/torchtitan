#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Filter a CSV for rows with Parallelism=='PP'"
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file"
    )
    args = parser.parse_args()

    # Read input
    df = pd.read_csv(args.input_csv)

    # Filter for PP
    df_pp = df[df["Parallelism"] == "PP"]

    # Construct output filename: <basename>_pp.csv
    base, ext = os.path.splitext(args.input_csv)
    output_csv = f"{base}_pp{ext}"

    # Write filtered data
    df_pp.to_csv(output_csv, index=False)
    print(f"Filtered {len(df_pp)} rows (Parallelism=='PP') → {output_csv}")

if __name__ == "__main__":
    main()
