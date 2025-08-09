#!/usr/bin/env python3
import argparse
import pandas as pd
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Collect the Parallelism labels for all NCCL:ncclAllGather events"
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file containing 'Name' and 'Parallelism' columns"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Check required columns
    for col in ("Name", "Parallelism"):
        if col not in df.columns:
            sys.exit(f"Error: Required column '{col}' not found in {args.input_csv}")

    # Filter for NCCL:ncclAllGather and collect their Parallelism labels
    mask = df["Name"] == "NCCL:ncclAllGather"
    parallelism_list = df.loc[mask, "Parallelism"].tolist()

    print(parallelism_list)

if __name__ == "__main__":
    main()
