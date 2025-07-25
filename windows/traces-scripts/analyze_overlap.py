#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Detect and print overlapping windows line-by-line."
    )
    parser.add_argument(
        "tsv_path",
        help="Path to input TSV file with start_ts and end_ts columns"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.tsv_path, sep="\t")
    prev = None
    found = False

    for idx, row in df.iterrows():
        if prev is not None:
            if prev["end_ts"] > row["start_ts"]:
                found = True
                overlap = prev["end_ts"] - row["start_ts"]
                print(f"Overlap between line {prev_idx} and {idx}:")
                print(f"  ▶ iteration={prev['iteration']}, "
                      f"kernel_name='{prev['kernel_name']}', end_ts={prev['end_ts']}")
                print(f"  ▶ iteration={row['iteration']}, "
                      f"kernel_name='{row['kernel_name']}', start_ts={row['start_ts']}")
                print(f"  ↳ Overlap: {overlap} ns\n")
        prev = row
        prev_idx = idx

    if not found:
        print("No overlaps detected.")

if __name__ == "__main__":
    main()
