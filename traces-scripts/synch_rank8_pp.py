#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def third_pp_end(df, target):
    """
    For each iteration in df, find the 3rd smallest end_ts among PP SendRecv kernels.
    Returns a Series indexed by iteration with the 3rd end_ts.
    """
    # Filter down to PP SendRecv rows
    df_pp = df[
        (df["kernel_name"] == target) &
        (df["parallelism_type"] == "PP")
    ]
    def get_third(series: pd.Series) -> int:
        if len(series) < 3:
            raise KeyError(f"Iteration {series.name} has fewer than 3 SendRecv rows")
        # sort the timestamps and pick the 3rd (index 2)
        return series.sort_values().iloc[2]

    # Group only the 'end_ts' values, so apply works on a SeriesGroupBy
    return df_pp.groupby("iteration")["end_ts"].apply(get_third)

def main():
    parser = argparse.ArgumentParser(
        description="Synchronize PP timestamps of a second rank to a baseline using the 3rd SendRecv per iteration."
    )
    parser.add_argument("baseline_tsv", help="Path to baseline TSV file")
    parser.add_argument("to_sync_tsv", help="Path to TSV file to synchronize")
    args = parser.parse_args()

    df_base = pd.read_csv(args.baseline_tsv, sep="\t")
    df_sync = pd.read_csv(args.to_sync_tsv, sep="\t")

    target = 'ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)'

    # Find the 3rd SendRecv end_ts in each
    base_third = third_pp_end(df_base, target)
    sync_third = third_pp_end(df_sync, target)

    # Compute per-iteration offsets
    offsets = (sync_third - base_third).to_dict()
    print("Computed 3rd-SendRecv PP offsets (to_subtract = sync_third - base_third):")
    for itr, off in offsets.items():
        print(f"  Iteration {itr}: offset = {off} ns")

    # Subtract offset from all rows in that iteration
    df_sync["start_ts"] = df_sync["start_ts"] - df_sync["iteration"].map(offsets).fillna(0)
    df_sync["end_ts"]   = df_sync["end_ts"]   - df_sync["iteration"].map(offsets).fillna(0)

    # Write out
    root, ext = os.path.splitext(args.to_sync_tsv)
    out_path = f"{root}_synch{ext}"
    df_sync.to_csv(out_path, sep="\t", index=False)
    print(f"Synchronized TSV written to {out_path}")

if __name__ == "__main__":
    main()
