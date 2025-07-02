#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Synchronize DP timestamps of a second rank to a baseline."
    )
    parser.add_argument("baseline_tsv", help="Path to baseline TSV file")
    parser.add_argument("to_sync_tsv", help="Path to TSV file to synchronize")
    args = parser.parse_args()

    # Read both files
    df_base = pd.read_csv(args.baseline_tsv, sep="\t")
    df_sync = pd.read_csv(args.to_sync_tsv, sep="\t")

    target = 'ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)'

    # Find, per-iteration, the last end_ts of that DP kernel
    def last_dp_end(df):
        mask = (
            (df["kernel_name"] == target) &
            (df["parallelism_type"] == "DP")
        )
        return df.loc[mask].groupby("iteration")["end_ts"].max()

    base_ends = last_dp_end(df_base)
   # print(f"base_ends: {base_ends}")
    sync_ends = last_dp_end(df_sync)
   # print(f"sync_ends: {sync_ends}")

    # Compute offsets: sync_end - base_end
    offsets = {}
    for itr, base_end in base_ends.items():
        if itr not in sync_ends:
            raise KeyError(f"Iteration {itr} missing target in to_sync file")
        offsets[itr] = sync_ends[itr] - base_end

    print("Computed offsets (to_subtract = sync_end - base_end):")
    for itr, off in offsets.items():
        print(f" Iter {itr}: offset = {off} ns")

    # Apply per-iteration correction to df_sync
    def apply_offset(row):
        off = offsets.get(row["iteration"], 0)
        row["start_ts"] -= off
        row["end_ts"]   -= off
        return row

    df_sync = df_sync.apply(apply_offset, axis=1)

    # Write out
    root, ext = os.path.splitext(args.to_sync_tsv)
    out_path = f"{root}_synch{ext}"
    df_sync.to_csv(out_path, sep="\t", index=False)
    print(f"Synchronized TSV written to {out_path}")

if __name__ == "__main__":
    main()
