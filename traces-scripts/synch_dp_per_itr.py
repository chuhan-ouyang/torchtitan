#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def load_dp_offsets(df):
    """
    For each iteration in df, find the first DP ncclDevKernel_AllReduce_Sum_f32_RING_LL(...) row
    and return a dict iteration -> end_ts.
    """
    target = 'ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)'
    dp_rows = df[
        (df['kernel_name'] == target) &
        (df['parallelism_type'] == 'DP')
    ]
    return dp_rows.groupby('iteration')['end_ts'].first().to_dict()

def main():
    p = argparse.ArgumentParser(
        description="Align DP events in file B to file A based on the first AllReduce DP offset per iteration."
    )
    p.add_argument("baseline", help="Path to baseline TSV")
    p.add_argument("to_sync",  help="Path to TSV to be time-shifted")
    args = p.parse_args()

    # derive output paths
    base_root, base_ext = os.path.splitext(args.baseline)
    sync_root, sync_ext = os.path.splitext(args.to_sync)
    baseline_out = f"{base_root}_dpsync{base_ext}"
    sync_out      = f"{sync_root}_dpsync{sync_ext}"

    # load
    df_base = pd.read_csv(args.baseline, sep="\t")
    df_sync = pd.read_csv(args.to_sync,  sep="\t")

    # compute per-iteration “zero” times from the DP kernel
    base_offsets = load_dp_offsets(df_base)
    sync_offsets = load_dp_offsets(df_sync)

    # sanity: same iterations?
    missing = set(base_offsets) - set(sync_offsets)
    if missing:
        raise ValueError(f"No DP AllReduce in to_sync for iterations {sorted(missing)}")

    # 1) write baseline copy unchanged
    df_base.to_csv(baseline_out, sep="\t", index=False)
    print(f"Baseline copy written to {baseline_out}")

    # 2) shift only DP rows in df_sync, keep original order and all other rows
    df_out = df_sync.copy()

    # compute per-iteration shift = sync_offset - base_offset
    shifts = {it: sync_offsets[it] - base_offsets[it] for it in base_offsets}

    # apply shift to DP rows
    dp_mask = df_out['parallelism_type'] == "DP"
    for itr, offset in shifts.items():
        mask = dp_mask & (df_out['iteration'] == itr)
        df_out.loc[mask, 'start_ts'] -= offset
        df_out.loc[mask, 'end_ts']   -= offset

    # write full table, original order intact
    df_out.to_csv(sync_out, sep="\t", index=False)
    print(f"Synchronized DP events (with all rows) written to {sync_out}")

if __name__ == "__main__":
    main()
