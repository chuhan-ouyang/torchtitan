#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def load_sendrecv_offsets(df):
    """
    For each iteration in df, find the first ncclDevKernel_SendRecv(...) row
    and return a dict iteration -> end_ts.
    """
    target = 'ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)'
    sr = df[df['kernel_name'] == target]
    return sr.groupby('iteration')['end_ts'].first().to_dict()

def main():
    p = argparse.ArgumentParser(
        description="Align PP events in file B to file A based on the first SendRecv offset per iteration."
    )
    p.add_argument("baseline", help="Path to baseline TSV")
    p.add_argument("to_sync",  help="Path to TSV to be time-shifted")
    args = p.parse_args()

    # derive output paths
    base_root, base_ext = os.path.splitext(args.baseline)
    sync_root, sync_ext = os.path.splitext(args.to_sync)
    baseline_out = f"{base_root}_ppsync{base_ext}"
    sync_out      = f"{sync_root}_ppsync{sync_ext}"

    # load
    df_base = pd.read_csv(args.baseline, sep="\t")
    df_sync = pd.read_csv(args.to_sync,  sep="\t")

    # compute per-iteration “zero” times
    base_offsets = load_sendrecv_offsets(df_base)
    sync_offsets = load_sendrecv_offsets(df_sync)

    # sanity: same iterations?
    missing = set(base_offsets) - set(sync_offsets)
    if missing:
        raise ValueError(f"No SendRecv in to_sync for iterations {sorted(missing)}")

    # 1) write baseline copy unchanged
    df_base.to_csv(baseline_out, sep="\t", index=False)
    print(f"Baseline copy written to {baseline_out}")

    # 2) shift only PP rows in df_sync, keep original order and all other rows
    df_out = df_sync.copy()

    # compute per-iteration shift = sync_offset - base_offset
    shifts = {it: sync_offsets[it] - base_offsets[it] for it in base_offsets}

    # apply shift to PP rows
    mask = df_out['parallelism_type'] == "PP"
    for itr, offset in shifts.items():
        itr_mask = mask & (df_out['iteration'] == itr)
        df_out.loc[itr_mask, 'start_ts'] -= offset
        df_out.loc[itr_mask, 'end_ts']   -= offset

    # write full table, original order intact
    df_out.to_csv(sync_out, sep="\t", index=False)
    print(f"Synchronized PP events (with all rows) written to {sync_out}")

if __name__ == "__main__":
    main()
