#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def get_third_sendrecv(df):
    """Return a Series of the 3rd-smallest end_ts of SendRecv per iteration."""
    target = "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)"
    df_t = df[df["kernel_name"] == target]
    def third(ts: pd.Series):
        if len(ts) < 3:
            raise KeyError(f"Iteration {ts.name} has fewer than 3 SendRecv rows")
        return ts.nsmallest(3).iloc[-1]
    return df_t.groupby("iteration")["end_ts"].apply(third)

def get_last_reducescatter(df):
    """Return a Series of the max end_ts of ReduceScatter per iteration."""
    target = "ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
    df_t = df[df["kernel_name"] == target]
    return df_t.groupby("iteration")["end_ts"].max()

def compute_offsets(base_df, sync_df, extractor):
    """
    Given two DataFrames and an extractor(series)->Series mapping each,
    compute sync_ts - base_ts per iteration.
    """
    base_ts = extractor(base_df)
    sync_ts = extractor(sync_df)
    common = base_ts.index.intersection(sync_ts.index)
    offsets = {}
    for itr in common:
        offsets[itr] = float(sync_ts[itr] - base_ts[itr])
    return offsets

def apply_offsets(df, offsets, ptype):
    """Subtract per-iteration offset from start_ts & end_ts for rows with parallelism_type==ptype."""
    mask = df["parallelism_type"] == ptype
    for itr, off in offsets.items():
        m = mask & (df["iteration"] == itr)
        df.loc[m, ["start_ts","end_ts"]] -= off

def main():
    parser = argparse.ArgumentParser(
        description="Sync PP by 3rd SendRecv and DP by last ReduceScatter differences"
    )
    parser.add_argument("pp_base", help="TSV with PP baseline timings")
    parser.add_argument("dp_base", help="TSV with DP baseline timings")
    parser.add_argument("to_sync", help="TSV to be synchronized")
    args = parser.parse_args()

    pp_base   = pd.read_csv(args.pp_base, sep="\t")
    dp_base   = pd.read_csv(args.dp_base, sep="\t")
    sync_df   = pd.read_csv(args.to_sync, sep="\t")

    # Compute PP offsets
    pp_offsets = compute_offsets(pp_base, sync_df, get_third_sendrecv)
    print("PP (3rd SendRecv) offsets (sync − base):")
    for itr, off in pp_offsets.items():
        print(f"  Iter {itr}: {off:.0f} ns")
    apply_offsets(sync_df, pp_offsets, "PP")

    # Compute DP offsets
    dp_offsets = compute_offsets(dp_base, sync_df, get_last_reducescatter)
    print("DP (last ReduceScatter) offsets (sync − base):")
    for itr, off in dp_offsets.items():
        print(f"  Iter {itr}: {off:.0f} ns")
    apply_offsets(sync_df, dp_offsets, "DP")

    # Write out
    root, ext = os.path.splitext(args.to_sync)
    out_path = f"{root}_synch{ext}"
    sync_df.to_csv(out_path, sep="\t", index=False)
    print(f"Synchronized TSV written to {out_path}")

if __name__ == "__main__":
    main()
