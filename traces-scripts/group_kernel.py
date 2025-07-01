#!/usr/bin/env python3
import argparse
import pandas as pd
import os

GROUP_NAMES = {
    "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
    "ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
}

def main():
    parser = argparse.ArgumentParser(
        description="Group consecutive AllGather/ReduceScatter kernels into single entries."
    )
    parser.add_argument("tsv_path", help="Path to input TSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.tsv_path, sep="\t")

    to_drop = {"in_msg_nelems", "bytes"}

    out_rows = []
    i = 0
    n = len(df)
    while i < n:
        name = df.loc[i, "kernel_name"]
        if name in GROUP_NAMES:
            # start a group
            start_idx = i
            # extend as long as next rows are in GROUP_NAMES
            while i + 1 < n and df.loc[i + 1, "kernel_name"] in GROUP_NAMES:
                i += 1
            end_idx = i
            first = df.loc[start_idx]
            last = df.loc[end_idx]

            # build new row from first, overriding start/end and bytes
            new = first.drop(labels=to_drop).to_dict()
            new["start_ts"] = first["start_ts"]
            new["end_ts"]   = last["end_ts"]
            new['duration_ns'] = new['end_ts'] - new['start_ts']
            new["group_first_kernel_bytes"] = int(first["bytes"])
            new["group_last_kernel_bytes"]  = int(last["bytes"])

            out_rows.append(new)
            i += 1
        else:
            # not groupable: emit single
            row = df.loc[i]
            new = row.drop(labels=to_drop).to_dict()
            new["group_first_kernel_bytes"] = int(row["bytes"])
            new["group_last_kernel_bytes"]  = int(row["bytes"])
            out_rows.append(new)
            i += 1

    out_df = pd.DataFrame(out_rows)

    base, ext = os.path.splitext(args.tsv_path)
    out_path = f"{base}_grouped{ext}"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote grouped kernels to {out_path}")

if __name__ == "__main__":
    main()
