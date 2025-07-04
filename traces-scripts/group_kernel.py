#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Group consecutive NCCL AllGather/ReduceScatter kernels into single entries"
    )
    parser.add_argument("tsv_path", help="Path to input TSV file")
    args = parser.parse_args()

    input_path = args.tsv_path
    root, ext = os.path.splitext(input_path)
    output_path = f"{root}_grouped{ext}"

    # Read input
    df = pd.read_csv(input_path, sep="\t")

    # Define which kernels to group
    group_targets = {
        'ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)',
        'ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)'
    }

    rows = []
    n = len(df)
    i = 0
    while i < n:
        row = df.iloc[i]
        name = row['kernel_name']
        # If this row is one of the group targets, gather consecutive run
        if name in group_targets:
            # find end of the group run
            j = i
            total_bytes = 0
            while j < n and df.iloc[j]['kernel_name'] in group_targets:
                total_bytes += df.iloc[j]['bytes']
                j += 1
            # df indices [i, j)
            first = df.iloc[i]
            last = df.iloc[j-1]
            # build new entry
            entry = first.to_dict()
            # update timing
            entry['start_ts'] = first['start_ts']
            entry['end_ts'] = last['end_ts']
            entry['duration_ns'] = entry['end_ts'] - entry['start_ts']
            # set grouped bytes columns
            entry['group_first_kernel_bytes'] = total_bytes
            entry['group_last_kernel_bytes'] = total_bytes
            rows.append(entry)
            i = j
        else:
            # non-target, pass through
            entry = row.to_dict()
            # set grouped bytes to original bytes
            entry['group_first_kernel_bytes'] = entry['bytes']
            entry['group_last_kernel_bytes'] = entry['bytes']
            rows.append(entry)
            i += 1

    # build output DataFrame
    out_df = pd.DataFrame(rows)

    # drop original bytes and in_msg_nelems if present
    for col in ['bytes', 'in_msg_nelems']:
        if col in out_df.columns:
            out_df.drop(columns=[col], inplace=True)

    # write to TSV
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"Grouped data written to {output_path}")

if __name__ == "__main__":
    main()
