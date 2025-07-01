#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Calculate global switch windows from a global circuit TSV."
    )
    parser.add_argument(
        "tsv_path", type=str,
        help="Path to input TSV file with columns including "
             "iteration, parallelism_type, circuit_start_ts, circuit_end_ts, "
             "group_first_kernel_bytes, group_last_kernel_bytes"
    )
    args = parser.parse_args()

    input_path = args.tsv_path
    root, ext = os.path.splitext(input_path)
    output_path = f"{root}_global_window{ext}"

    df = pd.read_csv(input_path, sep="\t")
    df = df.sort_values("circuit_end_ts").reset_index(drop=True)

    windows = []

    for i in range(1, len(df)):
        prev = df.loc[i-1]
        curr = df.loc[i]
        prev_type = prev["parallelism_type"]
        curr_type = curr["parallelism_type"]
        # only care when the type switches
        if prev_type == curr_type:
            continue

        # determine window_type
        if prev_type == "DP" and curr_type == "PP":
            window_type = "dp-pp"
        elif prev_type == "PP" and curr_type == "DP":
            window_type = "pp-dp"
        else:
            # unexpected label—skip
            continue

        # start is the end_ts of the prev circuit
        wind_start = prev["circuit_end_ts"]
        # end is the start_ts of the curr circuit
        wind_end = curr["circuit_start_ts"]

        # enforce monotonicity vs. last window end
        if wind_start > wind_end:
            print(f"Overlapping windows: wind_start {wind_start}, last_wndow_end {wind_end}")
            wind_start = wind_end

        wind_dur = wind_end - wind_start

        windows.append({
            "iteration": prev["iteration"],
            "window_type": window_type,
            "wind_start_ts": wind_start,
            "wind_end_ts": wind_end,
            "wind_duration_ns": wind_dur,
            "kernel_before_bytes": prev["group_last_kernel_bytes"],
            "kernel_after_bytes": curr["group_first_kernel_bytes"],
        })

        last_window_end = wind_end

    out_df = pd.DataFrame(windows)
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"Global windows written to {output_path}")

if __name__ == "__main__":
    main()
