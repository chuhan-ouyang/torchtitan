import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(description="Detect parallelism switch windows globally.")
    parser.add_argument("tsvpath", type=str, help="Path to input TSV file")
    args = parser.parse_args()

    input_path = args.tsvpath
    output_path = input_path.replace(".tsv", "_window.tsv")

    df = pd.read_csv(input_path, sep="\t")

    required_cols = {
        "iteration", "parallelism_type", "start_ts", "end_ts", "bytes"
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in input TSV: {required_cols - set(df.columns)}")

    # Sort globally by start_ts
    df_sorted = df.sort_values("start_ts").reset_index(drop=True)

    window_rows = []

    for i in range(1, len(df_sorted)):
        prev_type = df_sorted.loc[i - 1, "parallelism_type"]
        curr_type = df_sorted.loc[i, "parallelism_type"]

        if prev_type == "DP" and curr_type == "PP":
            window_type = "dp-pp"
        elif prev_type == "PP" and curr_type == "DP":
            window_type = "pp-dp"
        else:
            continue

        wind_start_ts = df_sorted.loc[i - 1, "end_ts"]
        wind_end_ts = df_sorted.loc[i, "start_ts"]
        wind_duration_ns = wind_end_ts - wind_start_ts
        kernel_before_bytes = df_sorted.loc[i - 1, "bytes"]
        kernel_after_bytes = df_sorted.loc[i, "bytes"]
        iteration = df_sorted.loc[i - 1, "iteration"]

        window_rows.append({
            "iteration": iteration,
            "window_type": window_type,
            "wind_start_ts": wind_start_ts,
            "wind_end_ts": wind_end_ts,
            "wind_duration_ns": wind_duration_ns,
            "kernel_before_bytes": kernel_before_bytes,
            "kernel_after_bytes": kernel_after_bytes
        })

    window_df = pd.DataFrame(window_rows)
    window_df.to_csv(output_path, sep="\t", index=False)
    print(f"Window info written to {output_path}")

if __name__ == "__main__":
    main()
