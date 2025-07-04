#!/usr/bin/env python3
import argparse
import os
import ast
import json
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Condense PP and DP circuit entries in a TSV."
    )
    parser.add_argument("tsv_path", help="Path to input TSV file")
    args = parser.parse_args()

    # Read input
    df = pd.read_csv(args.tsv_path, sep="\t", dtype=str)
    # Preserve numeric types
    df["circuit_start_ts"] = df["circuit_start_ts"].astype(int)
    df["circuit_end_ts"] = df["circuit_end_ts"].astype(int)
    df["circuit_duration_ns"] = df["circuit_duration_ns"].astype(int)
    df["group_first_kernel_bytes"] = df["group_first_kernel_bytes"].astype(int)
    df["group_last_kernel_bytes"] = df["group_last_kernel_bytes"].astype(int)
    df["iteration"] = df["iteration"].astype(int)

    out_rows = []
    i = 0
    n = len(df)
    while i < n:
        row = df.iloc[i]
        ptype = row["parallelism_type"]
        if ptype == "PP":
            # Parse circuit_ranks list
            ranks = ast.literal_eval(row["circuit_ranks"])
            assert len(ranks) == 4, f"PP entry at index {i} has wrong rank count"
            # Double the bytes for PP
            first_b = row["group_first_kernel_bytes"] * 2
            last_b = row["group_last_kernel_bytes"] * 2
            out_rows.append({
                "iteration": row["iteration"],
                "parallelism_type": "PP",
                "circuit_start_ts": row["circuit_start_ts"],
                "circuit_end_ts": row["circuit_end_ts"],
                "circuit_duration_ns": row["circuit_end_ts"] - row["circuit_start_ts"],
                "circuit_ranks": json.dumps(ranks),
                "group_first_kernel_bytes": first_b,
                "group_last_kernel_bytes": last_b
            })
            i += 1
        else:
            # DP run: collect consecutive DP entries
            j = i
            dp_group = []
            while j < n and df.iloc[j]["parallelism_type"] == "DP":
                dp_group.append(df.iloc[j])
                j += 1
            # At least one DP
            assert dp_group, f"No DP entries at index {i}"
            # Condense group
            start_ts = min(int(r["circuit_start_ts"]) for r in dp_group)
            end_ts = max(int(r["circuit_end_ts"]) for r in dp_group)
            duration = end_ts - start_ts
            # concat ranks
            all_ranks = []
            for r in dp_group:
                ranks = ast.literal_eval(r["circuit_ranks"])
                all_ranks.extend(ranks)
            # sum bytes
            first_b = sum(int(r["group_first_kernel_bytes"]) for r in dp_group)
            last_b = sum(int(r["group_last_kernel_bytes"]) for r in dp_group)
            out_rows.append({
                "iteration": int(dp_group[0]["iteration"]),
                "parallelism_type": "DP",
                "circuit_start_ts": start_ts,
                "circuit_end_ts": end_ts,
                "circuit_duration_ns": duration,
                "circuit_ranks": json.dumps(all_ranks),
                "group_first_kernel_bytes": first_b,
                "group_last_kernel_bytes": last_b
            })
            i = j

    # Write output
    out_df = pd.DataFrame(out_rows)
    root, ext = os.path.splitext(args.tsv_path)
    out_path = f"{root}_condensed{ext}"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Condensed circuit written to {out_path}")

if __name__ == "__main__":
    main()
