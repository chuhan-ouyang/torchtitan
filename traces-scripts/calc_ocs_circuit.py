#!/usr/bin/env python3
"""
Compute global configuration windows (circuits) across 4 ranks.
"""
import argparse
import pandas as pd
import os
import json


def main():
    parser = argparse.ArgumentParser(
        description="Compute global OCS circuit windows for 4 pipeline ranks."
    )
    parser.add_argument(
        "ranks", nargs=4, type=int,
        help="Four ranks to process (e.g. 0 1 2 3)"
    )
    args = parser.parse_args()
    ranks = args.ranks

    # Directory containing per-rank TSV files
    base_dir = "trace-events-global-wind"

    # Read each rank's grouped TSV
    dfs = {}
    for r in ranks:
        path = os.path.join(
            base_dir,
            f"rank{r}_dp_2_tp_4_pp_2_events_processed_synch_grouped.tsv"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        dfs[r] = pd.read_csv(path, sep="\t")

    # 1) Global PP windows across all 4 ranks
    pp_dfs = [dfs[r][dfs[r]["parallelism_type"] == "PP"] for r in ranks]
    for r, df in zip(ranks, pp_dfs):
        assert len(df) == 40, (
            f"Rank {r}: expected 40 PP rows, got {len(df)}"
        )

    pp_circuits = []
    for idx in range(40):
        rows = [df.iloc[idx] for df in pp_dfs]
        iterations = [int(row["iteration"]) for row in rows]
        assert len(set(iterations)) == 1, (
            f"Mismatch iteration in PP index {idx}: {iterations}"
        )
        iteration = iterations[0]
        # All ranks must agree on group bytes
        first_bytes = [int(row["group_first_kernel_bytes"]) for row in rows]
        last_bytes  = [int(row["group_last_kernel_bytes"])  for row in rows]
        assert len(set(first_bytes)) == 1, f"Mismatch first bytes at idx {idx}: {first_bytes}"
        assert len(set(last_bytes))  == 1, f"Mismatch last bytes at idx {idx}: {last_bytes}"

        # Circuit timing: max of all start_ts and end_ts
        start_ts = max(int(row["start_ts"]) for row in rows)
        end_ts   = max(int(row["end_ts"])   for row in rows)
        duration = end_ts - start_ts
        pp_circuits.append({
            "iteration": iteration,
            "parallelism_type": "PP",
            "circuit_start_ts": start_ts,
            "circuit_end_ts": end_ts,
            "circuit_duration_ns": duration,
            "circuit_ranks": json.dumps(ranks),
            "group_first_kernel_bytes": first_bytes[0],
            "group_last_kernel_bytes": last_bytes[0]
        })

    # 2) DP windows for each pipeline stage (first two vs last two ranks)
    dp_circuits = []
    for stage, group in [(1, ranks[:2]), (2, ranks[2:])]:
        dp_dfs = [
            dfs[r][dfs[r]["parallelism_type"] == "DP"]
            for r in group
        ]
        count = len(dp_dfs[0])
        assert all(len(df) == count for df in dp_dfs), (
            f"DP row count mismatch in stage {stage} for ranks {group}"
        )
        for idx in range(count):
            rows = [df.iloc[idx] for df in dp_dfs]
            iterations = [int(row["iteration"]) for row in rows]
            assert len(set(iterations)) == 1, (
                f"Mismatch iteration in DP idx {idx} stage {stage}: {iterations}"
            )
            iteration = iterations[0]
            first_bytes = [int(row["group_first_kernel_bytes"]) for row in rows]
            last_bytes  = [int(row["group_last_kernel_bytes"])  for row in rows]
            assert len(set(first_bytes)) == 1, f"Mismatch first bytes DP at idx {idx}: {first_bytes}"
            assert len(set(last_bytes))  == 1, f"Mismatch last bytes DP at idx {idx}: {last_bytes}"
            start_ts = max(int(row["start_ts"]) for row in rows)
            end_ts   = max(int(row["end_ts"])   for row in rows)
            duration = end_ts - start_ts
            dp_circuits.append({
                "iteration": iteration,
                "parallelism_type": "DP",
                "circuit_start_ts": start_ts,
                "circuit_end_ts": end_ts,
                "circuit_duration_ns": duration,
                "circuit_ranks": json.dumps(group),
                "group_first_kernel_bytes": first_bytes[0],
                "group_last_kernel_bytes": last_bytes[0]
            })

    # Combine and sort by circuit_end_ts
    all_circuits = pd.DataFrame(pp_circuits + dp_circuits)
    all_circuits = all_circuits.sort_values("circuit_end_ts").reset_index(drop=True)

    # Output file path
    out_name = f"rank_{ranks[0]}_{ranks[1]}_{ranks[2]}_{ranks[3]}_circuit.tsv"
    out_path = os.path.join(base_dir, out_name)
    all_circuits.to_csv(out_path, sep="\t", index=False)
    print(f"Global circuit windows written to {out_path}")


if __name__ == "__main__":
    main()
