#!/usr/bin/env python3
import argparse
import sys
import os

import pandas as pd
import numpy as np

def simulate(df_bounds, df_windows, lat_ms, hide_latency):
    total_times = []
    for itr in range(10, 20):
        # bounds for this iteration
        b = df_bounds[df_bounds["iteration"] == itr]
        if len(b) != 1:
            sys.exit(f"ERROR: expected one row for iteration {itr} in bounds, got {len(b)}")
        t0 = b["start_ts"].iat[0] / 1e6
        t1 = b["end_ts"].iat[0]   / 1e6

        # window durations
        wds = df_windows[df_windows["iteration"] == itr]["wind_duration_ns"] / 1e6
        sum_win = wds.sum()

        comm = (t1 - t0) - sum_win
        if hide_latency == 0:
            reconfig = (wds + lat_ms).sum()
        else:
            reconfig = np.maximum(wds, lat_ms).sum()

        total_times.append(comm + reconfig)
        print(f" Iter {itr}: t1-t0={(t1 - t0):.3f} ms, comm={comm:.3f} ms, reconfig={reconfig:.3f} ms, total={(comm + reconfig):.3f} ms")

    avg = np.mean(total_times)
    std = np.std(total_times, ddof=0)
    # Print results
    print(f"OCS reconfiguration latency (ms): {lat_ms}")
    print(f"Hide latency flag: {hide_latency}")
    print(f"Average total time: {avg:.3f} ms")
    print(f"Std dev of total time: {std:.3f} ms")
    return avg, std

def main():
    parser = argparse.ArgumentParser(
        description="Sweep OCS reconfig latency & hide‐flag, report avg/std of total iteration time"
    )
    parser.add_argument("iteration_bound_tsv",
                        help="TSV with [iteration,start_ts,end_ts] in ns")
    parser.add_argument("windows_tsv",
                        help="TSV with [iteration,wind_duration_ns] in ns")
    args = parser.parse_args()

    # load once
    df_bounds  = pd.read_csv(args.iteration_bound_tsv, sep="\t")
    df_windows = pd.read_csv(args.windows_tsv,     sep="\t")

    # sanity check iterations
    expected = list(range(10, 20))
    if sorted(df_bounds["iteration"].unique()) != expected:
        sys.exit("ERROR: bounds missing iterations 10–19")
    if sorted(df_windows["iteration"].unique()) != expected:
        sys.exit("ERROR: windows missing iterations 10–19")

    # sweep configurations
    latencies = [0, 0.1, 1, 5, 10, 20, 50, 100, 200, 500, 1000] #ms
    results = []
    for lat in latencies:
        for hide in (0, 1):
            avg, std = simulate(df_bounds, df_windows, lat, hide)
            results.append({
                "ocs_reconfig_lat_ms": lat,
                "hide": hide,
                "avg_iteration_time_ms": avg,
                "std_dev_ms": std
            })

    # write results
    out_path = "simulation_ocs_reconfig.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Sweep complete, results written to {out_path}")

if __name__ == "__main__":
    main()
