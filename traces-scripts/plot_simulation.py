#!/usr/bin/env python3
import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot TorchTitan Application Iteration Time vs Network Reconfiguration Delay"
    )
    parser.add_argument("csv_path", help="Path to CSV (ocs_reconfig_lat_ms,hide,avg_iteration_time_ms,std_dev_ms)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df["hide"] = df["hide"].astype(int)

    # Unique, sorted delays
    delays = sorted(df["ocs_reconfig_lat_ms"].unique())

    # Split baseline (hide=0) and masked (hide=1)
    base = df[df["hide"] == 0].set_index("ocs_reconfig_lat_ms").sort_index()
    mask = df[df["hide"] == 1].set_index("ocs_reconfig_lat_ms").sort_index()

    # Extract mean & std in order
    base_means   = [base.loc[d, "avg_iteration_time_ms"] for d in delays]
    base_stds    = [base.loc[d, "std_dev_ms"]          for d in delays]
    mask_means   = [mask.loc[d, "avg_iteration_time_ms"] for d in delays]
    mask_stds    = [mask.loc[d, "std_dev_ms"]          for d in delays]

    # X positions
    x = np.arange(len(delays))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width/2,
        base_means,
        width,
        yerr=base_stds,
        capsize=5,
        label="Baseline",
    )
    ax.bar(
        x + width/2,
        mask_means,
        width,
        yerr=mask_stds,
        capsize=5,
        label="Masked Network Reconfiguration",
    )

    ax.set_xlabel("Network Reconfiguration Delay (ms)")
    ax.set_yscale('log')
    ax.set_ylabel("Average Iteration Time (ms)")
    ax.set_title("TorchTitan Application Iteration Time")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in delays], rotation=45)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    # Save next to input CSV
    out_dir = os.path.dirname(args.csv_path) or "."
    out_path = os.path.join(out_dir, "iteration_time_vs_reconfig_delay.png")
    fig.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
