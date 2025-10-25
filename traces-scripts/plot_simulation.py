#!/usr/bin/env python3
import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def main():
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
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
    base_means   = [base.loc[d, "avg_iteration_time_ms"] / 1000 for d in delays]
    base_stds    = [base.loc[d, "std_dev_ms"] / 1000         for d in delays]
    mask_means   = [mask.loc[d, "avg_iteration_time_ms"] / 1000 for d in delays]
    mask_stds    = [mask.loc[d, "std_dev_ms"]  / 1000       for d in delays]

    # Normalize values by the first baseline mean
    norm_factor = base_means[0]
    base_means = [m / norm_factor for m in base_means]
    base_stds = [s / norm_factor for s in base_stds]
    mask_means = [m / norm_factor for m in mask_means]
    mask_stds = [s / norm_factor for s in mask_stds]

    # X positions
    x = np.arange(len(delays))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(
        x - width/2,
        base_means,
        width,
        label="Without provisioning",
        color=plt.cm.tab10(0),  # Native blue
    )
    ax.bar(
        x + width/2,
        mask_means,
        width,
        label="With provisioning",
        color=plt.cm.tab10(1),  # Native orange
    )
    # Add rotated text annotations on the bars
    for i, ratio in enumerate(base_means):
        ax.text(
            x[i] - width/2, base_means[i] + base_stds[i] + 0.01,
            f"{base_means[i]:.2f}", ha="center", va="bottom", fontsize=10, color=plt.cm.tab10(0), rotation=45
        )
    for i, ratio in enumerate(mask_means):
        ax.text(
            x[i] + width/2, mask_means[i] + mask_stds[i] + 0.01,
            f"{mask_means[i]:.2f}", ha="center", va="bottom", fontsize=10, color=plt.cm.tab10(1), rotation=45
        )

    ax.set_xlabel("Reconfig. latency (ms)")
    ax.set_xticks(np.arange(len(delays)))
    ax.set_xticklabels(
        ["0", "0.1", "1.0", "5.0", "10.0", "20.0", "50.0", "100", "200", "500", "1000"],
        fontsize=12
    )
    ax.set_ylabel("Normalized iter. time")
    ax.set_ylim(0.9, 1.8)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.set_size_inches(7, 3)  # Increase the width of the figure
    fig.tight_layout()

    # Save next to input CSV
    out_dir = os.path.dirname(args.csv_path) or "."
    out_path = os.path.join(out_dir, "iteration_time_vs_reconfig_delay_normalized.pdf")
    fig.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
