#!/usr/bin/env python3
"""
plot_per_rank_wnd.py

Reads a CSV with a 'wind_duration_ns' column and plots the CDF of window sizes in milliseconds.
Only a single input CSV path is required.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot CDF of window sizes (converted from ns to ms)"
    )
    parser.add_argument(
        "input_csv",
        help="Path to input CSV containing a 'wind_duration_ns' column"
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv)

    if 'wind_duration_ns' not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'wind_duration_ns' column"
        )

    # Convert to milliseconds
    durations_ms = df['wind_duration_ns'] / 1e6

    # Compute and print quantiles
    q_levels = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    quantiles = np.quantile(durations_ms, q_levels)
    print("Quantiles of window durations (ms):")
    for q, val in zip(q_levels, quantiles):
        label = f"{int(q*100)}th percentile" if q != 1.0 else "100th percentile"
        print(f"{label}: {val:.3f} ms")


    # Compute CDF
    sorted_vals = np.sort(durations_ms.values)
    cdf_vals = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.xscale("log")
    plt.xlabel("Window size (ms)")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.tight_layout()
    plt.title('CDF of Window Sizes for Rank0')

    plt.plot(sorted_vals, cdf_vals, label="rank0")
    plt.savefig('rank0_windows_cdf.png')


if __name__ == "__main__":
    main()
