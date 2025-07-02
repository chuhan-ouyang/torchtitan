#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Plot CDF of window durations (ms) from a TSV.")
    parser.add_argument("tsv_path", help="Path to input TSV file")
    args = parser.parse_args()

    # Load and compute durations in ms
    df = pd.read_csv(args.tsv_path, sep="\t")
    df["wind_duration_ms"] = df["wind_duration_ns"] / 1e6

    # Build CDF
    durations = np.sort(df["wind_duration_ms"].values)
    cdf = np.arange(1, len(durations) + 1) / len(durations)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(durations, cdf)
    plt.xlabel("Window Duration (ms)")
    plt.ylabel("CDF")
    plt.title("CDF of Window Durations")
    plt.grid(True)
    plt.tight_layout()

    # Save under same base name
    base, _ = os.path.splitext(args.tsv_path)
    out_png = f"{base}_cdf.png"
    plt.savefig(out_png)
    print(f"CDF plot saved to {out_png}")

if __name__ == "__main__":
    main()
