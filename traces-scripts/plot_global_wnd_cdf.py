#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot CDF of window durations (ms) from four TSVs on one graph."
    )
    parser.add_argument(
        "tsv_paths",
        nargs=4,
        metavar="TSV",
        help="Paths to 4 input TSV files (rails 1–4), in order."
    )
    args = parser.parse_args()
    quantile_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    plt.figure(figsize=(4, 3))
    plt.rcParams.update({'font.size': 14})

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Plot each rail's CDF
    for idx, path in enumerate(args.tsv_paths, start=1):
        df = pd.read_csv(path, sep="\t")
        df["wind_duration_ms"] = df["wind_duration_ns"] / 1e6
        durations = np.sort(df["wind_duration_ms"].values)
        cdf = np.arange(1, len(durations) + 1) / len(durations)
        plt.plot(durations, cdf, label=f"rail{idx}")

        qs = np.quantile(durations, quantile_levels)
        print(f"rail{idx} quantiles (ms):")
        for q_level, q_val in zip(quantile_levels, qs):
            pct = int(q_level * 100)
            print(f"  {pct:>3}%: {q_val:.2f}")
        print()

    plt.xscale("log")
    plt.xlabel("Window size (ms)")
    plt.ylabel("CDF")
    # plt.title("CDF of Reconfiguration Window Durations for Rails 1–4")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save to 'rails_cdf.png' in the same directory as the first TSV
    out_dir = os.path.dirname(args.tsv_paths[0]) or "."
    out_path = os.path.join(out_dir, "rails_cdf_new.pdf")
    plt.savefig(out_path)
    print(f"CDF plot saved to {out_path}")

if __name__ == "__main__":
    main()
