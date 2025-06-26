import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Plot CDF of window durations from TSV.")
    parser.add_argument("tsv_path", type=str, help="Path to input TSV file")
    args = parser.parse_args()

    input_path = args.tsv_path
    df = pd.read_csv(input_path, sep="\t")

    # Convert nanoseconds to milliseconds
    df["duration_ms"] = df["wind_duration_ns"] / 1e6

    # Compute CDF
    sorted_durations = np.sort(df["duration_ms"].values)
    cdf = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_durations, cdf)
    plt.xlabel("Window Duration (ms)")
    plt.ylabel("CDF")
    plt.title("CDF of Parallelism Switch Window Durations")
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    output_png = input_path.replace(".tsv", "_cdf.png")
    plt.savefig(output_png)
    print(f"CDF plot saved to {output_png}")

if __name__ == "__main__":
    main()
