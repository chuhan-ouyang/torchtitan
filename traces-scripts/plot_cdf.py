import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

def main():
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    parser = argparse.ArgumentParser(description="Plot CDF of window durations from TSV.")
    parser.add_argument("tsv_path", type=str, help="Path to input TSV file")
    args = parser.parse_args()

    input_path = args.tsv_path
    df = pd.read_csv(input_path, sep="\t")

    if "rank" not in df.columns:
        raise ValueError("TSV file must contain a 'rank' column.")

    df["duration_ms"] = df["wind_duration_ns"] / 1e6

    plt.figure(figsize=(10, 6))

    ranks = sorted(df["rank"].unique())
    cmap = cm.get_cmap("tab20", len(ranks))

    for idx, rank in enumerate(ranks):
        rank_df = df[df["rank"] == rank]
        durations = np.sort(rank_df["duration_ms"].values)
        cdf = np.arange(1, len(durations) + 1) / len(durations)

        linestyle = '-' if rank < 8 else '--'
        plt.plot(durations, cdf, label=f"Rank {rank}", color=cmap(idx), linestyle=linestyle)

    plt.xlabel("Window Duration (ms)")
    plt.ylabel("CDF")
    plt.title("CDF of Parallelism Switch Window Durations")
    plt.grid(True)

    # Place legend outside the plot on the right
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Ranks")
    plt.tight_layout()

    output_png = input_path.replace(".tsv", "_cdf_new.png")
    plt.savefig(output_png, bbox_inches='tight')
    print(f"CDF plot saved to {output_png}")

if __name__ == "__main__":
    main()
