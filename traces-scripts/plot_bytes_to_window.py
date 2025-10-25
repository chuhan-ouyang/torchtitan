import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    parser = argparse.ArgumentParser(description="Plot window count and average duration per kernel_after_bytes.")
    parser.add_argument("tsv_path", type=str, help="Path to input TSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.tsv_path, sep="\t")
    df["wind_duration_ms"] = df["wind_duration_ns"] / 1e6

    grouped = df.groupby("kernel_after_bytes")["wind_duration_ms"]
    averages = grouped.mean()
    # std_devs = grouped.std()
    counts = grouped.count()
    grouped_ranges = {}
    for k in averages.index:
        range_key = int(k / 2**20)
        if range_key not in grouped_ranges:
            grouped_ranges[range_key] = {"count": 0, "sum_avg": 0}
        grouped_ranges[range_key]["count"] += counts[k]
        grouped_ranges[range_key]["sum_avg"] += counts[k] * averages[k]

    grouped_averages = {k: v["sum_avg"] / v["count"] for k, v in grouped_ranges.items()}
    grouped_counts = {k: int(v["count"] / 10 + 0.5) for k, v in grouped_ranges.items()}

    print(grouped_averages)
    print("kernel_after_bytes\tCount\tAvg(ms)\tStd(ms)")
    for k in grouped_averages.keys():
        print(f"{int(k)}\t{grouped_counts[k]}\t{grouped_averages[k]:.2f}")

    labels = [str(int(k)) for k in grouped_averages.keys()]
    labels[0] = "<1"
    x = range(len(grouped_averages))

    plt.rcParams.update({'font.size': 14})
    fig, ax1 = plt.subplots(figsize=(4, 3))

    # Left Y-axis: count bar plot
    ax1.bar(x, grouped_counts.values(), hatch='/', edgecolor='black', color='white', label="Count")
    ax1.set_xlabel("Traffic size after reconfig. (MB)")
    ax1.set_ylabel("Number / iter.")
    ax1.tick_params(axis='y')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.1, 1.23), ncol=2)

    # Right Y-axis: average window duration line plot
    ax2 = ax1.twinx()
    ax2.plot(x, grouped_averages.values(), linestyle='--', marker='o', label="Window size")
    ax2.set_yscale('log')
    ax2.set_ylim(0.005, 2000)
    ax2.set_ylabel("Avg. window size (ms)")
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.8, 1.23), ncol=2)

    # fig.suptitle("Window Count and Average Duration per Kernel Communication Size")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path = args.tsv_path.replace(".tsv", "_count_duration_plot_new.pdf")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
