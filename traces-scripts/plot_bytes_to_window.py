import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot window count and average duration per kernel_after_bytes.")
    parser.add_argument("tsv_path", type=str, help="Path to input TSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.tsv_path, sep="\t")
    df["wind_duration_ms"] = df["wind_duration_ns"] / 1e6

    grouped = df.groupby("kernel_after_bytes")["wind_duration_ms"]
    averages = grouped.mean()
    std_devs = grouped.std()
    counts = grouped.count()

    print("kernel_after_bytes\tCount\tAvg(ms)\tStd(ms)")
    for k in averages.index:
        print(f"{int(k)}\t{counts[k]}\t{averages[k]:.2f}\t{std_devs[k]:.2f}")

    labels = [str(int(k)) for k in averages.index]
    x = range(len(averages))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left Y-axis: count bar plot
    ax1.bar(x, counts.values, color='tab:blue', label="Count")
    ax1.set_xlabel("Kernel After Window Communication Size (Bytes)")
    ax1.set_ylabel("Count", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)

    # Right Y-axis: average window duration line plot
    ax2 = ax1.twinx()
    ax2.plot(x, averages.values, color='tab:red', marker='o', label="Avg Duration (ms)")
    ax2.set_ylabel("Average Window Duration (ms)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle("Window Count and Average Duration per Kernel Communication Size")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path = args.tsv_path.replace(".tsv", "_count_duration_plot.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
