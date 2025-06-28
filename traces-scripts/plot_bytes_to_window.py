import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot average window duration per kernel_after_bytes with error bars.")
    parser.add_argument("tsv_path", type=str, help="Path to input TSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.tsv_path, sep="\t")

    df["wind_duration_ms"] = df["wind_duration_ns"] / 1e6

    grouped = df.groupby("kernel_after_bytes")["wind_duration_ms"]
    averages = grouped.mean()
    std_devs = grouped.std()
    counts = grouped.count()

    # Print stats
    print("kernel_after_bytes\tCount\tAvg(ms)\tStd(ms)")
    for k in averages.index:
        print(f"{int(k)}\t{counts[k]}\t{averages[k]:.2f}\t{std_devs[k]:.2f}")

    plt.figure(figsize=(10, 6))
    plt.bar(averages.index.astype(str), averages.values, yerr=std_devs.values, capsize=5)
    plt.xlabel("Kernel After Window Communication Size (Bytes)")
    plt.ylabel("Average Window Duration (ms)")
    plt.title("Average Window Duration per Kernel Communication Size")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = args.tsv_path.replace(".tsv", "_avg_window_duration.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
