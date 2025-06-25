import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(description="Process event CSV to compute bytes column.")
    parser.add_argument("csvpath", type=str, help="Path to input CSV file")
    args = parser.parse_args()

    input_path = args.csvpath
    output_path = input_path.replace(".csv", "_processed.csv")

    df = pd.read_csv(input_path)

    def compute_bytes(row):
        try:
            val = float(row["in_msg_nelems"])
            return int(val * 4)
        except (ValueError, TypeError):
            return 8388608 * 4

    df["bytes"] = df.apply(compute_bytes, axis=1)

    df.to_csv(output_path, index=False)
    print(f"Processed file written to {output_path}")

if __name__ == "__main__":
    main()
