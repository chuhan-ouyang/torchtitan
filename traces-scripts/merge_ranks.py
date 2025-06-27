import argparse
import os
import pandas as pd
import re

def main():
    parser = argparse.ArgumentParser(description="Merge rank window TSV files into one.")
    parser.add_argument("input_dir", type=str, help="Directory containing rank window TSV files")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = os.path.join(input_dir, "all_ranks_dp_2_tp_4_pp_2_events_processed_window.tsv")

    # Regex pattern to extract rank
    pattern = re.compile(r"rank(\d+)_dp_2_tp_4_pp_2_events_processed_window\.tsv")

    # Collect valid files and sort by rank
    matched_files = []
    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if match:
            rank = int(match.group(1))
            matched_files.append((rank, fname))

    matched_files.sort(key=lambda x: x[0])  # Sort by rank

    merged_dfs = []
    for rank, fname in matched_files:
        fpath = os.path.join(input_dir, fname)
        df = pd.read_csv(fpath, sep="\t")
        df.insert(0, "rank", rank)
        merged_dfs.append(df)

    if not merged_dfs:
        print("No matching files found.")
        return

    combined_df = pd.concat(merged_dfs, ignore_index=True)
    combined_df.to_csv(output_file, sep="\t", index=False)
    print(f"Merged file written to {output_file}")

if __name__ == "__main__":
    main()
