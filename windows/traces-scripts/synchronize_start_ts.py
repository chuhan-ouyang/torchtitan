import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Synchronize timestamps per iteration based on the first SendRecv offset"
    )
    parser.add_argument(
        "tsv_path", type=str,
        help="Path to input TSV file with iteration, kernel_name, start_ts, end_ts columns"
    )
    args = parser.parse_args()

    input_path = args.tsv_path
    root, ext = os.path.splitext(input_path)
    output_path = f"{root}_synch{ext}"

    df = pd.read_csv(input_path, sep="\t")

    # Define the target kernel name for synch
    target_name = 'ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)'

    # Compute offset (first end_ts of the target kernel) per iteration
    offset_series = (
        df[df['kernel_name'] == target_name]
          .groupby('iteration')['end_ts']
          .first()
    )
    offsets = offset_series.to_dict()

    print("Offsets per iteration (end_ts of first SendRecv):")
    for iteration, offset in offsets.items():
        print(f"Iteration {iteration}: offset end_ts = {offset}")

    # Subtract offset from start_ts and end_ts for each row by iteration
    df['start_ts'] = df['start_ts'] - df['iteration'].map(offsets)
    df['end_ts']   = df['end_ts']   - df['iteration'].map(offsets)

    # Write synchronized data
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Synchronized data written to {output_path}")

if __name__ == "__main__":
    main()
