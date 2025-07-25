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

   # find the very first SendRecv in the lowest iteration
    sendrecv = df[df["kernel_name"] == target_name]
    if sendrecv.empty:
        raise RuntimeError(f"No rows found with kernel_name == {target_name}")

    # get the row with minimal iteration, then minimal end_ts
    first = sendrecv.sort_values(["iteration","end_ts"]).iloc[0]
    base_iter = first["iteration"]
    base_offset = first["end_ts"]
    print(f"Using iteration {base_iter}'s first SendRecv end_ts as zero: {base_offset}")

    # shift everything by that one offset
    df["start_ts"] = df["start_ts"] - base_offset
    df["end_ts"]   = df["end_ts"]   - base_offset

    # write out
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Synchronized data written to {output_path}")


if __name__ == "__main__":
    main()
