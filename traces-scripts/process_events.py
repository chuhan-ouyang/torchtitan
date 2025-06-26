import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Process event TSV to compute bytes column.")
    parser.add_argument("tsvpath", type=str, help="Path to input TSV file")
    args = parser.parse_args()

    input_path = args.tsvpath
    output_path = input_path.replace(".tsv", "_processed.tsv")

    df = pd.read_csv(input_path, sep="\t")

    # Map dtype strings to bytes per element
    dtype_to_bytes = {
        "Float": 4,
        "Float32": 4,
        "BFloat16": 2,
        "Float16": 2,
        "Half": 2,
        "Float64": 8,
        "Int32": 4,
        "Int64": 8,
        "Long": 8,
        "Int8": 1,
        "UInt8": 1,
    }

    def compute_bytes(row):
        if pd.isna(row["in_msg_nelems"]):
            num_elements = 8388608  # default from NCCL log for send/recv
        else:
            num_elements = float(row["in_msg_nelems"])

        dtype = str(row.get("dtype", "")).strip()
        bytes_per_elem = dtype_to_bytes.get(dtype, 4)  # fallback to 4 bytes
        return int(num_elements * bytes_per_elem)

    df["bytes"] = df.apply(compute_bytes, axis=1)

    df.to_csv(output_path, sep="\t", index=False)
    print(f"Processed file written to {output_path}")

if __name__ == "__main__":
    main()
