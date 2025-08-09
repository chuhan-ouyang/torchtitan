#!/usr/bin/env python3
"""
label_parallelism.py

Read an NCCL trace CSV and label each event with data or pipeline parallelism.
"""
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Label NCCL events as data (DP) or pipeline (PP) parallelism.")
    parser.add_argument(
        'input_csv',
        help='Path to the input CSV file containing NCCL events')
    args = parser.parse_args()

    input_path = args.input_csv
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_parallelism_labeled{ext}"

    # Read the CSV
    df = pd.read_csv(input_path)

    # Define labeling function
    def label_parallelism(name: str) -> str:
        if name in ('NCCL:ncclAllGather', 'NCCL:ncclReduceScatter'):
            return 'DP'
        if name in ('NCCL:ncclSend', 'NCCL:ncclRecv'):
            return 'PP'
        return ''

    # Apply labeling
    df['Parallelism'] = df['Name'].map(label_parallelism)

    # Save labeled CSV
    df.to_csv(output_path, index=False)
    print(f"Labeled events written to {output_path}")

if __name__ == '__main__':
    main()
