#!/usr/bin/env python3
import argparse
import pandas as pd
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Clean NCCL trace: remove group-encapsulated AllGather/ReduceScatter events")
    p.add_argument('input', help='Input CSV file with trace rows')
    p.add_argument('-o', '--output', help='Output CSV file', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # find all group start and end indices
    starts = df.index[df['Name'] == 'NCCL:ncclGroupStart'].tolist()
    ends   = df.index[df['Name'] == 'NCCL:ncclGroupEnd'].tolist()
    # pair each start with the first end after it
    pairs = []
    for s in starts:
        # find first end > s
        e_candidates = [e for e in ends if e > s]
        if not e_candidates:
            print(f"Warning: no matching NCCL:ncclGroupEnd for start at row {s}", file=sys.stderr)
            continue
        pairs.append((s, e_candidates[0]))

    out_rows = []
    for idx, row in df.iterrows():
        name = row['Name']
        # always keep sends and recvs
        if name in ('NCCL:ncclSend', 'NCCL:ncclRecv'):
            out_rows.append(row)
            continue
        # conditional for allgather and reducescatter
        if name in ('NCCL:ncclAllGather', 'NCCL:ncclReduceScatter'):
            # find enclosing ranges
            encl = [(s, e) for s, e in pairs if s < idx < e]
            if len(encl) > 1:
                raise AssertionError(
                    f"{name} at row {idx} is enclosed by multiple ranges: {encl}")
            if len(encl) == 0:
                # ungrouped, keep
                out_rows.append(row)
            else:
                # inside a group range, drop and report
                s, e = encl[0]
                print(f"Dropping {name} at row {idx}, enclosed by groupStart at {s} -> groupEnd at {e}")
            continue
        # drop all others
        # e.g., groupStart/End and other events
        continue

    # assemble output
    cleaned = pd.DataFrame(out_rows)
    out_path = args.output or args.input.replace('.csv', '_cleaned.csv')
    cleaned.to_csv(out_path, index=False)
    print(f"Wrote {len(cleaned)} rows to {out_path}")

if __name__ == '__main__':
    main()
