#!/bin/bash

for x in {0..15}; do
  echo "Processing rank$x..."
  python3 process_events.py "trace-events/rank${x}_dp_2_tp_4_pp_2_events.tsv"
done
