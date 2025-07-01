#!/bin/bash

for x in {0..15}; do
  echo "Group rank$x..."
  python3 group_kernel.py "trace-events-global-wind/rank${x}_dp_2_tp_4_pp_2_events_processed_synch.tsv"
done
