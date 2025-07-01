#!/bin/bash

for x in {0..15}; do
  echo "Synch rank$x..."
  python3 synchronize_start_ts.py "trace-events-global-wind/rank${x}_dp_2_tp_4_pp_2_events_processed.tsv"
done
