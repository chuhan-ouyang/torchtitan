#!/bin/bash

for i in {0..15}; do
  input_path="trace-events/rank${i}_dp_2_tp_4_pp_2_events_processed.tsv"
  if [[ -f "$input_path" ]]; then
    echo "Processing $input_path"
    python3 calc_reconfig_wind.py "$input_path"
  else
    echo "File not found: $input_path"
  fi
done
