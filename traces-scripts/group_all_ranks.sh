#!/bin/bash

# for x in {0..15}; do
#   echo "Group rank$x..."
#   python3 group_kernel.py "trace-events-global-wind/rank${x}_dp_2_tp_4_pp_2_events_processed_ppsync_dpsync.tsv"
# done


python3 group_kernel.py "trace-events-global-wind/rank0_dp_2_tp_4_pp_2_events_processed_ppsync_dpsync.tsv"
python3 group_kernel.py "trace-events-global-wind/rank4_dp_2_tp_4_pp_2_events_processed_ppsync_dpsync.tsv"
python3 group_kernel.py "trace-events-global-wind/rank8_dp_2_tp_4_pp_2_events_processed_ppsync_dpsync.tsv"
python3 group_kernel.py "trace-events-global-wind/rank12_dp_2_tp_4_pp_2_events_processed_ppsync_dpsync.tsv"
