#!/bin/bash

# for x in {0..15}; do
#   echo "Synch rank$x..."
#   python3 synchronize_start_ts_all_itr.py "trace-events-global-wind/rank${x}_dp_2_tp_4_pp_2_events_processed.tsv"
# done

#####
# PP Syncs
#####
python3 synch_pp_per_itr.py trace-events-global-wind/rank0_dp_2_tp_4_pp_2_events_processed.tsv trace-events-global-wind/rank8_dp_2_tp_4_pp_2_events_processed.tsv
python3 synch_pp_per_itr.py trace-events-global-wind/rank4_dp_2_tp_4_pp_2_events_processed.tsv trace-events-global-wind/rank12_dp_2_tp_4_pp_2_events_processed.tsv

#####
# DP Syncs
#####
python3 synch_dp_per_itr.py trace-events-global-wind/rank0_dp_2_tp_4_pp_2_events_processed_ppsync.tsv trace-events-global-wind/rank4_dp_2_tp_4_pp_2_events_processed_ppsync.tsv
python3 synch_dp_per_itr.py trace-events-global-wind/rank8_dp_2_tp_4_pp_2_events_processed_ppsync.tsv trace-events-global-wind/rank12_dp_2_tp_4_pp_2_events_processed_ppsync.tsv
