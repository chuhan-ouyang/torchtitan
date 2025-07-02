#!/bin/bash
# python3 synch_rank4_dp.py trace-events-global-wind/rank0_dp_2_tp_4_pp_2_events_processed.tsv trace-events-global-wind/rank4_dp_2_tp_4_pp_2_events_processed.tsv
# python3 synch_rank8_pp.py trace-events-global-wind/rank0_dp_2_tp_4_pp_2_events_processed.tsv trace-events-global-wind/rank8_dp_2_tp_4_pp_2_events_processed.tsv
# python3 synch_rank12_dp_pp.py trace-events-global-wind/rank4_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank8_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank12_dp_2_tp_4_pp_2_events_processed.tsv

# python3 synch_rank4_dp.py trace-events-global-wind/rank1_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank5_dp_2_tp_4_pp_2_events_processed.tsv
# python3 synch_rank8_pp.py trace-events-global-wind/rank1_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank9_dp_2_tp_4_pp_2_events_processed.tsv
# python3 synch_rank12_dp_pp.py trace-events-global-wind/rank5_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank9_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank13_dp_2_tp_4_pp_2_events_processed.tsv

python3 synch_rank4_dp.py trace-events-global-wind/rank2_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank6_dp_2_tp_4_pp_2_events_processed.tsv
python3 synch_rank8_pp.py trace-events-global-wind/rank2_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank10_dp_2_tp_4_pp_2_events_processed.tsv
python3 synch_rank12_dp_pp.py trace-events-global-wind/rank6_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank10_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank14_dp_2_tp_4_pp_2_events_processed.tsv

python3 synch_rank4_dp.py trace-events-global-wind/rank3_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank7_dp_2_tp_4_pp_2_events_processed.tsv
python3 synch_rank8_pp.py trace-events-global-wind/rank3_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank11_dp_2_tp_4_pp_2_events_processed.tsv
python3 synch_rank12_dp_pp.py trace-events-global-wind/rank7_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank11_dp_2_tp_4_pp_2_events_processed_synch.tsv trace-events-global-wind/rank15_dp_2_tp_4_pp_2_events_processed.tsv

# for x in {0..15}; do
#   echo "Synch rank$x..."
#   python3 synchronize_start_ts_all_itr.py "trace-events-global-wind/rank${x}_dp_2_tp_4_pp_2_events_processed.tsv"
# done

# #####
# # PP Syncs
# #####
# python3 synch_pp_per_itr.py trace-events-global-wind/rank0_dp_2_tp_4_pp_2_events_processed.tsv trace-events-global-wind/rank8_dp_2_tp_4_pp_2_events_processed.tsv
# python3 synch_pp_per_itr.py trace-events-global-wind/rank4_dp_2_tp_4_pp_2_events_processed.tsv trace-events-global-wind/rank12_dp_2_tp_4_pp_2_events_processed.tsv

# #####
# # DP Syncs
# #####
# python3 synch_dp_per_itr.py trace-events-global-wind/rank0_dp_2_tp_4_pp_2_events_processed_ppsync.tsv trace-events-global-wind/rank4_dp_2_tp_4_pp_2_events_processed_ppsync.tsv
# python3 synch_dp_per_itr.py trace-events-global-wind/rank8_dp_2_tp_4_pp_2_events_processed_ppsync.tsv trace-events-global-wind/rank12_dp_2_tp_4_pp_2_events_processed_ppsync.tsv
