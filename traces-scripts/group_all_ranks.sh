#!/bin/bash

# for x in {0..15}; do
#   echo "Group rank$x..."
#   python3 group_kernel.py "trace-events-global-wind/rank${x}_dp_2_tp_4_pp_2_events_processed_synch.tsv"
# done


# python3 group_kernel.py "trace-events-global-wind/rank1_dp_2_tp_4_pp_2_events_processed_synch.tsv"
# python3 group_kernel.py "trace-events-global-wind/rank5_dp_2_tp_4_pp_2_events_processed_synch.tsv"
# python3 group_kernel.py "trace-events-global-wind/rank9_dp_2_tp_4_pp_2_events_processed_synch.tsv"
# python3 group_kernel.py "trace-events-global-wind/rank13_dp_2_tp_4_pp_2_events_processed_synch.tsv"

python3 group_kernel.py "trace-events-global-wind/rank2_dp_2_tp_4_pp_2_events_processed_synch.tsv"
python3 group_kernel.py "trace-events-global-wind/rank6_dp_2_tp_4_pp_2_events_processed_synch.tsv"
python3 group_kernel.py "trace-events-global-wind/rank10_dp_2_tp_4_pp_2_events_processed_synch.tsv"
python3 group_kernel.py "trace-events-global-wind/rank14_dp_2_tp_4_pp_2_events_processed_synch.tsv"

python3 group_kernel.py "trace-events-global-wind/rank3_dp_2_tp_4_pp_2_events_processed_synch.tsv"
python3 group_kernel.py "trace-events-global-wind/rank7_dp_2_tp_4_pp_2_events_processed_synch.tsv"
python3 group_kernel.py "trace-events-global-wind/rank11_dp_2_tp_4_pp_2_events_processed_synch.tsv"
python3 group_kernel.py "trace-events-global-wind/rank15_dp_2_tp_4_pp_2_events_processed_synch.tsv"
