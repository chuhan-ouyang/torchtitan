#!/bin/bash
# Rank0
python3 clean_tp_ar.py traces/node0_nvtx_pushpop_trace_pid_1114945.csv

# Rank8
python3 clean_tp_ar.py traces/node2_nvtx_pushpop_trace_pid_563290.csv