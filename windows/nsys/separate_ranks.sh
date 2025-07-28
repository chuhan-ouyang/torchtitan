#!/bin/bash
python3 separate_ranks.py traces/node0_nvtx_pushpop_trace.csv
python3 separate_ranks.py traces/node2_nvtx_pushpop_trace.csv