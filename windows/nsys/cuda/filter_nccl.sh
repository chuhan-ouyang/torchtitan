#!/bin/bash
python3 filter_nccl.py traces/node0_cuda_gpu_trace_local_rank_0.csv
# python3 separate_ranks.py traces/node2_cuda_gpu_trace.csv
# python3 separate_ranks.py traces/node2_nvtx_pushpop_trace.csv