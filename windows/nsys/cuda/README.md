CUDA GPU Trace Processing

Raw nsys rep files in :/pscratch/sd/c/co232/nsys_tp_4_dp_2_shard_pp_2_10itrs_nccl2.27_nsys2025-3

1. separate_ranks.py/separate_ranks.sh
Separate based on CUDA device
2. filter_nccl.sh/filter_nccl.py
Only keep NCCL events
3. label_parallelism.sh/label_parallelism.py
Reference nvtx trace's TP/DP labeling (based on presence of groupStart/End wrapping), label the cuda GPU trace assuming the same parallelism orders
4. remove_tp_na.sh/remove_tp_na.py: remove TP, NA
5. calc_wind.sh/calc_wind.py:Calculate window sizes