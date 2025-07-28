Process nsys-rep traces

/pscratch/sd/c/co232/nsys_tp_4_dp_2_shard_pp_2_10itrs_nccl2.27_nsys2025-3: contains all nsys-rep, sqlite files

1. gen_nvtx_csv.sh
Generate nvtx push pop kernels from nsys-rep files

2. separate_ranks.sh/separate_ranks.py

Turn 1 node's files into 4 ranks traces
---
Node0
TID         Rank
1114945     0
1115560     0

1114946     1
1115565     1

1114947     2
1115574     2

1114948     3
1115571     3


PID         Rank
1114945        0
1114946        1
1114947        2
1114948        3

---
Comm ID Verification: Rank 0, Node 0
3,574,919,951,526,054,331     DP
18,319,673,707,052,330,093    TP
6,004,158,948,986,910,404     PP  
6,831,535,697,949,778,466     PP  


2. clean_tp_ar.py, clean_tp_ar.sh:
Remove TP entries + AR + misc
Between start and end remove all TP entries
Remove NCCL:ncclCommInitRankConfig, NCCL:ncclBroadcast, :pp_fw_bw CCCL:*, NCCL:ncclAllReduce

3. label_parallelism.sh/label_parallelism.py
Label kernel as DP or PP

5. calc_wind.sh/calc_wind.py
window_type	wind_start_ts	wind_end_ts	wind_duration_ns	kernel_before_bytes	kernel_after_bytes

6. plot_per_rank_wnd.ipynb