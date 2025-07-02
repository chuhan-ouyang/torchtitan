Traces Processing Pipeline:

Perfetto SQL Engine: 
    run get_pp_dp_events.sql for each rank and copy result to tsv

process_events.py: 
    write to **_processed.tsv
    calculate size in bytes for all kernels based on dtype

###
Reconfig Window for Each Rank
###
calc_reconfig_wind.py : 
    write to **_window.tsv
    calculate window across all iterations

merge_ranks.py:
    combine all ranks into a tsv

plot_cdf.py:
    cfreate **_plot.png
    plot cdf of window size, a curve for eac rank

###
Reconfig Window for All Participating Rank
###
synchronize??.py:
    input: _processed.tsv
    for each iteration, use first S/R kernel's end_ts as time=0ns
    also process data by grouping all consecutive kernels of a parallelism type into one

group_kernel.py:
    input: _processed_synch.tsv
    group all DP calls into one

calc_oc_circuit.py:
    4 groups of 4 ranks
    compute ocs circuit