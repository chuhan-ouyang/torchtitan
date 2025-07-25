###
Simulation
###
simulate_ocs_reconfig.py

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
process_events 

synchronize??.py:
    input: _processed.tsv
    synch 4 to 0
    synch 8 to 0
    synch 12 to 4 and 8

group_kernel.py: - for DP also sum all the bytes of the kernel
    input: _processed_synch.tsv
    group all DP calls into one

calc_oc_circuit.py:
    4 groups of 4 ranks
    compute ocs circuit

condense_circuit.py 
    sum size up for consecutive DP circuits
    double size for PP circuits with 4 ranks

calc wind

plot cdf

plot bytes to window