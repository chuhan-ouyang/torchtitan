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
synchronize_start_ts.py:
    for each iteration, use first S/R kernel's end_ts as time=0ns