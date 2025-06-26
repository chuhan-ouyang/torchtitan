Traces Processing Pipeline:

Perfetto SQL Engine: run get_pp_dp_events.sql for each rank and copy result to tsv

process_events.py: write to **_processed.tsv
    calculate size in bytes for all kernels based on dtype

calc_reconfig_wind.py : write to **_window.tsv
    calculate window across all iterations