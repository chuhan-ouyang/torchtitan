Traces Processing Pipeline:

Perfetto SQL Engine: run get_pp_dp_events.sql for each rank and copy result to csv

process_events.py: write to **_processed.csv
    calculate size

calculate_reconfig_window.py: write to rank_X_window.csv
    calculate window