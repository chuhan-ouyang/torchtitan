-- Get time range of ProfilerStep#17
WITH step_bounds AS (
  SELECT
    ts AS start_ts,
    ts + dur AS end_ts,
    name AS step_name
  FROM slice
  WHERE category = 'gpu_user_annotation'
    AND name = 'ProfilerStep#17'
  LIMIT 1
),

-- Get all kernel slices within that time range
kernels_in_step AS (
  SELECT
    s.id,
    s.name AS kernel_name,
    s.ts,
    s.dur,
    s.arg_set_id
  FROM slice s, step_bounds sb
  WHERE s.ts BETWEEN sb.start_ts AND sb.end_ts
    AND s.category = 'kernel'
),

-- Filter kernels matching mesh_pp, mesh_dp_shard, or SendRecv prefix
filtered_kernels AS (
  SELECT *
  FROM kernels_in_step
  WHERE kernel_name GLOB 'ncclDevKernel_SendRecv*'
     OR arg_set_id IN (
        SELECT arg_set_id
        FROM args
        WHERE key = 'args.Process Group Description'
          AND display_value IN ('mesh_pp', 'mesh_dp_shard')
     )
),

-- Get the specific arguments we want per kernel
arg_summary AS (
  SELECT
    arg_set_id,
    MAX(CASE WHEN key = 'args.Collective name' THEN display_value END) AS collective_name,
    MAX(CASE WHEN key = 'args.In msg nelems' THEN display_value END) AS in_msg_nelems,
    MAX(CASE WHEN key = 'args.Process Group Description' THEN display_value END) AS group_desc,
    MAX(CASE WHEN key = 'args.Process Group Ranks' THEN display_value END) AS group_ranks
  FROM args
  GROUP BY arg_set_id
)

-- Final output
SELECT
  'ProfilerStep#17' AS step_name,
  CASE
    WHEN k.kernel_name GLOB 'ncclDevKernel_SendRecv*' OR a.group_desc = 'mesh_pp' THEN 'PP'
    WHEN a.group_desc = 'mesh_dp_shard' THEN 'DP'
    ELSE NULL
  END AS parallelism_type,
  k.ts AS start_ts,
  k.ts + k.dur AS end_ts,
  k.dur AS duration_ns,
  k.kernel_name,
  a.collective_name,
  a.in_msg_nelems,
  a.group_desc,
  a.group_ranks
FROM filtered_kernels k
LEFT JOIN arg_summary a ON k.arg_set_id = a.arg_set_id
ORDER BY k.ts;
