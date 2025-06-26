-- Get time ranges for ProfilerStep#10 to #19
WITH step_bounds AS (
  SELECT
    ts AS start_ts,
    ts + dur AS end_ts,
    CAST(REPLACE(name, 'ProfilerStep#', '') AS INT) AS iteration
  FROM slice
  WHERE category = 'gpu_user_annotation'
    AND name GLOB 'ProfilerStep#1[0-9]'
),

-- Get all kernel slices within each step's time range
kernels_in_steps AS (
  SELECT
    s.id,
    s.name AS kernel_name,
    s.ts,
    s.dur,
    s.arg_set_id,
    sb.iteration
  FROM slice s
  JOIN step_bounds sb
    ON s.ts BETWEEN sb.start_ts AND sb.end_ts
  WHERE s.category = 'kernel'
),

-- Filter kernels matching mesh_pp, mesh_dp_shard, or SendRecv prefix
filtered_kernels AS (
  SELECT *
  FROM kernels_in_steps
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
    MAX(CASE WHEN key = 'args.Process Group Ranks' THEN display_value END) AS group_ranks,
    MAX(CASE WHEN key = 'args.dtype' THEN display_value END) AS dtype
  FROM args
  GROUP BY arg_set_id
)

-- Final output
SELECT
  k.iteration,
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
  a.group_ranks,
  a.dtype
FROM filtered_kernels k
LEFT JOIN arg_summary a ON k.arg_set_id = a.arg_set_id
ORDER BY k.iteration, k.ts;
str