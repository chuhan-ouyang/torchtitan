#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

# From multinode_trainer.slurm
export LOGLEVEL=INFO
# Enable for A100
export FI_PROVIDER="efa"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BUFFSIZE=1048576

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml"}

# From run_train.sh
# TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
# PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
# TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \

dcgmi profile --pause

srun \
  --job-name=torchtitan_multi_node \
  --partition=train \
  --nodes=4 \
  --ntasks-per-node=1 \
  --gpus-per-task=4 \
  --cpus-per-task=16 \
  --kill-on-bad-exit=1 \
  --export=ALL \
  torchrun --nnodes 4 --nproc_per_node 4 --rdzv_id tt_multi_${SLURM_JOB_ID} --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500"\
  -m torchtitan.train --job.config_file ${CONFIG_FILE} "$@" \

dcgmi profile --resume