#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
NGPU=8
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"} 
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_8b.toml"} 

export HF_HOME=/pscratch/sd/c/co232/hf_cache

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
torchrun --nproc_per_node=${NGPU} \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.train --job.config_file ${CONFIG_FILE} $overrides

