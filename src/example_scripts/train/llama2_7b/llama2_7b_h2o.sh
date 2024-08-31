#!/bin/bash

save_dir=llama2_7b_h2o
root=<PATH_TO_ROOT>
venv_root=<PATH_TO_VENV>

cd ${root}
source ${venv_root}

export HUGGINGFACE_HUB_CACHE="${root}/.cache"
export HF_DATASETS_CACHE="${root}/.cache/datasets"

python ${root}/src/train_kernels.py \
    --save_dir ${root}/checkpoints/$save_dir \
    --model_name llama2 \
    --model_size 0 \
    --sampling_batch_size 2 \
    --seqs_to_collect 512 \
    --half_precision \
    --heavy_ratio 0.025 \
    --recent_ratio 0.025 \
    --ker_hid 512 \
    --ker_dim 8 \
    --lr 0.001 \
    --batch_size 2 \
    --epochs 40 \
    --device cuda:0 \
    --debug \
