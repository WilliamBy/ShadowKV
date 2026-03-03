#!/bin/bash

# PY="./.venv/bin/python"
# PY="python"
export CUDA_VISIBLE_DEVICES=1,3
export NUM_GPU=2
export OMP_NUM_THREADS=48
PY="torchrun --standalone --nnodes=1 --nproc_per_node ${NUM_GPU}"

DATASETS="ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2"

DATALEN=16384

# $PY test/eval_acc.py --datalen $DATALEN --method full --dataset_name $DATASETS --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k" || exit 1

$PY test/eval_acc.py --datalen $DATALEN --method optimized --dataset_name $DATASETS --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k" --sparse_budget 2048 --rank 160 --chunk_size 8 || exit 1
