#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPU=1
export OMP_NUM_THREADS=48
PY="torchrun --standalone --nnodes=1 --nproc_per_node ${NUM_GPU}"

DATASETS="niah"

DATALEN=16384

$PY test/eval_acc.py --datalen $DATALEN --method full --dataset_name $DATASETS --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k" || exit 1

# $PY test/eval_acc.py --datalen $DATALEN --method optimized --dataset_name $DATASETS --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k" --sparse_budget 2048 --rank 160 --chunk_size 8 || exit 1
