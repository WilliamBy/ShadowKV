#!/bin/bash

# PY="./.venv/bin/python"
# PY="python"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1
export OMP_NUM_THREADS=48
PY="python"

# LongBench QA tasks (all use ~32 token generation)
DATASETS="long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa"

# Context length - examples will be filtered to fit within this length
DATALEN=16384

# Model path - update this to your model
MODEL_NAME="gradientai/Llama-3-8B-Instruct-Gradient-1048k"

echo "Running LongBench evaluation with:"
echo "  Datasets: $DATASETS"
echo "  Context length: $DATALEN"
echo "  Model: $MODEL_NAME"
echo "  GPUs: $NUM_GPU"

# Run with full attention (baseline)
$PY data/long_bench/pred.py --datalen $DATALEN --method fullkv --dataset_name $DATASETS --model_name "$MODEL_NAME" | tee llama3-fullkv-lbc.log || exit 1

# Run with ShadowKV attention
$PY data/long_bench/pred.py --datalen $DATALEN --method shadowkv --dataset_name $DATASETS --model_name "$MODEL_NAME" --sparse_budget 2048 --rank 160 --chunk_size 8 | tee llama3-shadowkv-lbc.log || exit 1

echo "LongBench evaluation complete!"
