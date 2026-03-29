#!/bin/bash

CUDA_VISIBLE_DEVICES=0
NUM_GPU=1
OMP_NUM_THREADS=48
PY="python"

# DATASETS="ruler/fwe,ruler/vt,ruler/niah_single_1"
DATASETS="niah"
MODEL_NAME="gradientai/Llama-3-8B-Instruct-Gradient-1048k"

SPARSE=2048
# DATALENS=(16384 32768 65536 98304 131072)
DATALENS=(16384 32768 65546 98304 131072)

for DATALEN in "${DATALENS[@]}"; do
    $PY test/eval_acc.py \
        --datalen $DATALEN \
        --method local_div \
        --dataset_name $DATASETS \
        --model_name "$MODEL_NAME" \
        --sparse_budget $SPARSE \
        --rank 160 \
        --chunk_size 8 \
        --dynamic_ratio 0 \
        --num_samples 64 \
        | tee "llama-random-$DATALEN-ruler-mini.log"
        
        echo "✓ DATALEN=$DATALEN 完成"
    
done

wait # 等待所有后台任务完成
echo "所有测试完成！"