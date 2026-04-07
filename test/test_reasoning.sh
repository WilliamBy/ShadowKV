#!/bin/bash

# Test script for reasoning datasets
# This script runs predictions on reasoning datasets and evaluates the results

export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1
export OMP_NUM_THREADS=48
PY="python"

# Model configuration
MODEL_NAME="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
OUTPUT_DIR="archive/reasoning"

# Reasoning datasets to test
# Available datasets: AIME, AIME24, GPQAd, GPQAm, GPQA50, GPQA50c, MATH500, MATH50
DATASETS=("MATH50")

# Attention modes to test
ATTN_MODES=("full")

# Additional configuration
SPARSE_BUDGET=512
RANK=32
CHUNK_SIZE=128
MAX_GEN=16384
TEMPERATURE=0.6
TOP_P=0.95
BSZ=1

echo "========================================="
echo "Reasoning Dataset Test"
echo "Model: $MODEL_NAME"
echo "Datasets: ${DATASETS[*]}"
echo "========================================="

for DATASET in "${DATASETS[@]}"; do
    for ATTN_MODE in "${ATTN_MODES[@]}"; do
        echo ""
        echo "========================================="
        echo "Running: Dataset=$DATASET, Mode=$ATTN_MODE"
        echo "========================================="
        
        # Step 1: Generate predictions
        echo "Step 1: Generating predictions..."
        $PY data/reasoning/pred.py \
            --model_name "$MODEL_NAME" \
            --dataset "$DATASET" \
            --top_p "$TOP_P" \
            --attn_mode "$ATTN_MODE" \
            --output_dir "$OUTPUT_DIR" \
            --max_gen $MAX_GEN \
            --temperature $TEMPERATURE \
            --batch_size $BSZ \
            --seed 42 \
            --sparse_budget $SPARSE_BUDGET \
            --rank $RANK \
            --chunk_size $CHUNK_SIZE
        
        if [ $? -ne 0 ]; then
            echo "✗ Prediction failed for $DATASET with $ATTN_MODE"
            continue
        fi
        echo "✓ Prediction completed for $DATASET with $ATTN_MODE"
        
        # Step 2: Evaluate results
        echo ""
        echo "Step 2: Evaluating results..."
        $PY data/reasoning/eval.py \
            --data_dir "$OUTPUT_DIR" \
            --model_name "$MODEL_NAME" \
            --dataset "$DATASET" \
            --max_length $MAX_GEN
        
        if [ $? -eq 0 ]; then
            echo "✓ Evaluation completed for $DATASET with $ATTN_MODE"
        else
            echo "✗ Evaluation failed for $DATASET with $ATTN_MODE"
        fi
    done
done

echo ""
echo "========================================="
echo "All tests completed!"
echo "Results saved to: $OUTPUT_DIR/results.json"
echo "Correct answers saved to: $OUTPUT_DIR/corrects.json"
echo "========================================="
