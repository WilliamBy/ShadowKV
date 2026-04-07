#!/bin/bash

# Test script for Optimized attention method with different chunk_size configurations
# This script tests the optimized method on ruler/qa1 dataset with various chunk sizes

export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1
export OMP_NUM_THREADS=48
PY="python"

# Model configuration
MODEL_NAME="gradientai/Llama-3-8B-Instruct-Gradient-1048k"

# Test configuration
DATASET="ruler/qa_1"
NUM_SAMPLES=16
DATALEN=131072
SPARSE_BUDGET=2048
RANK=160

# Different chunk_size values to test
# These sizes represent different trade-offs between memory and computation
CHUNK_SIZES=(4 8 16 32 64 128 256)

# Output directory for logs and results
LOG_DIR="archive/test_logs/chunk_size_tests"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Optimized Method - Chunk Size Testing"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Samples: $NUM_SAMPLES"
echo "Sparse Budget: $SPARSE_BUDGET"
echo "Rank: $RANK"
echo "========================================="
echo ""

# Create results summary file
SUMMARY_FILE="$LOG_DIR/chunk_size_summary.txt"
echo "Optimized Method Chunk Size Test Results" > "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "Model: $MODEL_NAME" >> "$SUMMARY_FILE"
echo "Dataset: ruler/$DATASET" >> "$SUMMARY_FILE"
echo "Samples: $NUM_SAMPLES" >> "$SUMMARY_FILE"
echo "Data Length: $DATALEN" >> "$SUMMARY_FILE"
echo "Sparse Budget: $SPARSE_BUDGET" >> "$SUMMARY_FILE"
echo "Rank: $RANK" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Chunk Size | Accuracy | Log File" >> "$SUMMARY_FILE"
echo "-----------|----------|----------" >> "$SUMMARY_FILE"

# Temporary file to store results
TEMP_RESULTS="$LOG_DIR/temp_results.txt"
> "$TEMP_RESULTS"

for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    echo "========================================="
    echo "Testing with chunk_size=$CHUNK_SIZE"
    echo "========================================="
    
    # Log file for this run
    LOG_FILE="$LOG_DIR/chunk${CHUNK_SIZE}_test.log"
    
    # Run test and capture output
    echo "Running test with chunk_size=$CHUNK_SIZE..."
    $PY test/eval_acc.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET" \
        --datalen $DATALEN \
        --num_samples $NUM_SAMPLES \
        --method optimized \
        --sparse_budget $SPARSE_BUDGET \
        --rank $RANK \
        --chunk_size $CHUNK_SIZE \
        2>&1 | tee "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "✗ Test failed for chunk_size=$CHUNK_SIZE (exit code: $EXIT_CODE)"
        echo "$CHUNK_SIZE     | FAILED   | $LOG_FILE" >> "$SUMMARY_FILE"
        echo "$CHUNK_SIZE: FAILED" >> "$TEMP_RESULTS"
    else
        echo "✓ Test completed for chunk_size=$CHUNK_SIZE"
        
        # Extract accuracy from log file
        # Look for the accuracy line in the markdown table output
        ACCURACY=$(grep -A 20 "^|.*model.*dataset.*|" "$LOG_FILE" | grep -v "^--" | tail -1 | awk -F'|' '{print $NF}' | tr -d ' ')
        
        if [ -z "$ACCURACY" ]; then
            ACCURACY="N/A"
        fi
        
        echo "  Accuracy: $ACCURARY"
        echo "$CHUNK_SIZE     | $ACCURACY   | $LOG_FILE" >> "$SUMMARY_FILE"
        echo "$CHUNK_SIZE: $ACCURACY" >> "$TEMP_RESULTS"
    fi
    
    echo ""
    
    # Clean up GPU memory
    sleep 2
done

echo "========================================="
echo "All chunk size tests completed!"
echo "========================================="
echo ""
echo "Summary:"
cat "$SUMMARY_FILE"
echo ""
echo "========================================="
echo "Test Configuration:"
echo "  - Model: $MODEL_NAME"
echo "  - Dataset: ruler/$DATASET"
echo "  - Samples: $NUM_SAMPLES"
echo "  - Data Length: $DATALEN"
echo "  - Sparse Budget: $SPARSE_BUDGET"
echo "  - Rank: $RANK"
echo "  - Tested Chunk Sizes: ${CHUNK_SIZES[*]}"
echo ""
echo "Results Summary:"
cat "$TEMP_RESULTS"
echo ""
echo "========================================="
echo "Logs saved to: $LOG_DIR"
echo "Summary saved to: $SUMMARY_FILE"
echo "========================================="
