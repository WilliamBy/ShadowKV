#!/bin/bash

# Test script for ShadowKV method with different rank configurations
# This script tests the ShadowKV method on ruler/qa1 dataset with various rank values

export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1
export OMP_NUM_THREADS=48
PY="python"

# Model configuration
MODEL_NAME="gradientai/Llama-3-8B-Instruct-Gradient-1048k"

# Test configuration
DATASET="ruler/qa_1"
NUM_SAMPLES=16
DATALEN=65536  # 64k context length
SPARSE_BUDGET=1024
CHUNK_SIZE=8

# Different rank values to test
# Rank controls the dimension of low-rank approximation for landmarks
# Higher ranks may capture more information but use more memory/computation
RANKS=(16 32 64 128 160 192 256 320 512 640)

# Output directory for logs and results
LOG_DIR="archive/test_logs/rank_tests"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "ShadowKV Method - Rank Parameter Testing"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Samples: $NUM_SAMPLES"
echo "Data Length: $DATALEN (64k)"
echo "Sparse Budget: $SPARSE_BUDGET"
echo "Chunk Size: $CHUNK_SIZE"
echo "========================================="
echo ""

# Create results summary file
SUMMARY_FILE="$LOG_DIR/rank_summary.txt"
echo "ShadowKV Rank Parameter Test Results" > "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "Model: $MODEL_NAME" >> "$SUMMARY_FILE"
echo "Dataset: $DATASET" >> "$SUMMARY_FILE"
echo "Samples: $NUM_SAMPLES" >> "$SUMMARY_FILE"
echo "Data Length: $DATALEN" >> "$SUMMARY_FILE"
echo "Sparse Budget: $SPARSE_BUDGET" >> "$SUMMARY_FILE"
echo "Chunk Size: $CHUNK_SIZE" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Rank | Accuracy | Prefill Time | Decode Time | Total Time | Log File" >> "$SUMMARY_FILE"
echo "-----|----------|--------------|-------------|------------|----------" >> "$SUMMARY_FILE"

# Temporary file to store results
TEMP_RESULTS="$LOG_DIR/temp_rank_results.txt"
> "$TEMP_RESULTS"

# Track the best accuracy
BEST_RANK=0
BEST_ACCURACY=0

for RANK in "${RANKS[@]}"; do
    echo "========================================="
    echo "Testing with rank=$RANK"
    echo "========================================="
    
    # Log file for this run
    LOG_FILE="$LOG_DIR/rank${RANK}_test.log"
    
    # Run test and capture output
    echo "Running test with rank=$RANK..."
    START_TIME=$(date +%s)
    
    $PY test/eval_acc.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET" \
        --datalen $DATALEN \
        --num_samples $NUM_SAMPLES \
        --method shadowkv \
        --sparse_budget $SPARSE_BUDGET \
        --rank $RANK \
        --chunk_size $CHUNK_SIZE \
        2>&1 | tee "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "✗ Test failed for rank=$RANK (exit code: $EXIT_CODE)"
        echo "$RANK   | FAILED   | -            | -           | -          | $LOG_FILE" >> "$SUMMARY_FILE"
        echo "$RANK: FAILED" >> "$TEMP_RESULTS"
    else
        echo "✓ Test completed for rank=$RANK (duration: ${DURATION}s)"
        
        # Extract metrics from log file
        # Look for accuracy in markdown table
        ACCURACY=$(grep -A 20 "^|.*model.*dataset.*|" "$LOG_FILE" | grep -v "^--" | tail -1 | awk -F'|' '{print $NF}' | tr -d ' ')
        
        # Try to extract timing information if available
        PREFILL_TIME=$(grep "Prefill" "$LOG_FILE" | tail -1 | awk '{print $NF}' | tr -d 's' || echo "N/A")
        DECODE_TIME=$(grep "Decode" "$LOG_FILE" | tail -1 | awk '{print $NF}' | tr -d 's' || echo "N/A")
        TOTAL_TIME=$(grep "Total" "$LOG_FILE" | tail -1 | awk '{print $NF}' | tr -d 's' || echo "${DURATION}s")
        
        if [ -z "$ACCURACY" ]; then
            ACCURACY="N/A"
        fi
        
        echo "  Accuracy: $ACCURACY"
        echo "  Prefill Time: $PREFILL_TIME"
        echo "  Decode Time: $DECODE_TIME"
        echo "  Total Time: $TOTAL_TIME"
        
        echo "$RANK   | $ACCURACY   | ${PREFILL_TIME}s        | ${DECODE_TIME}s        | ${TOTAL_TIME}s        | $LOG_FILE" >> "$SUMMARY_FILE"
        echo "$RANK: $ACCURACY (prefill: ${PREFILL_TIME}s, decode: ${DECODE_TIME}s)" >> "$TEMP_RESULTS"
        
        # Track best accuracy
        if [[ "$ACCURACY" != "N/A" ]]; then
            ACC_NUM=$(echo $ACCURACY | awk '{print $1}')
            if (( $(echo "$ACC_NUM > $BEST_ACCURACY" | bc -l) )); then
                BEST_ACCURACY=$ACC_NUM
                BEST_RANK=$RANK
            fi
        fi
    fi
    
    echo ""
done

# Print summary
echo "========================================="
echo "Test Summary"
echo "========================================="
cat "$TEMP_RESULTS"

echo ""
echo "========================================="
echo "Best Accuracy"
echo "========================================="
echo "Rank: $BEST_RANK"
echo "Accuracy: $BEST_ACCURACY"

echo ""
echo "========================================="
echo "Full results saved to: $SUMMARY_FILE"
echo "========================================="

# Generate visualization script
cat > "$LOG_DIR/plot_rank_results.py" << 'EOF'
import matplotlib.pyplot as plt
import re
import sys

def parse_results(summary_file):
    """Parse the summary file to extract rank vs accuracy data."""
    ranks = []
    accuracies = []
    
    with open(summary_file, 'r') as f:
        in_data = False
        for line in f:
            if '-----|----------|' in line:
                in_data = True
                continue
            if in_data and line.strip():
                parts = line.split('|')
                if len(parts) >= 2:
                    try:
                        rank = int(parts[0].strip())
                        acc_str = parts[1].strip()
                        if acc_str != 'FAILED' and acc_str != 'N/A':
                            # Extract numeric part (handle percentage or decimal)
                            acc_val = float(re.findall(r'[\d.]+', acc_str)[0])
                            if acc_val > 1:  # If > 1, likely percentage
                                acc_val = acc_val / 100
                            ranks.append(rank)
                            accuracies.append(acc_val)
                    except (ValueError, IndexError):
                        continue
    
    return ranks, accuracies

def plot_results(ranks, accuracies, output_file):
    """Create a plot of rank vs accuracy."""
    plt.figure(figsize=(12, 6))
    
    # Plot line
    plt.plot(ranks, accuracies, marker='o', linewidth=2, markersize=8, 
             color='#2E86AB', label='Accuracy')
    
    # Highlight best point
    best_idx = accuracies.index(max(accuracies))
    plt.scatter(ranks[best_idx], accuracies[best_idx], 
               s=200, color='#A23B72', zorder=5, 
               label=f'Best (rank={ranks[best_idx]}, acc={accuracies[best_idx]:.4f})',
               marker='*', edgecolors='gold', linewidth=2)
    
    plt.xlabel('Rank', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('ShadowKV: Rank vs Accuracy on ruler/qa1 (64k context, 16 samples)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Set x-axis to log scale for better visualization
    plt.xscale('log')
    plt.xticks(ranks, labels=ranks, rotation=45)
    
    # Add horizontal line at best accuracy
    plt.axhline(y=accuracies[best_idx], color='red', linestyle='--', 
               alpha=0.3, linewidth=1)
    
    # Add text annotations for each point
    for rank, acc in zip(ranks, accuracies):
        plt.annotate(f'{acc:.4f}', 
                    xy=(rank, acc), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_rank_results.py <summary_file>")
        sys.exit(1)
    
    summary_file = sys.argv[1]
    output_file = summary_file.replace('.txt', '_plot.png')
    
    ranks, accuracies = parse_results(summary_file)
    
    if not ranks:
        print("No valid data found in summary file!")
        sys.exit(1)
    
    print(f"Parsed {len(ranks)} data points")
    print(f"Ranks: {ranks}")
    print(f"Accuracies: {accuracies}")
    
    plot_results(ranks, accuracies, output_file)
EOF

# Generate plot
echo ""
echo "Generating visualization..."
python "$LOG_DIR/plot_rank_results.py" "$SUMMARY_FILE"

echo ""
echo "========================================="
echo "Testing Complete!"
echo "========================================="
echo "Results summary: $SUMMARY_FILE"
echo "Visualization: ${SUMMARY_FILE%.txt}_plot.png"
echo "Logs directory: $LOG_DIR"
echo "========================================="
