#!/bin/bash

# Simple test script for Oracle TopK KV Cache
# This script tests the basic functionality of Oracle TopK

export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1
export OMP_NUM_THREADS=48
PY="python"

echo "========================================="
echo "Oracle TopK Test"
echo "========================================="

MODEL_NAME="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
SPARSE_BUDGET=512
MAX_LENGTH=4096

echo ""
echo "Test 1: Basic Prefill with Oracle TopK"
echo "----------------------------------------"

$PY -c "
import torch
import sys
sys.path.insert(0, '/root/workspace/ShadowKV')

from models import choose_model_class

print('Loading model with Oracle TopK...')
model = choose_model_class('$MODEL_NAME')(
    model_name='$MODEL_NAME',
    batch_size=1,
    max_length=$MAX_LENGTH,
    device='cuda:0',
    attn_mode='oracle_topk',
    sparse_budget=$SPARSE_BUDGET,
    dtype=torch.bfloat16
)

print(f'Model loaded successfully!')
print(f'KV Cache type: {model.kv_cache.__class__.__name__}')
print(f'Attention type: {model.attn_hdlr.__class__.__name__}')
print(f'Sparse budget: {model.kv_cache.sparse_budget}')

# Test prefill with a short sequence
print('\\nTesting prefill with 512 tokens...')
test_input = torch.randint(0, 128000, (1, 512)).cuda()
logits = model.prefill(test_input)
print(f'Prefill successful! Logits shape: {logits.shape}')

# Test decode
print('\\nTesting decode...')
next_token = model.generate(test_input, max_new_tokens=10)
print(f'Decode successful! Generated {next_token.shape[-1] - 512} tokens')

# Print cache stats
model.print_kv_stats()

print('\\n✓ All tests passed!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Oracle TopK test completed successfully!"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "✗ Oracle TopK test failed!"
    echo "========================================="
    exit 1
fi
