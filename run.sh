#!/bin/bash

# PY="./.venv/bin/python"
# PY="python"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1
export OMP_NUM_THREADS=48
PY="python"

# LongBench QA tasks
# DATASETS="long_bench/narrativeqa,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/musique,long_bench/dureader,long_bench/gov_report,long_bench/samsum,long_bench/passage_retrieval_en,long_bench/lcc"

# InfiniBench
# DATASETS="infini_bench/code_debug,infini_bench/code_run,infini_bench/kv_retrieval,infini_bench/longbook_choice_eng,infini_bench/longbook_qa_chn longbook_qa_eng,infini_bench/longbook_sum_eng,infini_bench/longdialogue_qa_eng,infini_bench/math_find,infini_bench/number_string,infini_bench/passkey"

# RULER
DATASETS="ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2"

# Context length - examples will be filtered to fit within this length
DATALEN=131072
# DATALEN=98304

# Model path - update this to your model
MODEL_NAME="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
# MODEL_NAME="zai-org/glm-4-9b-chat-1m"

# LongBench SPARSE settings
# SPARSE=256

# RULER/InfiniBench SPARSE settings
SPARSE=2048

# Run with full attention (baseline)
# $PY test/eval_acc.py --datalen $DATALEN --method full --dataset_name $DATASETS --model_name "$MODEL_NAME" | tee llama-full-ruler.log || exit 1

# Run with ShadowKV attention
# $PY test/eval_acc.py --datalen $DATALEN --method shadowkv --dataset_name $DATASETS --model_name "$MODEL_NAME" --sparse_budget $SPARSE --rank 160 --chunk_size 8 | tee llama-quest-lbc.log || exit 1

# Run with Quest attention
$PY test/eval_acc.py --datalen $DATALEN --method quest --dataset_name $DATASETS --model_name "$MODEL_NAME" --sparse_budget $SPARSE --chunk_size 8 | tee llama-quest-ruler.log || exit 1

echo "LongBench evaluation complete!"
