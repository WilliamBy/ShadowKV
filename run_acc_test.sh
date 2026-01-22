#!/bin/bash

PYTHON=./.venv/bin/python

DATASETS="ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2"

$PYTHON test/eval_acc.py --datalen 131072 --method full --dataset_name $DATASETS --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k"

$PYTHON test/eval_acc.py --datalen 131072 --method shadowkv --dataset_name $DATASETS --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k" --sparse_budget 2048 --rank 160 --chunk_size 8
