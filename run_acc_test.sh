#!/bin/bash

PYTHON=./.venv/bin/python

$PYTHON test/eval_acc.py --datalen 65536 --method full --dataset_name "ruler/niah_single_1" --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k" --minference

$PYTHON test/eval_acc.py --datalen 65536 --method shadowkv --dataset_name "ruler/niah_single_1" --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k" --sparse_budget 2048 --rank 160 --chunk_size 8 --minference