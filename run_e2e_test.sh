#!/bin/bash

PY=./.venv/bin/python
OMP_NUM_THREADS=48

$PY test/e2e.py --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k" --datalen "60k" --method "tova"