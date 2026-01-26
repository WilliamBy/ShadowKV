#!/bin/bash

PYTHON=./.venv/bin/python

$PYTHON test/e2e.py --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k" --datalen "60k" 