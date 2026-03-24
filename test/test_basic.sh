PY=".venv/bin/python"
OMP_NUM_THREADS=48

$PY test/test_generate.py --method shadowkv --max_length 4096 --test basic