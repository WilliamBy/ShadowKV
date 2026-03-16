MODEL_TYPE="gradientai/Llama-3-8B-Instruct-Gradient-1048k"

MODEL_NAME=$(basename $MODEL_TYPE)
MAX_LENGTH=32000
NUM_GPUS=1
INPUT_DIR="./data/long_bench/data/long"
OUTPUT_DIR="./archive"
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}/longgen_${MAX_LENGTH}_${method}.json"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 确保输出目录存在
# 运行 inference.py
python inference.py --model $MODEL_TYPE --max_length $MAX_LENGTH --gpu $NUM_GPUS  --input_file $INPUT_DIR  --output_file $OUTPUT_FILE 
CSV_PATH="/home/yuhao/THREADING-THE-NEEDLE/Evalution/results/accuracy_results.csv"
# 运行 eval.py
python eval.py --data $OUTPUT_FILE --csv $CSV_PATH --gpu $NUM_GPUS