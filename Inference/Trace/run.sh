
DATASET=/your_path/datasets/fvqa_test.jsonl

OUTPUT_PATH=/your_path/result/fvqa_test

MODEL_PATH=/your_path/mmsearch-mem-trace/mmsearch

export TEST_CACHE_DIR="/your_path/datasets/FVQA_Cache/fvqa_test_cache"

export SERVICE_URL="SERVICE_URL/8002/"

export MEMORY_URL="http://127.0.0.1:5000/memory"

export MEMORY_BASE_URL="http://127.0.0.1:5000"

export MAX_LLM_CALL_PER_RUN=30

export JUDGE_URL="JUDGE_URL/8002/v1"

export JUDGE_PROMPT_TYPE="default"

python -u run_multi_react.py \
    --dataset "$DATASET" \
    --output "$OUTPUT_PATH" \
    --max_workers 4 \
    --max_tokens 10000\
    --model $MODEL_PATH \
    --temperature 0 \
    --presence_penalty 1.1 \
    --top_p 1.0 \
    --roll_out_count 1 \
    --main_ports "8000"