export ANONYMIZED_TELEMETRY=false

DATASET=/your_path/datasets/2wiki.jsonl

OUTPUT_PATH=/your_path/mmsearch-mem0/2wiki

MODEL_PATH=/your_path/mmsearch-mem-trace/mmsearch

export SERVICE_URL="http://127.0.0.1:8001/"

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

