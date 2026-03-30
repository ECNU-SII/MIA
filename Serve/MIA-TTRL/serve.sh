
export AGENT_URL="http://localhost:8000/v1"

export SERVICE_URL="SERVICE_URL/8002/"

# export SERVICE_URL="http://127.0.0.1:8001/"

export TEST_CACHE_DIR="/your_path/datasets/livevqa_search_results_cache"
export MAX_LLM_CALL_PER_RUN=20

export MEMORY_URL="MEMORY_URL/8002/v1"
export TTRL_SAVE="/your_path/ttrl_save/fvqa_test/"

export PARQUET_PATH="/your_path/datasets_ttrl/livevqa_final_converted.parquet"

python agent_serve_ttrl.py \
  --model_name qwen \
  --port 5000 \
  --host 0.0.0.0

