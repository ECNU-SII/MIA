   
export AGENT_URL="http://localhost:8000/v1"

export SERVICE_URL="http://127.0.0.1:8001/"

export TEST_CACHE_DIR="/your_path/datasets/FVQA_Cache/fvqa_train_cache, /your_path/datasets/FVQA_Cache/fvqa_test_cache"
export MAX_LLM_CALL_PER_RUN=20

python agent_serve.py