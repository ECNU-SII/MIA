export MEMORY_URL="http://localhost:8002/v1"
export PLAN_URL="http://localhost:8002/v1"

python Memento.py \
  --model_name qwen \
  --port 5000 \
  --host 0.0.0.0