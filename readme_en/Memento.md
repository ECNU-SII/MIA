# Memento

### 1. Deploy Judger
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /your_path/Qwen/Qwen3-32B \
    --tensor-parallel-size 4 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8002
```

### 2. Deploy Text Search Service
The core implementation is in `local_search` (offline) and `web_tools` (online).

### 3. Deploy Memory Service

Navigate to `/Memory-Serve`.

Configure the run script `Memento.sh`:
- `MEMORY_URL`: can be ignored.
- `PLAN_URL`: `Planner` service, set to the `Qwen3-32B` service URL.

Start the service:
```bash
bash run.sh
```

### 4. Inference and Evaluation

Navigate to `/Inference/MIA-noTTRL`.

Configure and deploy the `Executor`:
```bash
bash deploy.sh
```

Configure the run script `run.sh`:
- `DATASET`: test set path, ending with `.json` or `.jsonl`
- `OUTPUT_PATH`: output path
- `MODEL_PATH`: tokenizer model path
- `MAX_REFLECTION_TIMES`: maximum number of reflection iterations
- `TEST_CACHE_DIR`: image search cache path (required for multimodal tasks)
- `SERVICE_URL`: search service URL
- `PLAN_URL`: `plan` service `URL` + `/plan`
- `REPLAN_JUDGE_URL`: `reflect` service `URL` + `/judge_replan`
- `REPLAN_URL`: `replan` service `URL` + `/replan`
- `MEMORY_BASE_URL`: memory service URL
- `MAX_LLM_CALL_PER_RUN`: maximum number of tool interaction rounds
- `JUDGE_URL`: Judger service URL

In `react_agent.py`, configure the text search method (choose one):
```python
from tool_search_local import *   # offline text search
from tool_serper import *   # online text search
```

Run:
```bash
bash run.sh
```