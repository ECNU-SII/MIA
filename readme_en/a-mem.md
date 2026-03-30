# A-mem

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

Navigate to `/Memory-Serve/a-mem`.

Deploy the backbone model (Qwen3-32B is used here):
```bash
bash deploy.sh
```
Start the service on port `5000`:
```bash
bash run.sh
```

### 4. Inference and Evaluation

Navigate to `/Inference/Trace`.

Configure and deploy the `Executor`:
```bash
bash deploy.sh
```

Set the environment variable:
```bash
export MEMORY_MODE="amem"
```

Configure the run script `run.sh`:
- `DATASET`: test set path, ending with `.json` or `.jsonl`
- `OUTPUT_PATH`: output path
- `MODEL_PATH`: tokenizer model path
- `TEST_CACHE_DIR`: image search cache path (required for multimodal tasks)
- `SERVICE_URL`: search service URL
- `MEMORY_URL`: memory service `URL` + `/memory_context`
- `MEMORY_BASE_URL`: memory service `URL`
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