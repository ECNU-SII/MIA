# Mem0

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

### 3. Inference and Evaluation

Navigate to `/Inference/Mem0`.

Configure and deploy the `Executor`:
```bash
bash deploy.sh
```

Open `react_agent.py` and configure:
- `VLLM_API_BASE`: the URL of the memory extraction service; we use a locally deployed `Qwen3-32B`
- `VLLM_API_KEY`: API key
- `LOCAL_EMBED_MODEL_PATH`: path to the `embedding` model

Configure the run script `run.sh`:
- `DATASET`: test set path, ending with `.json` or `.jsonl`
- `OUTPUT_PATH`: output path
- `MODEL_PATH`: tokenizer model path
- `TEST_CACHE_DIR`: image search cache path (required for multimodal tasks)
- `SERVICE_URL`: search service URL
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