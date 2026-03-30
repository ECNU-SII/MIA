# RAG

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

Open `memory_serve.py` and edit the `llm_get_trace` method:
```python
# LLM-extracted workflow summary as memory
def llm_get_trace(trace: str) -> str:
    prompt = get_trace_prompt.format(trace=trace)
    messages = [
        {"role": "user", "content": prompt}
    ]
    response_obj = memory_client.chat.completions.create(
        model="qwen",
        temperature=0,
        messages=messages,
        max_tokens=4096,
        timeout=100.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response_obj.choices[0].message.content.strip()
    return content

# Raw trajectory as memory
def llm_get_trace(trace: str) -> str:
    return trace
```

Configure the run script `run.sh`:
- `MEMORY_URL`: `Memory Manager` service; set to the `Qwen3-32B` service URL if workflow extraction is needed, otherwise can be ignored.
- `PLAN_URL`: `Planner` service, can be ignored.

Start the service:
```bash
bash run.sh
```

### 4. Inference and Evaluation

Navigate to `/Inference/Trace`.

Configure and deploy the `Executor`:
```bash
bash deploy.sh
```

Configure the run script `run.sh`:
- `DATASET`: test set path, ending with `.json` or `.jsonl`
- `OUTPUT_PATH`: output path
- `MODEL_PATH`: tokenizer model path
- `TEST_CACHE_DIR`: image search cache path (required for multimodal tasks)
- `SERVICE_URL`: search service URL
- `MEMORY_URL`: memory service `URL` + `/memory`
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