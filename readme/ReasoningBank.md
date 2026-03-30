# ReasoningBank

### 1. 部署Judger
``` bash
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /your_path/Qwen/Qwen3-32B \
    --tensor-parallel-size 4 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8002
```

### 2. 部署文本搜索服务
核心实现在 `local_search` (离线), `web_tools` (在线)。

### 3. 部署记忆服务

打开`/Memory-Serve`

配置运行脚本 `run.sh`:
- `MEMORY_URL`: `Memory Manager`服务，填`Qwen3-32B`的服务。
- `PLAN_URL`: `Planner`服务，可以忽视。

启动脚本
```bash
bash run.sh
```

### 4. 推理与评估

打开 `/Inference/ReasoningBank`

配置并部署 `Executor`:
```bash
bash deploy.sh
```

配置 `config.yaml`:
- `memory`: 记忆存储方式
- `embedding_model`: `embedding` 模型
- `llm`: `llm` 驱动，这里使用 `Qwen3-32B`

配置运行脚本 `run.sh`:
- `DATASET`: 测试集路径，`.json` 或 `.jsonl` 结尾
- `OUTPUT_PATH`: 输出路径
- `MODEL_PATH`: 分词模型路径
- `TEST_CACHE_DIR`: 图像搜索缓存路径（多模态任务需填写）
- `SERVICE_URL`: 搜索服务路径
- `MEMORY_URL`: 记忆服务 `URL` + `/memory`
- `MEMORY_BASE_URL`: 记忆服务 `URL`
- `MAX_LLM_CALL_PER_RUN`: 工具交互轮次上限
- `JUDGE_URL`: `Judger` 服务 `URL`

其中，在 `react_agent.py` 对文本搜索的设置（请选择其中一个）：
``` python
from tool_search_local import *   # 离线文本搜索
from tool_serper import *   # 在线文本搜索
```

运行:
```bash
bash run.sh
```
