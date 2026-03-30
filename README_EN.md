# MIA

[[Chinese](./README.md)] [[English](./README_EN.md)]

## Tools

### 1. Online Text Search
The core implementation is mainly in `web_tools/server`.
Open `web_tools/run.sh` and configure the Google Search Serper key:
```bash
export SERPER_KEY_ID="xxxxx"
```
Start the run script:
```bash
cd web_tools
bash ./run.sh
```
Service `SERVICE_URL/server`, method `SERVICE_URL/server/search`

### 2. Offline Text Search
The core implementation is mainly in `local_search`.
Refer to the setup instructions in [search-r1](https://github.com/PeterGriffinJin/Search-R1/blob/main). This project uses wiki25 local retrieval.
Configure the path and start the run script:
```bash
cd local_search
bash ./run.sh
```
Service `http://localhost:8001/`, method `http://localhost:8001/retrieve`

### 3. Image-to-Image Search

The image search cache used in this project: [image_search_cache](https://huggingface.co/datasets/LightningCreeper/MIA/tree/main/image_search_cache)

---

## Environment
```bash
conda create -n verl python==3.10.12
```
Run the `install.sh` script in the `train` directory to install dependencies.
Flash-attention needs to be installed separately:
```bash
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install --no-cache-dir flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

---

## Data Preparation

Training: [Train](https://huggingface.co/datasets/LightningCreeper/MIA/tree/main/Train)
Testing: [Test](https://huggingface.co/datasets/LightningCreeper/MIA/tree/main/Test), [TTRL](https://huggingface.co/datasets/LightningCreeper/MIA/tree/main/TTRL)

---

## Two-Stage RL Training

### Executor Training

Our implementation is based on VeRL. Key modifications:
- The core interaction implementation is in `/Executor-Train/Train/verl/experimental/tool_agent_loop.py`.
- `prompt` is defined in `/Executor-Train/Train/local_search/prompt.py`.
- Custom dataset processing (`CustomRLHFDataset`) and reward score computation (`compute_score`) are in `Executor-Train/Train/local_search/mmsearch.py`.
- Tool implementations are in `verl.tools.search_tool.SearchTool` and `verl.tools.web_image_to_image_search_tool.WebImageToImageSearchTool`.
- The run script is at `/Executor-Train/Train/local_search/run_mmsearch_grpo.sh`.

**1.** Deploy the local text search tool.

**2.** Configure `/Executor-Train/Train/local_search/mm_search_tool_config.yaml` and `/Executor-Train/Train/local_search/mmsearch.yaml`:
- `mm_search_tool_config.yaml`
  - `tools[0].config.retrieval_service_url`: local search service URL
  - `tools[1].config.fvqa_train_cache_path`, `tools[1].config.test_cache_path`: image search cache paths for the test and validation sets
- `mmsearch.yaml`
  - `hydra.searchpath`: trainer config path
  - `data.custom_cls.path`: custom dataset code path
  - `actor_rollout_ref.rollout.multi_turn.tool_config_path`: path to the tool config `mm_search_tool_config.yaml`

**3.** Deploy Qwen3-32B on Node 1 as `Planner & Judger`:
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
LLM service at `your_url/8002/v1`

**4.** Deploy the `Memory-Planner` service (can be on the same node as `Planner & Judger`):
```bash
cd Memory-Serve
cd TRAIN_PLANNER
```
Configure the run script `run.sh`: set both `MEMORY_URL` and `PLAN_URL` to the LLM service deployed in the previous step.
To improve training efficiency, `memory content` and `initial plan` are collected in advance. Only the `replan` service is needed here: `your_url/5000/replan_train`.

**5.** Configure the training script `/Executor-Train/Train/local_search/run_mmsearch_grpo.sh`:
- `JUDGE_URL`: judge service, set to `your_url/8002/v1`
- `REPLAN_URL`: replan service, set to `your_url/5000/replan_train`
- `WANDB_API_KEY`: WandB API key (optional)
- `SAVE_CHECKPOINT_DIR`: model save path
- `DATASET_TRAIN`: training dataset path
- `DATASET_VAL`: validation dataset path
- `REF_MODEL_PATH`: pretrained model path

**6.** Start training on Node 2. Navigate to `/Executor-Train/Train/`:
```bash
bash ./local_search/run_mmsearch_grpo.sh
```

**7.** Export the model:
```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /your_path/actor \
    --target_dir /your_path
```

Download our trained `Executor` [here]()

---

### Planner Training

Our implementation is based on VeRL. Key modifications:
- The core interaction implementation is in `/Planner-Train/mem-plan/verl/experimental/multi_turn_loop.py`.
- `prompt` is defined in `/Planner-Train/mem-plan/local_search/prompt.py`.
- Custom dataset processing (`CustomRLHFDataset`) and reward score computation (`compute_score`) are in `/Planner-Train/mem-plan/local_search/mmsearch.py`.
- The run script is at `/Planner-Train/mem-plan/local_search/run_mmsearch_grpo.sh`.

**1.** Deploy the `Judger` service on Node 1:
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

**2.** Deploy the Executor service on Node 2.

Deploy the trained Executor:
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /your_path/Executor \
    --tensor-parallel-size 4 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8002
```
Navigate to `/Serve/Train_Planner` and configure the run script `serve.sh`:
- `AGENT_URL`: Executor service URL
- `SERVICE_URL`: offline text search service URL
- `TEST_CACHE_DIR`: image-to-image search cache path
- `MAX_LLM_CALL_PER_RUN`: maximum number of interaction rounds between Executor and tools

Start the service:
```bash
bash serve.sh
```

**3.** Configure the training script `/Planner-Train/mem-plan/local_search/run_mmsearch_grpo.sh`:
- `JUDGE_URL`: judge service, set to `your_url/8002/v1`
- `PLAN_URL`: Executor service for `plan` responses, set to `your_url/5000/plan`
- `REPLAN_URL`: Executor service for `replan` responses, set to `your_url/5000/replan`
- `WANDB_API_KEY`: WandB API key (optional)
- `SAVE_CHECKPOINT_DIR`: model save path
- `DATASET_TRAIN`: training dataset path
- `DATASET_VAL`: validation dataset path
- `REF_MODEL_PATH`: pretrained model path

To improve training efficiency, `memory content` and `image caption` are collected in advance.

**4.** Start training on Node 3. Navigate to `/Planner-Train/mem-plan/`:
```bash
bash ./local_search/run_mmsearch_grpo.sh
```

**5.** Export the model:
```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /your_path/actor \
    --target_dir /your_path
```

Download our trained `Planner` [here]()

---

## Inference

- [Base](./readme_en/Base.md)
- [RAG](./readme_en/RAG.md)
- [mem0](./readme_en/mem0.md)
- [a-mem](./readme_en/a-mem.md)
- [Expel](./readme_en/Expel.md)
- [ReasoningBank](./readme_en/ReasoningBank.md)
- [Memento](./readme_en/Memento.md)
- [ours (no TTRL)](./readme_en/MIA.md)
- [ours (no TTRL and no GT)](./readme_en/MIA-nogt.md)

---

## TTRL

**1.** Deploy the `Memory Manager & Judger` service on Node 1:
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

**2.** Deploy the Executor service on Node 2:
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /your_path/Executor \
    --tensor-parallel-size 4 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8002
```
Navigate to `/Serve/MIA-TTRL` and configure the run script `serve.sh`:
- `AGENT_URL`: Executor service URL
- `SERVICE_URL`: offline/online text search service URL
- `TEST_CACHE_DIR`: image-to-image search cache path
- `MAX_LLM_CALL_PER_RUN`: maximum number of interaction rounds between Executor and tools
- `MEMORY_URL`: Memory Manager service URL
- `TTRL_SAVE`: output path during exploration
- `PARQUET_PATH`: test set path with images (required for image datasets)

In `/Serve/MIA-TTRL/call_agent.py`, configure the text search method (choose one):
```python
from tool_search_local import *   # offline text search
from tool_serper import *   # online text search
```

Switch the Python script in `/Serve/MIA-TTRL/serve.sh` to select the mode (supervised / unsupervised):
```bash
python agent_serve_ttrl.py ....   # scenario where Ground-Truth is available after each question
python agent_serve_ttrl_nogt.py ....   # scenario where Ground-Truth is unavailable after each question
```

**3.** Configure the script `/TTRL/TTRL/local_search/run_mmsearch_grpo.sh` (supervised) or `/TTRL/TTRL-nogt/local_search/run_mmsearch_grpo.sh` (unsupervised):
- `JUDGE_URL`: judge service, set to `your_url/8002/v1`
- `MEMORY_URL`: memory retrieval service, set to `your_url/5000/memory`
- `PLAN_URL`: Executor service for `plan` responses, set to `your_url/5000/plan`
- `REPLAN_URL`: Executor service for `replan` responses, set to `your_url/5000/replan`
- `MEMORY_BANK_SAVE_URL`: service for saving memories to the buffer (current batch exploration not yet complete), set to `your_url/5000/memory_bank_save`
- `BATCH_EVALUATE_URL`: service for evaluating current batch samples, set to `your_url/5000/batch_evaluate`
- `CONSOLIDATE_MEMORIES_URL`: service for extracting memories from all buffered samples, set to `your_url/5000/consolidate_memories`
- `SAVE_MEMORIES_URL`: service for saving all memories, set to `your_url/5000/save_memory`
- `WANDB_API_KEY`: WandB API key (optional)
- `SAVE_CHECKPOINT_DIR`: model save path
- `DATASET_TRAIN`: dataset path
- `DATASET_VAL`: unused, set to the same as `DATASET_TRAIN`
- `REF_MODEL_PATH`: initial Planner path

Start the service:
```bash
bash serve.sh
```

**4.** Start the `Planner` training on Node 3. Navigate to `/TTRL/TTRL/` or `/TTRL/TTRL-nogt/`:
```bash
bash ./local_search/run_mmsearch_grpo.sh
```
