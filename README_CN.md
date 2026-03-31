<h1 align="center">
  <img src="readme_en/MIA.png" alt="MIA Title" width="120"/><br>Memory Intelligence Agent
</h1>
<div align="center">


[![Paper](https://img.shields.io/badge/Paper-ComingSoon-b5212f.svg?logo=arxiv)](https://arxiv.org/) [![MIA Models](https://img.shields.io/badge/Models-MIA-yellow?logo=huggingface)](https://huggingface.co/LightningCreeper/MIA) [![MIA Checkpoints](https://img.shields.io/badge/Datasets-MIA-purple?logo=huggingface)](https://huggingface.co/datasets/LightningCreeper/MIA) [![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT)

[[英文](./README.md)] [[中文](./README_CN.md)]

</div>

## 🚀 新闻
- **[April 1, 2026]**:  🌈 完整的训练，评估代码和模型以及数据集已在Huggingface上公布。

## 🛠️ 工具

### 1. 在线文本搜索 💻
核心实现主要在 `web_tools/server` 中。
打开`web_tools/run.sh`，配置谷歌搜索serper key
```bash
export SERPER_KEY_ID="xxxxx"
```
启动运行脚本
```bash
cd web_tools
bash ./run.sh
```
服务 `SERVICE_URL/server`，方法 `SERVICE_URL/server/search`

### 2. 离线文本搜索 📖
核心实现主要在 `local_search` 中。
参照[search-r1](https://github.com/PeterGriffinJin/Search-R1/blob/main)里面的搭建方式，本项目使用的是wiki25本地检索。
配置路径，启动运行脚本
```bash
cd local_search
bash ./run.sh
```
服务 `http://localhost:8001/`，方法 `http://localhost:8001/retrieve`

### 3. 图搜图 🎨

项目使用的图像搜缓存：[image_search_cache](https://huggingface.co/datasets/LightningCreeper/MIA/tree/main/image_search_cache)

## ⚙️ 环境
```bash
conda create -n verl python==3.10.12
```
执行train中的install.sh脚本安装依赖.
flash-attention需要单独安装
``` bash
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install --no-cache-dir flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 🧬 数据准备

训练：🤗 [Train](https://huggingface.co/datasets/LightningCreeper/MIA/tree/main/Train)

测试：🤗 [Test](https://huggingface.co/datasets/LightningCreeper/MIA/tree/main/Test), 🤗 [TTRL](https://huggingface.co/datasets/LightningCreeper/MIA/tree/main/TTRL)


## ✨ 两阶段RL训练

### ⚡ Executor训练

我们的实现基于VeRL，主要修改部分：
交互核心实现主要在 `/Executor-Train/Train/verl/experimental/tool_agent_loop.py` 中。
`prompt` 定义在 `/Executor-Train/Train/local_search/prompt.py` 中。
自定义数据集处理 (`CustomRLHFDataset`)、奖励评分计算 (`compute_score`)在 `Executor-Train/Train/local_search/mmsearch.py` 中。
工具实现在 `verl.tools.search_tool.SearchTool`，`verl.tools.web_image_to_image_search_tool.WebImageToImageSearchTool` 。
运行脚本在 `/Executor-Train/Train/local_search/run_mmsearch_grpo.sh` 中。
**1.** 部署本地文本搜索工具

**2.** 配置 `/Executor-Train/Train/local_search/mm_search_tool_config.yaml` 与 `/Executor-Train/Train/local_search/mmsearch.yaml`：
- `mm_search_tool_config.yaml`
   - `tools[0].config.retrieval_service_url`: 本地搜索服务
   - `tools[1].config.fvqa_train_cache_path`、`tools[1].config.test_cache_path`: 测试集与验证集的图像搜索缓存路径
- `mmsearch.yaml`
   - `hydra.searchpath`: trainer配置路径
   - `data.custom_cls.path`: 自定义数据集代码路径
   - `actor_rollout_ref.rollout.multi_turn.tool_config_path`: 工具配置`mm_search_tool_config.yaml`路径

**3.** 在节点1上部署Qwen3-32B作为 `Planner & Judger` ：
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
LLM服务 `your_url/8002/v1`

**4.** 部署 `Memory-Planner` 服务，可以与`Planner & Judger`在同一节点
``` bash
cd Memory-Serve
cd TRAIN_PLANNER
```

配置运行脚本 `run.sh`: `MEMORY_URL` 与 `PLAN_URL` 全部设置为上一步部署的LLM服务。
为了提升训练效率，`memory content` 与 `initial plan` 是提前收集好的，这里只需要拿到`replan`的服务：`your_url/5000/replan_train`。

**5.** 配置训练脚本 `/Executor-Train/Train/local_search/run_mmsearch_grpo.sh`
- `JUDGE_URL`: `judge`服务，填 `your_url/8002/v1`
- `REPLAN_URL`: `replan`服务，填 `your_url/5000/replan_train`
- `WANDB_API_KEY`: WandB API 密钥（可选）
- `SAVE_CHECKPOINT_DIR`: 模型保存路径
- `DATASET_TRAIN`: 训练数据集路径
- `DATASET_VAL`: 验证数据集路径
- `REF_MODEL_PATH`: 预训练模型路径

**6.** 在节点2上启动训练
打开 `/Executor-Train/Train/` 目录
```bash
bash ./local_search/run_mmsearch_grpo.sh
```

**7.** 导出模型
```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /your_path/actor \
    --target_dir /your_path
```

我们训练的 `Executor` 🤗 [下载](https://huggingface.co/LightningCreeper/MIA/tree/main/Trained-Executor)

### ⚡ Planner训练

我们的实现基于VeRL，主要修改部分：
交互核心实现主要在 `/Planner-Train/mem-plan/verl/experimental/multi_turn_loop.py` 中。
`prompt` 定义在 `/Planner-Train/mem-plan/local_search/prompt.py` 中。
自定义数据集处理 (`CustomRLHFDataset`)、奖励评分计算 (`compute_score`)在 `/Planner-Train/mem-plan/local_search/mmsearch.py` 中。
运行脚本在 `/Planner-Train/mem-plan/local_search/run_mmsearch_grpo.sh` 中。

**1.** 在节点1上部署 `Judger` 服务：
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

**2.** 在节点2上部署Executor服务：

部署训练好的Executor：
``` bash
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /your_path/Executor \
    --tensor-parallel-size 4 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8002
```
打开 `/Serve/Train_Planner`，配置运行脚本 `serve.sh`:
- `AGENT_URL`: Executor服务的URL
- `SERVICE_URL`: 文本搜索（离线）服务的URL
- `TEST_CACHE_DIR`: 图搜图缓存路径
- `MAX_LLM_CALL_PER_RUN`: Executor与工具交互的最大轮数

启动
``` bash
bash serve.sh
```

**3.** 配置训练脚本 `/Planner-Train/mem-plan/local_search/run_mmsearch_grpo.sh`

- `JUDGE_URL`: `judge`服务，填 `your_url/8002/v1`
- `PLAN_URL`: 对`plan`进行响应的`Executor`服务，填 `your_url/5000/plan`
- `REPLAN_URL`: 对`replan`进行响应的`Executor`服务，填 `your_url/5000/replan`
- `WANDB_API_KEY`: `WandB API` 密钥（可选）
- `SAVE_CHECKPOINT_DIR`: 模型保存路径
- `DATASET_TRAIN`: 训练数据集路径
- `DATASET_VAL`: 验证数据集路径
- `REF_MODEL_PATH`: 预训练模型路径

为了提升训练效率，`memory content` 与 `image caption` 是提前收集好的。

**4.** 在节点3上启动训练
打开 `/Planner-Train/mem-plan/` 目录
```bash
bash ./local_search/run_mmsearch_grpo.sh
```

**5.** 导出模型
```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /your_path/actor \
    --target_dir /your_path
```

我们训练的 `Planner` 🤗 [下载](https://huggingface.co/LightningCreeper/MIA/tree/main/Trained-Planner)

## 🔍 推理

- [Base](./readme_cn/Base.md)

- [RAG](./readme_cn/RAG.md)

- [mem0](./readme_cn/mem0.md)

- [a-mem](./readme_cn/a-mem.md)

- [Expel](./readme_cn/Expel.md)

- [ReasoningBank](./readme_cn/ReasoningBank.md)

- [Memento](./readme_cn/Memento.md)

- [ours (no TTRL)](./readme_cn/MIA.md)

- [ours (no TTRL and no GT)](./readme_cn/MIA-nogt.md)

## 💡 TTRL

**1.** 在节点1上部署 `Memory Manager & Judger` 服务：
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

**2.** 在节点2上部署Executor服务：
``` bash
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /your_path/Executor \
    --tensor-parallel-size 4 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8002
```
打开 `/Serve/MIA-TTRL`，配置运行脚本 `serve.sh`:
- `AGENT_URL`: `Executor`服务的`URL`
- `SERVICE_URL`: 文本搜索（离线/在线）服务的`URL`
- `TEST_CACHE_DIR`: 图搜图缓存路径
- `MAX_LLM_CALL_PER_RUN`: `Executor`与工具交互的最大轮数
- `MEMORY_URL`: `Memory Maneger`服务的`URL`
- `TTRL_SAVE`: 探索期间输出路径
- `PARQUET_PATH`: 带图像测试集路径（带图像数据集需填写）

其中，在 `/Serve/MIA-TTRL/call_agent.py` 对文本搜索的设置（请选择其中一个）：
``` python
from tool_search_local import *   # 离线文本搜索
from tool_serper import *   # 在线文本搜索
```

切换 `/Serve/MIA-TTRL/serve.sh` 运行的python代码，修改模式（有监督/无监督）：
``` bash
python agent_serve_ttrl.py ....   # 每个问题结束后可以拿到Ground-Truth的场景
python agent_serve_ttrl_nogt.py ....   # 每个问题结束后无法拿到Ground-Truth的场景
```

**3.** 配置脚本 `/TTRL/TTRL/local_search/run_mmsearch_grpo.sh`（有监督），`/TTRL/TTRL-nogt/local_search/run_mmsearch_grpo.sh`（无监督）：

- `JUDGE_URL`: `judge`服务，填 `your_url/8002/v1`
- `MEMORY_URL`: 读取记忆服务，填 `your_url/5000/memory`
- `PLAN_URL`: 对`plan`进行响应的`Executor`服务，填 `your_url/5000/plan`
- `REPLAN_URL`: 对`replan`进行响应的`Executor`服务，填 `your_url/5000/replan`
- `MEMORY_BANK_SAVE_URL`: 向缓存区存储记忆服务（当前批次探索未完成），填 `your_url/5000/memory_bank_save`
- `BATCH_EVALUATE_URL`: 对当前批次样本进行评估的服务，填 `your_url/5000/batch_evaluate`
- `CONSOLIDATE_MEMORIES_URL`: 提取所有缓存区样本记忆的服务，填 `your_url/5000/consolidate_memories`
- `SAVE_MEMORIES_URL`: 保存全部记忆的服务，填 `your_url/5000/save_memory`
- `WANDB_API_KEY`: `WandB API` 密钥（可选）
- `SAVE_CHECKPOINT_DIR`: 模型保存路径
- `DATASET_TRAIN`: 数据集路径
- `DATASET_VAL`: 无效，与`DATASET_TRAIN`相同即可
- `REF_MODEL_PATH`: 初始Planner路径

启动
``` bash
bash serve.sh
```

**4.** 在节点3上启动`Planner`服务
打开 `/TTRL/TTRL/` 或 `/TTRL/TTRL-nogt/` 目录
```bash
bash ./local_search/run_mmsearch_grpo.sh
```

## ⚖️ 许可

该项目遵循MIT许可协议发布。
