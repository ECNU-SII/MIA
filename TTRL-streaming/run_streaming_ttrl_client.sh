#!/bin/bash

set -x


export WANDB_API_KEY=""

export JUDGE_URL="JUDGE_URL/8002/v1"

export TTRL_AGENT_BASE="TTRL_AGENT_BASE/5000"

export GT_REWARD_URL="http://127.0.0.1:6000"

export WANDB_MODE="offline"
export MODEL_MAX_LEN=32786

# ---------- 客户端：与 run_streaming_ttrl_server.sh / 服务端对齐的只有「模型路径、长度、raw_chat」等；train_batch_size 只在服务端设 ----------
export TTRL_SERVER="http://127.0.0.1:8765"

export DATASET_PARQUET="/your_path/2wiki_converted.parquet"


export REF_MODEL_PATH="/your_path/planner"

# 可选：global_steps % SAVE_STEP == 0 时 POST /save_checkpoint（0=从不）
export SAVE_STEP=50
export MAX_PROMPT_LENGTH=24576
export TRUNCATION=error
export RETURN_RAW_CHAT=true
export TRUST_REMOTE_CODE=false

export CLIENT_BATCH_SIZE=32


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTRL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TTRL_ROOT}"


mkdir -p ./logs

PYTHONUNBUFFERED=1 python3 streaming_ttrl_client.py \
    2>&1 | tee ./logs/streaming_ttrl_client.log
