# Copyright 2025 the MIA / TTRL authors
"""
由单一前缀 ``TTRL_AGENT_BASE``（无尾斜杠）推导 Agent 相关 HTTP 地址。

仅在对应环境变量**未设置或为空**时写入，便于 shell 只导出 ``JUDGE_URL`` + ``TTRL_AGENT_BASE``。
``JUDGE_URL``、``GT_REWARD_URL`` 从不由此模块推导，须在 shell 中单独指定（GT 服务与 Agent 根地址不同）。
"""

from __future__ import annotations

import os

# 相对 Agent 根路径（与 agent_serve_ttrl_streaming 路由一致）
_AGENT_PATHS: dict[str, str] = {
    "MEMORY_URL": "/memory",
    "PLAN_URL": "/plan",
    "REPLAN_URL": "/replan",
    "MEMORY_BANK_SAVE_URL": "/memory_bank_save",
    "BATCH_EVALUATE_URL": "/batch_evaluate",
    "CONSOLIDATE_MEMORIES_URL": "/consolidate_memories",
    "SAVE_MEMORIES_URL": "/save_memory",
}


def ensure_agent_service_urls() -> None:
    base = (os.environ.get("TTRL_AGENT_BASE") or "").strip().rstrip("/")
    if not base:
        return
    if not (os.environ.get("AGENT_REGISTER_URL") or "").strip():
        os.environ["AGENT_REGISTER_URL"] = base
    for key, path in _AGENT_PATHS.items():
        if (os.environ.get(key) or "").strip():
            continue
        os.environ[key] = f"{base}{path}"
