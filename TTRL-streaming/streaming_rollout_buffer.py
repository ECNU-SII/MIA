# Copyright 2025 the MIA / TTRL authors
"""
流式 TTRL：服务端缓存 **已完成 rollout** 的 DataProto（每行对应一条展开后的序列）。

累计 ``len(DataProto)`` 达到阈值后再 ``concat`` 并做 GRPO 更新。阈值应设为
``data.train_batch_size * rollout.n``：每条 prompt 经 ``rollout.n`` 次采样后行数会乘 n，
若仍用「仅 prompt 条数」作阈值，会在 ``n>1`` 时只切下前 ``train_batch_size`` 行，
破坏整组 GRPO 且 dump 的 jsonl 行数不会是 ``batch×n``。

支持单次请求携带多样本（len(DataProto)==k）：按累计序列行数判断，而非按 HTTP 请求次数。
"""

from __future__ import annotations

import threading
from typing import Any


class StreamingRolloutBuffer:
    """
    线程安全的 rollout 缓存。

    每次 append 一个 **post-rollout** ``DataProto``（``len(dp)`` 为序列行数，已含 ``rollout.n``）。
    累计 ``sum(len(dp))`` 达到 ``rows_per_update`` 时 ``concat`` 并返回；若合并后超出，则切出
    前 ``rows_per_update`` 行用于更新，剩余放回缓存。
    """

    def __init__(self, rows_per_update: int):
        if rows_per_update < 1:
            raise ValueError("rows_per_update must be >= 1")
        self._rows_per_update = rows_per_update
        self._cache: list[Any] = []
        self._lock = threading.Lock()

    def _total_samples(self) -> int:
        return sum(len(x) for x in self._cache)

    def append(self, dp: Any) -> Any:
        """
        追加一次 post-rollout。累计序列行数未满则返回 None；满则返回至少含 ``rows_per_update`` 行
        的拼接 ``DataProto``（通常恰好为该数）。
        """
        from verl import DataProto

        with self._lock:
            self._cache.append(dp)
            total = self._total_samples()
            if total < self._rows_per_update:
                return None

            merged = DataProto.concat(self._cache)
            self._cache.clear()

            if len(merged) > self._rows_per_update:
                out = merged.slice(0, self._rows_per_update)
                rest = merged.slice(self._rows_per_update, len(merged))
                if len(rest) > 0:
                    self._cache.append(rest)
                return out

            return merged

    def flush_partial(self) -> Any:
        """拼接并清空缓存中剩余样本（不足一批时用于收尾）。"""
        from verl import DataProto

        with self._lock:
            if not self._cache:
                return None
            merged = DataProto.concat(self._cache)
            self._cache.clear()
            return merged if len(merged) > 0 else None

    def __len__(self) -> int:
        """当前缓存中的样本条数（非请求次数）。"""
        with self._lock:
            return self._total_samples()
