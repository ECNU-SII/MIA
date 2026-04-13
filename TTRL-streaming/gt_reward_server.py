#!/usr/bin/env python3
"""
独立 GT 奖励服务（默认端口 6000）：维护 ``data_id -> ground_truth``，实现原 ``mmsearch.compute_score`` 中
基于标准答案的打分逻辑（见 ``gt_reward_core``）。

* ``POST /register_gts`` — 客户端流式注册 GT。
* ``POST /compute_gt_reward`` — 训练侧（verl worker）请求打分；body 为任意 JSON 字典，
  至少能通过 ``data_id`` 或 ``extra_info.data_id`` 解析样本 id；``extra_info`` 中应含 ``messages`` 等。
* ``GET /health`` — 已注册 ``data_id`` 数量。

环境变量：``JUDGE_URL``（与 ``mmsearch`` / ``gt_reward_core`` 一致，用于 judge LLM）。
"""

from __future__ import annotations

import argparse
import logging

from flask import Flask, jsonify, request

from gt_reward_core import compute_gt_reward_core, normalize_ground_truth

logger = logging.getLogger(__name__)

app = Flask(__name__)

DATA_ID_TO_GT: dict[str, str] = {}


def _merge_extra_info(body: dict) -> dict:
    extra = body.get("extra_info")
    if not isinstance(extra, dict):
        extra = {}
    else:
        extra = dict(extra)
    for k, v in body.items():
        if k == "extra_info":
            continue
        if k not in extra:
            extra[k] = v
    return extra


def _resolve_data_id(body: dict, extra: dict) -> str | None:
    did = body.get("data_id")
    if did is None and isinstance(extra, dict):
        did = extra.get("data_id")
    if did is None:
        return None
    return str(did).strip()


@app.route("/register_gts", methods=["POST"])
def register_gts():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    entries = data.get("entries")
    if entries is None:
        if data.get("data_id") is not None:
            entries = [data]
        else:
            return jsonify({"error": "Missing 'entries' or 'data_id'"}), 400
    if not isinstance(entries, list):
        return jsonify({"error": "'entries' must be a list"}), 400
    n = 0
    for ent in entries:
        if not isinstance(ent, dict):
            continue
        raw = ent.get("data_id")
        if raw is None:
            continue
        did = str(raw).strip()
        if not did:
            continue
        DATA_ID_TO_GT[did] = normalize_ground_truth(ent.get("ground_truth"))
        n += 1
        print("add", did, DATA_ID_TO_GT[did])
    return jsonify({"status": "ok", "registered": n, "total": len(DATA_ID_TO_GT)})


@app.route("/compute_gt_reward", methods=["POST"])
def compute_gt_reward():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Invalid JSON"}), 400

    extra = _merge_extra_info(body)
    data_id = _resolve_data_id(body, extra)
    if not data_id:
        return jsonify({"score": 0.0, "judgement": "incorrect", "error": "missing_data_id"})

    gt = DATA_ID_TO_GT.get(data_id, "")
    if not gt:
        return jsonify(
            {
                "score": 0.0,
                "judgement": "incorrect",
                "error": "unknown_data_id_or_empty_gt",
                "data_id": data_id,
            }
        )

    data_source = body.get("data_source", "")
    solution_str = body.get("solution_str", "")
    result = compute_gt_reward_core(data_source, solution_str, gt, extra)
    result["data_id"] = data_id
    print(result)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "registered_gts": len(DATA_ID_TO_GT)})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=6000)
    args = p.parse_args()
    logger.info("GT reward server DATA_ID_TO_GT=%s entries (startup)", len(DATA_ID_TO_GT))
    app.run(host=args.host, port=args.port, threaded=True)
