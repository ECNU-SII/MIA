#!/usr/bin/env python3
"""
流式 TTRL 客户端：本地读 parquet，按批 POST /submit_sample。

* ``CLIENT_BATCH_SIZE`` / ``--client-batch-size``：每请求合并的样本条数（默认 1）。
  须满足 ``train_batch_size % client_batch_size == 0``（train_batch_size 来自服务端 /health）。
* 服务端按**样本条数**凑满 train_batch_size 再 GRPO 更新。
* 可选 ``AGENT_REGISTER_URL`` / ``--agent-register-url``：每批在 POST TTRL **之前** 调用 Agent
  ``POST /streaming/register_images``，填充 ``DATA_ID_TO_IMAGES``。
* 发往 TTRL ``/submit_sample`` 的 ``rollout`` **不含** ``images`` / ``videos``（服务端不收）。
构造训练样本时在 ``sample_from_row_dict`` **之前** 即从行数据中去掉图/视频列；图片仅经
  ``AGENT_REGISTER_URL`` 注册给 Agent。
* ``GT_REWARD_URL``（默认 ``http://127.0.0.1:6000``）：流式注册 ``data_id -> ground_truth``（服务见仓库根目录
  ``gt_reward_server.py`` / ``run_gt_reward_server.sh``）；发往 TTRL 的 ``reward_model.ground_truth`` 会被清空。

日志与进度（``nohup``）：
  * 日志走 **stdout**，带时间戳并每条 flush；进度条（``tqdm``）走 **stderr**，含 ETA/速率。
    默认 ``nohup ...`` 会把两者都写入 ``nohup.out``。
  * ``--log-level DEBUG`` 或环境变量 ``LOG_LEVEL`` 可查看每个 chunk 的详细状态。
  * 需要更少缓冲时可使用 ``python -u streaming_ttrl_client.py ...``。
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
import time
from typing import Any

import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


_LOG = logging.getLogger(__name__)

_sd = os.path.dirname(os.path.abspath(__file__))
if _sd not in sys.path:
    sys.path.insert(0, _sd)

import streaming_agent_urls
streaming_agent_urls.ensure_agent_service_urls()



class _FlushStreamHandler(logging.StreamHandler):
    """保证每条日志立刻刷到 stdout，便于 nohup 下及时看到输出。"""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _configure_logging(level: int = logging.INFO) -> None:
    log = logging.getLogger(__name__)
    log.setLevel(level)
    log.handlers.clear()
    h = _FlushStreamHandler(sys.stdout)
    h.setLevel(level)
    h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    log.addHandler(h)
    log.propagate = False


class _DummyTqdm:
    """无 tqdm 时的占位，接口与 tqdm 上下文一致。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._n = 0
        total = kwargs.get("total")
        if total is None and args:
            total = args[0]
        self._total = int(total or 0)

    def __enter__(self) -> _DummyTqdm:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, n: int = 1) -> None:
        self._n += n
        if self._total:
            _LOG.info("进度 %s/%s 个 chunk", self._n, self._total)

    def set_postfix_str(self, s: str) -> None:
        if s:
            _LOG.debug("postfix %s", s)


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _row_dict_without_media(row: dict, *, image_key: str, video_key: str) -> dict:
    """深拷贝并去掉图/视频列，供仅文本的 rollout 样本（与 Agent 侧注册分离）。"""
    out = copy.deepcopy(row)
    out.pop(image_key, None)
    out.pop(video_key, None)
    return out


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def ground_truth_from_row_dict(row: dict) -> str:
    rm = row.get("reward_model")
    if not isinstance(rm, dict):
        return ""
    gt = rm.get("ground_truth")
    if gt is None:
        return ""
    if isinstance(gt, (list, tuple)) and len(gt) > 0:
        return str(gt[0])
    return str(gt)


def register_gt_entries(gt_server_base: str, row_dicts: list, *, timeout: float = 120.0) -> None:
    """将本批 ``data_id`` / ``ground_truth`` 注册到 ``GT_REWARD_URL``（默认 :6000）。"""
    if not gt_server_base or not row_dicts:
        return
    base = gt_server_base.rstrip("/")
    entries: list[dict] = []
    for row in row_dicts:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("data_id")
        if raw_id is None:
            continue
        entries.append({"data_id": str(raw_id), "ground_truth": ground_truth_from_row_dict(row)})
    if not entries:
        return
    r = requests.post(f"{base}/register_gts", json={"entries": entries}, timeout=timeout)
    r.raise_for_status()


def register_agent_images(agent_base: str, row_dicts: list, *, timeout: float = 120.0) -> None:
    """
    在 POST /submit_sample 之前，将本批 parquet 行的 ``data_id`` / ``images`` 同步到
    ``agent_serve_ttrl_streaming`` 的 ``POST /streaming/register_images``。
    """
    if not agent_base or not row_dicts:
        return
    base = agent_base.rstrip("/")
    entries: list[dict] = []
    for row in row_dicts:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("data_id")
        if raw_id is None:
            continue
        data_id = str(raw_id)
        images = row.get("images")
        if images is None:
            images = []
        if not isinstance(images, (list, tuple)):
            images = [images]
        entries.append({"data_id": data_id, "images": list(images)})
    if not entries:
        return
    url = f"{base}/streaming/register_images"
    r = requests.post(url, json={"entries": entries}, timeout=timeout)
    r.raise_for_status()


def _maybe_save_checkpoint(base: str, save_step: int, global_steps: int | None) -> None:
    if save_step <= 0 or global_steps is None:
        return
    # server 在 prepare 后从 global_steps=1 起计，每完成一轮 GRPO 再 +1。
    # 已完成的训练轮次 = global_steps - 1；按「每 save_step 轮存一次」应对 completed 取模（否则 save_step=2 时第 1 轮结束 gs=2 会误触发2%2==0）。
    if global_steps <= 1:
        return
    completed = global_steps - 1
    if completed % save_step != 0:
        return
    r = requests.post(f"{base}/save_checkpoint", timeout=7200)
    r.raise_for_status()
    _LOG.info(
        "checkpoint 已保存 (completed_training_steps=%s, server global_steps=%s, save_step=%s)",
        completed,
        global_steps,
        save_step,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="流式 TTRL 客户端：按批 POST /submit_sample")
    p.add_argument("--server", default=os.environ.get("TTRL_SERVER", "http://127.0.0.1:8765"))
    p.add_argument("--parquet", type=str, default=os.environ.get("DATASET_PARQUET", ""))
    p.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("REF_MODEL_PATH", ""),
        help="HF 模型目录（tokenizer/processor）；也可用环境变量 REF_MODEL_PATH",
    )
    p.add_argument(
        "--client-batch-size",
        type=int,
        default=None,
        help="每请求合并的样本条数；须整除服务端 train_batch_size。默认读 CLIENT_BATCH_SIZE 或 1",
    )
    p.add_argument(
        "--save-step",
        type=int,
        default=None,
        help=">0 时每完成 save_step 轮训练触发 /save_checkpoint（按 global_steps-1 计轮次）；默认读 SAVE_STEP",
    )
    p.add_argument("--max-prompt-length", type=int, default=None, help="默认读 MAX_PROMPT_LENGTH")
    p.add_argument("--truncation", type=str, default=None, help="默认读 TRUNCATION")
    p.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="默认读 TRUST_REMOTE_CODE（未设则为 false）",
    )
    p.add_argument(
        "--return-raw-chat",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="默认读 RETURN_RAW_CHAT（未设则为 true）",
    )
    p.add_argument(
        "--agent-register-url",
        type=str,
        default=None,
        help="Agent 根地址，用于 POST /streaming/register_images；默认读 AGENT_REGISTER_URL",
    )
    p.add_argument(
        "--gt-reward-url",
        type=str,
        default=None,
        help="GT 服务根地址，用于 POST /register_gts；默认读 GT_REWARD_URL（http://127.0.0.1:6000）",
    )
    p.add_argument("--sleep", type=float, default=0.0, help="请求间隔（秒）")
    p.add_argument("--epoch", type=int, default=0, help="传给服务端 training/epoch 日志")
    p.add_argument("--no-send-finish", action="store_true", help="最后一条不发送 finish=true")
    p.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO；也可用环境变量 LOG_LEVEL）",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭 chunk 进度条（仍输出日志）",
    )
    args = p.parse_args()

    _configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.parquet:
        _LOG.error("必须指定 --parquet 或环境变量 DATASET_PARQUET")
        raise SystemExit(2)
    if not args.model_path:
        _LOG.error("必须指定 --model-path 或环境变量 REF_MODEL_PATH")
        raise SystemExit(2)

    client_bs = args.client_batch_size if args.client_batch_size is not None else _env_int("CLIENT_BATCH_SIZE", 1)
    if client_bs < 1:
        _LOG.error("client_batch_size 必须 >= 1")
        raise SystemExit(2)

    save_step = args.save_step if args.save_step is not None else _env_int("SAVE_STEP", 0)
    max_prompt_length = (
        args.max_prompt_length if args.max_prompt_length is not None else _env_int("MAX_PROMPT_LENGTH", 24576)
    )
    truncation = args.truncation if args.truncation is not None else os.environ.get("TRUNCATION", "error")
    trust_remote_code = (
        _env_bool("TRUST_REMOTE_CODE", False) if args.trust_remote_code is None else args.trust_remote_code
    )
    return_raw_chat = _env_bool("RETURN_RAW_CHAT", True) if args.return_raw_chat is None else args.return_raw_chat

    agent_register_url = (
        args.agent_register_url
        if args.agent_register_url is not None
        else os.environ.get("AGENT_REGISTER_URL", "").strip()
    )
    gt_reward_url = (
        args.gt_reward_url
        if args.gt_reward_url is not None
        else os.environ.get("GT_REWARD_URL", "http://127.0.0.1:6000").strip()
    )

    base = args.server.rstrip("/")

    r = requests.get(f"{base}/health", timeout=30)
    r.raise_for_status()
    hi = r.json()
    _LOG.info("服务端 health: %s", hi)

    train_bs = int(hi.get("train_batch_size", 0) or 0)
    if train_bs < 1:
        _LOG.error("服务端未返回有效的 train_batch_size")
        raise SystemExit(2)
    if train_bs % client_bs != 0:
        _LOG.error(
            "client_batch_size=%s 必须整除服务端 train_batch_size=%s（即 train_batch_size %% client_batch_size == 0）",
            client_bs,
            train_bs,
        )
        raise SystemExit(2)

    try:
        import pandas as pd
    except ImportError as e:
        _LOG.error("需要: pip install pandas pyarrow")
        raise SystemExit(1) from e

    sd = _script_dir()
    if sd not in sys.path:
        sys.path.insert(0, sd)

    from omegaconf import OmegaConf

    from local_search.mmsearch import CustomRLHFDataset
    from streaming_batch_codec import (
        batch_dict_to_jsonable,
        strip_ground_truth_for_server_submit,
        strip_media_for_server_submit,
    )
    from streaming_row_utils import pandas_row_to_jsonable
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local

    _LOG.info("加载 tokenizer/processor（copy_to_local 大目录时可能较慢）...")
    local_path = copy_to_local(args.model_path, use_shm=False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    _LOG.info("模型路径就绪: %s", local_path)
    if agent_register_url:
        _LOG.info("将同步图片到 Agent: %s/streaming/register_images", agent_register_url.rstrip("/"))
    _LOG.info("将同步 GT 到: %s/register_gts", gt_reward_url.rstrip("/"))

    data_config = OmegaConf.create(
        {
            "max_prompt_length": max_prompt_length,
            "return_raw_chat": return_raw_chat,
            "truncation": truncation,
        }
    )
    ds = CustomRLHFDataset.from_config_only(tokenizer, processor, data_config)

    df = pd.read_parquet(args.parquet)
    n_rows = len(df)
    n_chunks = math.ceil(n_rows / client_bs) if n_rows > 0 else 0
    _LOG.info(
        "已读 parquet：共 %s 行；client_batch_size=%s（每请求 %s 条，约 %s 次 POST）",
        n_rows,
        client_bs,
        client_bs,
        n_chunks,
    )

    def submit_chunk(
        samples: list,
        *,
        flush_partial: bool,
        finish: bool,
        chunk_idx: int,
        n_chunks_total: int,
        row_lo: int,
        row_hi: int,
    ) -> tuple[bool, dict[str, Any]]:
        _LOG.debug(
            "[chunk %s/%s] 行 [%s, %s) 共 %s 条 → POST /submit_sample",
            chunk_idx + 1,
            n_chunks_total,
            row_lo,
            row_hi,
            len(samples),
        )
        batch_dict = collate_fn(samples)
        batch_dict = strip_media_for_server_submit(batch_dict)
        batch_dict = strip_ground_truth_for_server_submit(batch_dict)
        payload = batch_dict_to_jsonable(batch_dict)
        send_finish = finish and not args.no_send_finish
        resp = requests.post(
            f"{base}/submit_sample",
            json={
                "rollout": payload,
                "epoch": args.epoch,
                "finish": send_finish,
                "flush_partial": flush_partial,
            },
            timeout=7200,
        )
        resp.raise_for_status()
        data = resp.json()
        gs = data.get("global_steps")
        if gs is not None:
            try:
                gs = int(gs)
            except (TypeError, ValueError):
                gs = None
        st = data.get("status")
        info: dict[str, Any] = {
            "status": st,
            "global_steps": data.get("global_steps"),
            "done": data.get("done"),
            "rollout_buffer_len": data.get("rollout_buffer_len"),
        }
        if st == "step":
            _LOG.debug(
                "chunk -> %s global_steps=%s done=%s",
                st,
                data.get("global_steps"),
                data.get("done"),
            )
            _maybe_save_checkpoint(base, save_step, gs)
        else:
            _LOG.debug(
                "chunk -> %s global_steps=%s rollout_buffer_samples=%s",
                st,
                data.get("global_steps"),
                data.get("rollout_buffer_len"),
            )
        return bool(data.get("done")), info

    if n_rows == 0:
        _LOG.warning("parquet 无行，退出。")
        return

    def _make_pbar() -> Any:
        if args.no_progress or n_chunks <= 0:
            return _DummyTqdm(total=n_chunks, desc="chunks")
        if tqdm is None:
            _LOG.warning("未安装 tqdm，进度条降级为日志；可 pip install tqdm")
            return _DummyTqdm(total=n_chunks, desc="chunks")
        return tqdm(
            total=n_chunks,
            desc="HTTP chunks",
            unit="req",
            file=sys.stderr,
            mininterval=0.5,
            smoothing=0.08,
            dynamic_ncols=True,
        )

    chunk_idx = 0
    with _make_pbar() as pbar:
        for start in range(0, n_rows, client_bs):
            end = min(start + client_bs, n_rows)
            samples = []
            row_dicts = []
            print("start:", start)
            for i in range(start, end):
                row = df.iloc[i]
                row_dict = pandas_row_to_jsonable(row)
                row_dicts.append(row_dict)
                row_for_train = _row_dict_without_media(
                    row_dict, image_key=ds.image_key, video_key=ds.video_key
                )
                samples.append(ds.sample_from_row_dict(row_for_train))
            register_agent_images(agent_register_url, row_dicts)
            register_gt_entries(gt_reward_url, row_dicts)
            is_last = end >= n_rows
            t0 = time.perf_counter()
            done, info = submit_chunk(
                samples,
                flush_partial=is_last,
                finish=is_last,
                chunk_idx=chunk_idx,
                n_chunks_total=n_chunks,
                row_lo=start,
                row_hi=end,
            )
            elapsed = time.perf_counter() - t0
            chunk_idx += 1
            postfix = (
                f"{info.get('status')} gs={info.get('global_steps')} buf={info.get('rollout_buffer_len')} "
                f"last={elapsed:.1f}s"
            )
            pbar.set_postfix_str(postfix, refresh=False)
            pbar.update(1)
            if done:
                _LOG.info("训练结束（步数上限或已发送 finish）。")
                break
            if args.sleep:
                time.sleep(args.sleep)

    _LOG.info("完成。")


if __name__ == "__main__":
    main()
