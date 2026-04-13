#!/usr/bin/env python3
"""
流式 TTRL 服务端（Flask）。

- 训练数据只在客户端：服务端不读 parquet / Dataset。
- ``POST /submit_sample``：每次只跑 **一条** rollout，结果写入缓存；凑满
  ``data.train_batch_size × actor_rollout_ref.rollout.n`` 条 **序列**（rollout 展开后的  ``DataProto`` 行数）后拼接并执行 **update**。若阈值仍用「prompt 条数」会在 ``n>1`` 时只切下前  ``train_batch_size`` 行，导致 jsonl 只有 32 条而非 32×n。
- ``POST /submit_batch``：兼容旧用法，一次请求 = 整批 rollout + update（与 dataloader 单步一致）。
- ``POST /save_checkpoint``、``GET /health``

环境变量：STREAMING_HOST、STREAMING_PORT
"""

from __future__ import annotations

import os
import sys
import threading

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
# Hydra 用 hydra.runtime.cwd 解析 verl 配置；未 cd 到仓库根时会导致找不到 ppo_trainer
try:
    os.chdir(_SCRIPT_DIR)
except OSError:
    pass
os.environ.setdefault("VERL_TRAINER_CONFIG_DIR", os.path.join(_SCRIPT_DIR, "verl", "trainer", "config"))
os.environ.setdefault("TTRL_ROOT", _SCRIPT_DIR)



import streaming_agent_urls
streaming_agent_urls.ensure_agent_service_urls()



def _create_trainer(config):
    """
    与 TaskRunner.run 到 init_workers 相同，但不调用 fit；无 train/val 数据集。
    """
    import socket
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.trainer.main_ppo import TaskRunnerSetup
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_processor, hf_tokenizer

    print(f"TTRL streaming server hostname: {socket.gethostname()}, PID: {os.getpid()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    runner = TaskRunnerSetup()
    actor_rollout_cls, ray_worker_group_cls = runner.add_actor_rollout_worker(config)
    runner.add_critic_worker(config)
    runner.add_reward_model_worker(config)
    runner.add_ref_policy_worker(config, actor_rollout_cls)

    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(runner.role_worker_mapping),
        use_critic=need_critic(config),
    )

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
    )
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    reward_fn = load_reward_manager(
        config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
    )
    val_reward_fn = load_reward_manager(
        config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
    )

    resource_pool_manager = runner.init_resource_pool_mgr(config)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=runner.role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=None,
        val_dataset=None,
        collate_fn=collate_fn,
        train_sampler=None,
    )
    trainer.init_workers()
    return trainer


def run_server(config):
    import ray
    from omegaconf import OmegaConf, open_dict

    from flask import Flask, jsonify, request
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    from streaming_rollout_buffer import StreamingRolloutBuffer

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    with open_dict(config.data):
        config.data.streaming_no_dataset = True

    trainer = _create_trainer(config)
    logger = trainer.prepare_streaming_training()

    host = os.environ.get("STREAMING_HOST", "0.0.0.0")
    port = int(os.environ.get("STREAMING_PORT", "8765"))

    train_bs = int(config.data.train_batch_size)
    rollout_n = int(config.actor_rollout_ref.rollout.get("n", 1))
    # 缓存的是 run_rollout_from_batch_dict 之后的 DataProto，行数 = prompt 条数 × rollout.n
    buffer_rows_per_update = train_bs * rollout_n
    rollout_buffer = StreamingRolloutBuffer(buffer_rows_per_update)

    def _safe_metrics(metrics: dict) -> dict:
        safe: dict = {}
        for k, v in metrics.items():
            try:
                if isinstance(v, (int, float)):
                    safe[k] = v
                elif hasattr(v, "item"):
                    safe[k] = float(v.item())
            except Exception:
                continue
        return safe

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "global_steps": trainer.global_steps,
                "total_training_steps": trainer.total_training_steps,
                "train_batch_size": train_bs,
                "rollout_n": rollout_n,
                # rollout_buffer_*：缓存中的序列行数（post-rollout，已含 n）；非 HTTP 次数
                "rollout_buffer_samples": len(rollout_buffer),
                "rollout_buffer_len": len(rollout_buffer),
            }
        )

    @app.route("/save_checkpoint", methods=["POST"])
    def save_checkpoint():
        # 流式路径在每次更新结束已对 trainer.global_steps +=1；目录名需与 fit() 一致（刚完成的步号）
        gs = trainer.global_steps
        ckpt_step = gs - 1
        if ckpt_step < 1:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "尚无已完成的训练步可保存（至少完成一次 GRPO 更新后再调用）",
                        "global_steps": gs,
                    }
                ),
                400,
            )
        trainer._save_checkpoint(checkpoint_global_step=ckpt_step)
        return jsonify({"ok": True, "checkpoint_global_step": ckpt_step, "global_steps": gs})

    @app.route("/submit_sample", methods=["POST"])
    def submit_sample():
        """
        单条 rollout：run_rollout_from_batch_dict → 缓存 append；
        累计序列行数达 train_batch_size×rollout.n 或 flush_partial 时 run_update_from_rollout_batch。
        """
        from streaming_batch_codec import batch_dict_from_jsonable

        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            body = {}
        raw = body.get("rollout")
        if not isinstance(raw, dict):
            return jsonify(
                {"error": 'JSON 需包含 "rollout": {...}（与 batch_dict_to_jsonable(collate_fn([sample])) 一致）'}
            ), 400
        log_epoch = body.get("epoch", 0)
        try:
            log_epoch = int(log_epoch)
        except (TypeError, ValueError):
            return jsonify({"error": "epoch 必须为整数"}), 400
        finish_raw = body.get("finish", False)
        if isinstance(finish_raw, str):
            finish_after_step = finish_raw.lower() in ("1", "true", "yes")
        else:
            finish_after_step = bool(finish_raw)
        flush_partial = body.get("flush_partial", False)
        if isinstance(flush_partial, str):
            flush_partial = flush_partial.lower() in ("1", "true", "yes")
        else:
            flush_partial = bool(flush_partial)

        try:
            batch_dict = batch_dict_from_jsonable(raw)
        except Exception as e:
            return jsonify({"error": f"rollout 解析失败: {e!s}"}), 400

        dp = trainer.run_rollout_from_batch_dict(batch_dict)

        to_update = rollout_buffer.append(dp)
        # 未满一批时，客户端可在最后一条请求上带 flush_partial 以更新尾部
        if to_update is None and flush_partial:
            to_update = rollout_buffer.flush_partial()

        if to_update is not None:
            metrics, done = trainer.run_update_from_rollout_batch(
                to_update,
                epoch=log_epoch,
                logger=logger,
                finish_after_step=finish_after_step,
            )
            return jsonify(
                {
                    "status": "step",
                    "metrics": _safe_metrics(metrics),
                    "global_steps": trainer.global_steps,
                    "done": done,
                    "epoch": log_epoch,
                }
            )

        return jsonify(
            {
                "status": "rollout",
                "global_steps": trainer.global_steps,
                "rollout_buffer_len": len(rollout_buffer),
                "epoch": log_epoch,
            }
        )

    @app.route("/submit_batch", methods=["POST"])
    def submit_batch():
        """整批一步：rollout + update（等价于 run_step_from_batch_dict）。"""
        from streaming_batch_codec import batch_dict_from_jsonable

        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            body = {}
        raw = body.get("batch")
        if not isinstance(raw, dict):
            return jsonify({"error": 'JSON 需为 {"batch": {...}}（batch_dict_to_jsonable 结果）'}), 400
        log_epoch = body.get("epoch", 0)
        try:
            log_epoch = int(log_epoch)
        except (TypeError, ValueError):
            return jsonify({"error": "epoch 必须为整数"}), 400
        try:
            batch_dict = batch_dict_from_jsonable(raw)
        except Exception as e:
            return jsonify({"error": f"batch 解析失败: {e!s}"}), 400
        finish_raw = body.get("finish", False)
        if isinstance(finish_raw, str):
            finish_after_step = finish_raw.lower() in ("1", "true", "yes")
        else:
            finish_after_step = bool(finish_raw)
        metrics, done = trainer.run_step_from_batch_dict(
            batch_dict,
            epoch=log_epoch,
            logger=logger,
            finish_after_step=finish_after_step,
        )
        return jsonify(
            {
                "status": "step",
                "metrics": _safe_metrics(metrics),
                "global_steps": trainer.global_steps,
                "done": done,
                "epoch": log_epoch,
            }
        )

    def _run():
        app.run(host=host, port=port, threaded=True, use_reloader=False)

    t = threading.Thread(target=_run, daemon=False, name="flask")
    t.start()
    print(f"Streaming TTRL (Flask) http://{host}:{port}")
    print(
        f"逐样本: POST /submit_sample（缓存 {buffer_rows_per_update} 条 rollout 序列 = "
        f"{train_bs} prompts × rollout.n={rollout_n} 后 update）；整批: POST /submit_batch"
    )
    print("接口: GET /health, POST /save_checkpoint, POST /submit_sample, POST /submit_batch")
    t.join()


def main():
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="./local_search/configs", config_name="mmsearch", version_base=None)
    def _main(config):
        OmegaConf.resolve(config)
        run_server(config)

    _main()


if __name__ == "__main__":
    main()
