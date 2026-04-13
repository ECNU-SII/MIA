# Copyright 2025 the MIA / TTRL authors
#
# 流式 TTRL 拆分为两阶段（便于「逐样本 rollout + 按批 update」）：
# - rollout：生成 → 奖励/旧 logprob/ref/价值 → token 级奖励（adv 前准备）→ 环境/记忆 HTTP 钩子
#   不含 compute_advantage（GRPO 等需要整批统计）。
# - update：compute_advantage → 更新 critic/actor → 日志与 global_steps。
#
# run_single_ppo_training_step：整批一步，内部依次调用上述两阶段（与 dataloader 单步一致）。

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import ray
import torch

from verl import DataProto



def _coerce_seconds(v: Any) -> float | None:
    """Best-effort float for timer values (Python scalars or numpy)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item") and callable(getattr(v, "item")):
        try:
            return float(v.item())
        except (TypeError, ValueError):
            pass
    return None


def _ensure_step_in_timing_raw(timing_raw: dict) -> None:
    """
    Outer ``with marked_timer("step")`` writes ``timing_raw["step"]`` only after the block exits.
    When update runs inside that block (streaming HTTP path), ``step`` is missing until then.
    Approximate with the sum of already-recorded sub-timers so metrics/throughput do not KeyError.
    """
    if "step" in timing_raw:
        return
    total = 0.0
    for k, v in timing_raw.items():
        if k == "step":
            continue
        x = _coerce_seconds(v)
        if x is not None:
            total += x
    timing_raw["step"] = total


def run_streaming_rollout_phase(
    trainer: Any,
    batch_dict: dict,
    metrics: dict,
    timing_raw: dict,
) -> DataProto:
    """
    单批/单样本 rollout：从 collated batch_dict 做到「更新前」所需张量 + 钩子，返回 DataProto。
    调用方可将多个结果 concat 后再 run_streaming_update_phase。
    """
    from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
    from verl.trainer.ppo.ray_trainer import (
        apply_kl_penalty,
        batch_evaluate_url,
        compute_response_mask,
        consolidate_memories_url,
        get_user_response_from_url,
        save_memory_url,
    )
    from verl.trainer.ppo.reward import compute_reward, compute_reward_async
    from verl.utils.debug import marked_timer

    batch: DataProto = DataProto.from_single_dict(batch_dict)
    batch.non_tensor_batch["uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
    )

    gen_batch = trainer._get_gen_batch(batch)
    gen_batch.meta_info["global_steps"] = trainer.global_steps
    gen_batch = gen_batch.repeat(repeat_times=trainer.config.actor_rollout_ref.rollout.n, interleave=True)

    with marked_timer("gen", timing_raw, color="red"):
        if not trainer.async_rollout_mode:
            gen_batch_output = trainer.actor_rollout_wg.generate_sequences(gen_batch)
        else:
            gen_batch_output = trainer.async_rollout_manager.generate_sequences(gen_batch)
        timing_raw.update(gen_batch_output.meta_info["timing"])
        gen_batch_output.meta_info.pop("timing", None)

    if trainer.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
        if trainer.reward_fn is None:
            raise ValueError("A reward_fn is required for REMAX advantage estimation.")
        with marked_timer("gen_max", timing_raw, color="purple"):
            gen_baseline_batch = deepcopy(gen_batch)
            gen_baseline_batch.meta_info["do_sample"] = False
            if not trainer.async_rollout_mode:
                gen_baseline_output = trainer.actor_rollout_wg.generate_sequences(gen_baseline_batch)
            else:
                gen_baseline_output = trainer.async_rollout_manager.generate_sequences(gen_baseline_batch)
            batch = batch.union(gen_baseline_output)
            reward_baseline_tensor = trainer.reward_fn(batch)
            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
            batch.batch["reward_baselines"] = reward_baseline_tensor
            del gen_baseline_batch, gen_baseline_output

    batch = batch.repeat(repeat_times=trainer.config.actor_rollout_ref.rollout.n, interleave=True)
    batch = batch.union(gen_batch_output)

    if "response_mask" not in batch.batch.keys():
        batch.batch["response_mask"] = compute_response_mask(batch)
    if trainer.config.trainer.balance_batch:
        trainer._balance_batch(batch, metrics=metrics)

    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

    reward_extra_infos_dict: dict = {}
    future_reward = None
    with marked_timer("reward", timing_raw, color="yellow"):
        if trainer.use_rm and "rm_scores" not in batch.batch.keys():
            reward_tensor = trainer.rm_wg.compute_rm_score(batch)
            batch = batch.union(reward_tensor)

        if trainer.config.reward_model.launch_reward_fn_async:
            future_reward = compute_reward_async.remote(data=batch, reward_fn=trainer.reward_fn)
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, trainer.reward_fn)

    with marked_timer("old_log_prob", timing_raw, color="blue"):
        old_log_prob = trainer.actor_rollout_wg.compute_log_prob(batch)
        entropys = old_log_prob.batch["entropys"]
        response_masks = batch.batch["response_mask"]
        loss_agg_mode = trainer.config.actor_rollout_ref.actor.loss_agg_mode
        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
        metrics.update({"actor/entropy": entropy_agg.detach().item()})
        old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)

        if "rollout_log_probs" in batch.batch.keys():
            from verl.utils.debug.metrics import calculate_debug_metrics

            metrics.update(calculate_debug_metrics(batch))

    if trainer.use_reference_policy:
        with marked_timer("ref", timing_raw, color="olive"):
            if not trainer.ref_in_actor:
                ref_log_prob = trainer.ref_policy_wg.compute_ref_log_prob(batch)
            else:
                ref_log_prob = trainer.actor_rollout_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

    if trainer.use_critic:
        with marked_timer("values", timing_raw, color="cyan"):
            values = trainer.critic_wg.compute_values(batch)
            batch = batch.union(values)

    # token 级分数与（可选）KL：供后续 compute_advantage 使用；此处不做优势估计
    with marked_timer("adv_prep", timing_raw, color="brown"):
        if trainer.config.reward_model.launch_reward_fn_async:
            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
        batch.batch["token_level_scores"] = reward_tensor

        if reward_extra_infos_dict:
            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

        if trainer.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(
                batch, kl_ctrl=trainer.kl_ctrl_in_reward, kl_penalty=trainer.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

    # 与评估/记忆服务交互（仍属 rollout 语义，不依赖整批优势）
    get_user_response_from_url(batch_evaluate_url, [])
    get_user_response_from_url(consolidate_memories_url, [])
    get_user_response_from_url(save_memory_url, [])

    return batch


def run_streaming_update_phase(
    trainer: Any,
    batch: DataProto,
    epoch: int,
    logger: Optional[Any],
    prev_step_profile: bool,
    curr_step_profile: bool,
    finish_after_step: bool,
    metrics: dict,
    timing_raw: dict,
) -> tuple[dict, bool, bool, bool]:
    """
    拼接批上的优势估计 + 策略更新 + 日志；并递增 global_steps。
    调用方通常用 marked_timer(\"step\") 包住 rollout+update 或仅包住本函数。
    """
    from verl.trainer.ppo.ray_trainer import compute_advantage
    from verl.experimental.dataset.sampler import AbstractCurriculumSampler
    from verl.trainer.ppo.metric_utils import (
        compute_data_metrics,
        compute_throughout_metrics,
        compute_timing_metrics,
    )
    from verl.utils.debug import marked_timer
    from verl.utils.metric import reduce_metrics

    is_last_step = trainer.total_training_steps is not None and trainer.global_steps >= trainer.total_training_steps

    with marked_timer("adv", timing_raw, color="brown"):
        norm_adv_by_std_in_grpo = trainer.config.algorithm.get("norm_adv_by_std_in_grpo", True)
        batch = compute_advantage(
            batch,
            adv_estimator=trainer.config.algorithm.adv_estimator,
            gamma=trainer.config.algorithm.gamma,
            lam=trainer.config.algorithm.lam,
            num_repeat=trainer.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=trainer.config.algorithm,
        )

    if trainer.use_critic:
        with marked_timer("update_critic", timing_raw, color="pink"):
            critic_output = trainer.critic_wg.update_critic(batch)
        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

    if trainer.config.trainer.critic_warmup <= trainer.global_steps:
        with marked_timer("update_actor", timing_raw, color="red"):
            batch.meta_info["multi_turn"] = trainer.config.actor_rollout_ref.rollout.multi_turn.enable
            actor_output = trainer.actor_rollout_wg.update_actor(batch)
        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

    rollout_data_dir = trainer.config.trainer.get("rollout_data_dir", None)
    if rollout_data_dir:
        reward_extra_infos_dict: dict = {}
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = trainer.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = trainer.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch
            ]
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            trainer._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=rollout_data_dir,
            )

    with marked_timer("stop_profile", timing_raw):
        next_step_profile = (
            trainer.global_steps + 1 in trainer.config.global_profiler.steps
            if trainer.config.global_profiler.steps is not None
            else False
        )
        trainer._stop_profiling(
            curr_step_profile and not next_step_profile
            if trainer.config.global_profiler.profile_continuous_steps
            else curr_step_profile
        )
        new_prev_step_profile = curr_step_profile
        new_curr_step_profile = next_step_profile

    _ensure_step_in_timing_raw(timing_raw)

    steps_duration = timing_raw["step"]
    trainer.max_steps_duration = max(trainer.max_steps_duration, steps_duration)

    metrics.update(
        {
            "training/global_step": trainer.global_steps,
            "training/epoch": epoch,
        }
    )
    metrics.update(compute_data_metrics(batch=batch, use_critic=trainer.use_critic))
    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
    n_gpus = trainer.resource_pool_manager.get_n_gpus()
    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

    if (
        getattr(trainer, "train_dataloader", None) is not None
        and isinstance(trainer.train_dataloader.sampler, AbstractCurriculumSampler)
    ):
        trainer.train_dataloader.sampler.update(batch=batch)

    if logger is not None:
        logger.log(data=metrics, step=trainer.global_steps)

    trainer.global_steps += 1

    if (
        hasattr(trainer.config.actor_rollout_ref.actor, "profiler")
        and trainer.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
    ):
        trainer.actor_rollout_wg.dump_memory_snapshot(
            tag=f"post_update_step{trainer.global_steps}", sub_dir=f"step{trainer.global_steps}"
        )

    should_stop = finish_after_step

    if not is_last_step and hasattr(trainer.train_dataset, "on_batch_end"):
        trainer.train_dataset.on_batch_end(batch=batch)

    return metrics, should_stop, new_prev_step_profile, new_curr_step_profile


def run_single_ppo_training_step(
    trainer: Any,
    batch_dict: dict,
    epoch: int,
    logger: Optional[Any],
    prev_step_profile: bool,
    curr_step_profile: bool,
    finish_after_step: bool = False,
) -> tuple[dict, bool, bool, bool]:
    """
    与训练 dataloader 单步一致：整批先 rollout，再 update（内部仍拆成两阶段函数）。
    """
    from verl.utils.debug import marked_timer

    metrics: dict = {}
    timing_raw: dict = {}

    with marked_timer("start_profile", timing_raw):
        trainer._start_profiling(
            not prev_step_profile and curr_step_profile
            if trainer.config.global_profiler.profile_continuous_steps
            else curr_step_profile
        )

    with marked_timer("step", timing_raw):
        batch = run_streaming_rollout_phase(trainer, batch_dict, metrics, timing_raw)
        return run_streaming_update_phase(
            trainer,
            batch,
            epoch,
            logger,
            prev_step_profile,
            curr_step_profile,
            finish_after_step,
            metrics,
            timing_raw,
        )
