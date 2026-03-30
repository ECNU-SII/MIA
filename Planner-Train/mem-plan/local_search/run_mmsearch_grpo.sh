#!/bin/bash

set -x


export WANDB_API_KEY=""

export JUDGE_URL="JUDGE_URL/8002/v1"
export PLAN_URL="EXECUTOR_SERVE/5000/plan"
export REPLAN_URL="EXECUTOR_SERVE/5000/replan"


export WANDB_MODE="offline"
export MODEL_MAX_LEN=32786
PROJECT_NAME="deepreseach"
EXPERIMENT_NAME=""


SAVE_CHECKPOINT_DIR=/your_path/mem-plan/verl_checkpoints

DATASET_TRAIN=/your_path/datasets/fvqa_matpo_train_planer.parquet
DATASET_VAL=/your_path/datasets/gaia_fvqa_merged.parquet
REF_MODEL_PATH=/your_path/model/Qwen3-8B/Qwen/Qwen3-8B

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=/your_path/local_search/configs \
    --config-name='mmsearch' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=[${DATASET_VAL}] \
    data.train_batch_size=128 \
    data.max_prompt_length=24576 \
    data.max_response_length=8192 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=2 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192 \
    trainer.critic_warmup=0 \
    trainer.rollout_data_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/roolout_saved \
    trainer.validation_data_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/val_saved \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=5 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
