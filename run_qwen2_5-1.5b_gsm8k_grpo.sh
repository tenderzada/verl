#!/bin/bash
# Qwen2.5-1.5B GSM8K GRPO训练脚本
# 针对有限计算资源优化

set -x

# ============ 配置区域 ============
# 模型路径（本地）
MODEL_PATH=/mnt/data/Qwen2.5-1.5B-Instruct

# GPU配置 - 根据你的GPU数量调整
# 如果只有1个GPU，设置为0；如果有2个GPU，设置为0,1
export CUDA_VISIBLE_DEVICES=0

# Wandb配置（可选）
NOW=$(date +%Y%m%d_%H%M%S)
export WANDB_PROJECT=qwen2.5-1.5b-gsm8k-grpo
export WANDB_NAME=grpo-lora-${NOW}

# 训练参数
NUM_GPUS=1                    # GPU数量
TRAIN_BATCH_SIZE=8            # 每轮训练的prompt数量（资源有限时可以设为4或8）
ROLLOUT_N=5                   # 每个prompt采样的响应数量（GRPO关键参数）
MINI_BATCH_SIZE=8             # PPO mini batch大小
TOTAL_EPOCHS=15               # 总训练轮数

# LoRA配置
LORA_RANK=32                  # LoRA秩（32或64）
LORA_ALPHA=32                 # LoRA alpha

# 学习率
LEARNING_RATE=3e-5            # 对于1.5B模型，3e-5是一个较好的起点

# ============ 启动训练 ============

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    data.train_files=~/data/gsm8k/train.parquet \
    data.val_files=~/data/gsm8k/test.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.model.target_modules=all-linear \
    \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_NAME} \
    trainer.val_before_train=False \
    $@ 2>&1 | tee training_${NOW}.log

echo "训练完成！日志保存在 training_${NOW}.log"
