#!/bin/bash
# GRPO训练脚本 - 解决HF验证问题的包装器

set -e

echo "=================================="
echo "  Qwen2.5-1.5B GRPO训练"
echo "=================================="
echo ""

# ============ 关键环境变量 ============
# 完全禁用HuggingFace Hub验证
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DISABLE_TELEMETRY=1

# 禁用符号链接警告
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# 强制使用本地文件
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "环境变量已设置:"
echo "  HF_HUB_OFFLINE=1"
echo "  TRANSFORMERS_OFFLINE=1"
echo ""

# ============ 路径检查 ============
MODEL_PATH="/mnt/data/Qwen2.5-1.5B-Instruct"
TRAIN_DATA="/mnt/data/GSM8K/train.parquet"
VAL_DATA="/mnt/data/GSM8K/test.parquet"

echo "检查路径..."
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型目录不存在: $MODEL_PATH"
    exit 1
fi
echo "✓ 模型路径: $MODEL_PATH"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误: 训练数据不存在: $TRAIN_DATA"
    exit 1
fi
echo "✓ 训练数据: $TRAIN_DATA"

if [ ! -f "$VAL_DATA" ]; then
    echo "错误: 验证数据不存在: $VAL_DATA"
    exit 1
fi
echo "✓ 验证数据: $VAL_DATA"

echo ""
echo "开始训练..."
echo "=================================="
echo ""

set -x

# 运行训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_qwen2.5_1.5b' \
    trainer.experiment_name='grpo_lora_local' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 "$@"
