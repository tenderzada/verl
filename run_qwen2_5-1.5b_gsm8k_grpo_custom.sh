#!/bin/bash
# Qwen2.5-1.5B GSM8K GRPO训练脚本
#
# 模型路径: /mnt/data/Qwen2.5-1.5B-Instruct
# 数据路径: /mnt/data/GSM8K (处理后存储在 ~/data/gsm8k/)
#
# 针对有限计算资源优化

set -x

# ============ 路径配置 ============
# 模型路径（本地）
MODEL_PATH=/mnt/data/Qwen2.5-1.5B-Instruct

# 数据路径（处理后的parquet文件）
TRAIN_DATA=/mnt/data/GSM8K/train.parquet
VAL_DATA=/mnt/data/GSM8K/test.parquet

# ============ 环境检查 ============
# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型目录不存在: $MODEL_PATH"
    echo "请确认模型已下载到该路径"
    exit 1
fi

# 检查数据是否存在
if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
    echo "错误: 数据文件不存在"
    echo "请先运行: bash prepare_gsm8k_data_custom.sh"
    exit 1
fi

echo "✓ 模型路径: $MODEL_PATH"
echo "✓ 训练数据: $TRAIN_DATA"
echo "✓ 验证数据: $VAL_DATA"
echo ""

# ============ GPU配置 ============
# 根据你的GPU数量调整
# 单GPU: export CUDA_VISIBLE_DEVICES=0
# 双GPU: export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0

# GPU数量
NUM_GPUS=1

# ============ Wandb配置 ============
NOW=$(date +%Y%m%d_%H%M%S)
export WANDB_PROJECT=qwen2.5-1.5b-gsm8k-grpo
export WANDB_NAME=grpo-lora-${NOW}

# 如果不想使用wandb，取消下面这行的注释
# export WANDB_MODE=disabled

# ============ 训练超参数 ============
# 核心参数 - 根据GPU显存调整
TRAIN_BATCH_SIZE=8            # 每轮训练的prompt数量
ROLLOUT_N=5                   # 每个prompt采样的响应数量（GRPO核心参数，建议>=5）
MINI_BATCH_SIZE=8             # PPO mini batch大小
TOTAL_EPOCHS=15               # 总训练轮数

# LoRA配置
LORA_RANK=32                  # LoRA秩（可选: 16, 32, 64）
LORA_ALPHA=32                 # LoRA alpha（通常等于或2倍rank）

# 学习率
LEARNING_RATE=3e-5            # 初始学习率

# 显存优化配置
GPU_MEMORY_UTIL=0.4           # vLLM使用的GPU显存比例（0.3-0.6）
PARAM_OFFLOAD=True            # 参数卸载到CPU
OPTIMIZER_OFFLOAD=True        # 优化器卸载到CPU

# ============ 显存配置建议 ============
#
# 如果你有单个16GB GPU:
#   TRAIN_BATCH_SIZE=4
#   ROLLOUT_N=4
#   MINI_BATCH_SIZE=4
#   GPU_MEMORY_UTIL=0.4
#   PARAM_OFFLOAD=True
#   OPTIMIZER_OFFLOAD=True
#
# 如果你有单个24GB+ GPU:
#   TRAIN_BATCH_SIZE=16
#   ROLLOUT_N=5
#   MINI_BATCH_SIZE=16
#   GPU_MEMORY_UTIL=0.5
#   PARAM_OFFLOAD=False
#   OPTIMIZER_OFFLOAD=False
#
# 如果你有双GPU (每个16GB+):
#   export CUDA_VISIBLE_DEVICES=0,1
#   NUM_GPUS=2
#   TRAIN_BATCH_SIZE=16
#   ROLLOUT_N=5
#   MINI_BATCH_SIZE=16
#
# ============================================

echo "训练配置："
echo "  Batch Size: $TRAIN_BATCH_SIZE"
echo "  Rollout N: $ROLLOUT_N"
echo "  Mini Batch Size: $MINI_BATCH_SIZE"
echo "  Total Epochs: $TOTAL_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  LoRA Rank: $LORA_RANK"
echo ""

# ============ 启动训练 ============

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL} \
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

echo ""
echo "=================================="
echo "训练完成！"
echo "=================================="
echo ""
echo "日志文件: training_${NOW}.log"
echo ""
echo "Checkpoint保存在:"
echo "  checkpoints/${WANDB_PROJECT}/${WANDB_NAME}/"
echo ""
