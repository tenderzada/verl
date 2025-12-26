#!/bin/bash
# Qwen2.5-1.5B GRPO训练脚本（修改自官方示例）
# 适配本地模型路径: /mnt/data/Qwen2.5-1.5B-Instruct

# ============ 路径配置 ============
# 模型路径（本地）- 修改为你的实际路径
MODEL_PATH=/mnt/data/Qwen2.5-1.5B-Instruct

# 数据路径 - 使用处理后的数据
TRAIN_DATA=$HOME/data/gsm8k/train.parquet
VAL_DATA=$HOME/data/gsm8k/test.parquet

# 检查路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型目录不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
    echo "错误: 数据文件不存在，请先运行: bash prepare_gsm8k_data_custom.sh"
    exit 1
fi

# ============ 环境变量 ============
# 禁用HF Hub在线验证（重要！）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# GPU配置 - 根据你的实际GPU数量调整
export CUDA_VISIBLE_DEVICES=0

# Wandb配置
NOW=$(date +%Y%m%d_%H%M%S)
export WANDB_DIR=gsm8k-grpo-lora-qwen2.5-1.5b-${NOW}
export WANDB_PROJECT=qwen2.5-1.5b-gsm8k-grpo
export WANDB_EXP=1.5b-${NOW}

# ============ 训练配置 ============
# 根据GPU显存调整这些参数
# 默认配置适合单个16GB GPU
nproc_per_gpu=128
nnodes=1
ngpu_per_node=1
total_procs=$(( nproc_per_gpu * nnodes * ngpu_per_node ))
mini_batch_size=$(( total_procs ))

echo "=================================="
echo "  训练配置"
echo "=================================="
echo "模型路径: $MODEL_PATH"
echo "训练数据: $TRAIN_DATA"
echo "验证数据: $VAL_DATA"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Batch size: $total_procs"
echo "=================================="
echo ""

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=${total_procs} \
    data.val_batch_size=${total_procs} \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.model.use_shm=True  \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@ 2>&1 | tee training_${NOW}.log

echo ""
echo "=================================="
echo "训练完成！日志: training_${NOW}.log"
echo "=================================="
