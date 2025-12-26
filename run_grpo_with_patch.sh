#!/bin/bash
# 使用补丁运行GRPO训练 - 解决HF验证错误

set -e

echo "=================================="
echo "  使用补丁运行GRPO训练"
echo "=================================="
echo ""

# 设置环境变量
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/user/verl:$PYTHONPATH

# 路径配置
MODEL_PATH="/mnt/data/Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_DATA="/mnt/data/GSM8K/train.parquet"
VAL_DATA="/mnt/data/GSM8K/test.parquet"

# 检查路径
echo "检查路径..."
[ -d "$MODEL_PATH" ] || { echo "错误: 模型不存在"; exit 1; }
[ -f "$TRAIN_DATA" ] || { echo "错误: 训练数据不存在"; exit 1; }
[ -f "$VAL_DATA" ] || { echo "错误: 验证数据不存在"; exit 1; }
echo "✓ 所有路径检查通过"
echo ""

# 首先尝试加载补丁
echo "加载HuggingFace验证补丁..."
python3 -c "
import os
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

try:
    from huggingface_hub.utils import _validators
    original_validate = _validators.validate_repo_id

    def patched_validate(repo_id, *args, **kwargs):
        if os.path.isabs(repo_id):
            return
        return original_validate(repo_id, *args, **kwargs)

    _validators.validate_repo_id = patched_validate
    print('✓ 补丁加载成功')
except Exception as e:
    print(f'警告: 补丁加载失败 ({e})')
"

echo ""
echo "开始训练..."
echo "=================================="
echo ""

set -x

# 使用Python直接运行，在启动时应用补丁
python3 << 'EOF'
import os
import sys

# 设置环境变量
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 应用补丁
try:
    from huggingface_hub.utils import _validators
    _original_validate_repo_id = _validators.validate_repo_id

    def patched_validate_repo_id(repo_id, *args, **kwargs):
        # 如果是绝对路径，直接返回
        if os.path.isabs(repo_id):
            return
        # 否则调用原始验证
        return _original_validate_repo_id(repo_id, *args, **kwargs)

    _validators.validate_repo_id = patched_validate_repo_id
    print("[Patch] HuggingFace validation patched successfully!")
except Exception as e:
    print(f"[Patch] Warning: {e}")

# 现在运行训练
from verl.trainer.main_ppo import main
sys.argv = [
    'main_ppo',
    'algorithm.adv_estimator=grpo',
    'trainer.val_before_train=False',
    'data.train_files=/mnt/data/GSM8K/train.parquet',
    'data.val_files=/mnt/data/GSM8K/test.parquet',
    'data.train_batch_size=16',
    'data.max_prompt_length=512',
    'data.max_response_length=1024',
    'data.filter_overlong_prompts=True',
    'data.truncation=error',
    'data.shuffle=False',
    'actor_rollout_ref.model.path=/mnt/data/Qwen/Qwen2.5-1.5B-Instruct',
    'actor_rollout_ref.model.trust_remote_code=True',
    'actor_rollout_ref.model.lora_rank=64',
    'actor_rollout_ref.model.lora_alpha=32',
    'actor_rollout_ref.actor.optim.lr=3e-6',
    'actor_rollout_ref.model.use_remove_padding=True',
    'actor_rollout_ref.actor.ppo_mini_batch_size=16',
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40',
    'actor_rollout_ref.actor.use_kl_loss=True',
    'actor_rollout_ref.actor.kl_loss_coef=0.001',
    'actor_rollout_ref.actor.kl_loss_type=low_var_kl',
    'actor_rollout_ref.actor.entropy_coeff=0',
    'actor_rollout_ref.model.enable_gradient_checkpointing=True',
    'actor_rollout_ref.actor.fsdp_config.param_offload=False',
    'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False',
    'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40',
    'actor_rollout_ref.rollout.tensor_model_parallel_size=2',
    'actor_rollout_ref.rollout.name=vllm',
    'actor_rollout_ref.rollout.gpu_memory_utilization=0.6',
    'actor_rollout_ref.rollout.n=5',
    'actor_rollout_ref.rollout.load_format=safetensors',
    'actor_rollout_ref.rollout.layered_summon=True',
    'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40',
    'actor_rollout_ref.ref.fsdp_config.param_offload=True',
    'algorithm.use_kl_in_reward=False',
    'trainer.critic_warmup=0',
    'trainer.logger=["console","wandb"]',
    'trainer.project_name=verl_grpo_qwen2.5_1.5b',
    'trainer.experiment_name=grpo_lora_patched',
    'trainer.n_gpus_per_node=2',
    'trainer.nnodes=1',
    'trainer.save_freq=20',
    'trainer.test_freq=5',
    'trainer.total_epochs=15',
]

main()
EOF
