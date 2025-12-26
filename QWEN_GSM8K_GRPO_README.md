# Qwen2.5-1.5B GSM8K GRPO 训练指南

本指南将帮助你使用verl库在GSM8K数据集上训练Qwen2.5-1.5B模型，采用GRPO（Group Relative Policy Optimization）算法。

## 🎯 训练概览

- **模型**: Qwen2.5-1.5B-Instruct (本地路径: `/mnt/data/Qwen2.5-1.5B-Instruct`)
- **数据集**: GSM8K (数学推理任务)
- **算法**: GRPO (无需Critic的强化学习算法)
- **训练方式**: LoRA微调 (节省显存)

## 📦 环境要求

### 最低配置
- **GPU**: 1张 GPU (16GB+ 显存，如RTX 3090, A100等)
- **内存**: 32GB+ RAM
- **存储**: 20GB+ 可用空间

### 推荐配置
- **GPU**: 2张 GPU (16GB+ 显存)
- **内存**: 64GB+ RAM

## 🚀 快速开始

### 1. 准备数据集

```bash
# 进入verl目录
cd /home/user/verl

# 运行数据准备脚本
bash prepare_gsm8k_data.sh
```

这个脚本会：
- 从Hugging Face下载GSM8K数据集
- 转换为parquet格式
- 保存到 `~/data/gsm8k/` 目录

### 2. 检查模型文件

确认模型已经下载到本地：

```bash
ls -lh /mnt/data/Qwen2.5-1.5B-Instruct/
```

你应该能看到类似以下文件：
```
config.json
model.safetensors (或 pytorch_model.bin)
tokenizer.json
tokenizer_config.json
...
```

### 3. 启动训练

```bash
# 添加执行权限
chmod +x run_qwen2_5-1.5b_gsm8k_grpo.sh

# 开始训练
bash run_qwen2_5-1.5b_gsm8k_grpo.sh
```

训练日志会自动保存到 `training_<时间戳>.log` 文件。

## ⚙️ 核心参数解释

### GRPO关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm.adv_estimator` | grpo | 使用GRPO优势估计器 |
| `rollout.n` | 5 | **核心参数**：每个问题采样5个答案进行对比学习 |
| `actor.use_kl_loss` | True | 使用KL散度约束，防止模型偏离太远 |
| `actor.kl_loss_coef` | 0.001 | KL损失系数 |

### 资源控制参数

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `data.train_batch_size` | 8 | 每轮训练的问题数量 | 显存不足时减少到4 |
| `rollout.gpu_memory_utilization` | 0.4 | vLLM使用的GPU显存比例 | 单GPU可调整到0.5-0.6 |
| `actor.fsdp_config.param_offload` | True | 参数卸载到CPU | 显存紧张时保持True |
| `actor.fsdp_config.optimizer_offload` | True | 优化器状态卸载到CPU | 显存紧张时保持True |

### LoRA参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.lora_rank` | 32 | LoRA矩阵的秩，越大效果越好但显存占用越多 |
| `model.lora_alpha` | 32 | LoRA缩放因子，通常设置为rank的1-2倍 |
| `model.target_modules` | all-linear | 应用LoRA的层，all-linear表示所有线性层 |

### 学习率与训练

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `actor.optim.lr` | 3e-5 | 学习率 |
| `trainer.total_epochs` | 15 | 总训练轮数 |
| `trainer.save_freq` | 5 | 每5个epoch保存一次checkpoint |
| `trainer.test_freq` | 5 | 每5个epoch在测试集上评估 |

## 🔧 根据你的GPU调整配置

### 单个16GB GPU (如RTX 3090)

```bash
# 修改训练脚本中的以下参数：
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
TRAIN_BATCH_SIZE=4              # 减小batch size
MINI_BATCH_SIZE=4
ROLLOUT_N=4                     # 减少采样数量

# gpu_memory_utilization可以适当提高
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

### 双GPU (每个16GB+)

```bash
# 修改训练脚本中的以下参数：
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
TRAIN_BATCH_SIZE=16             # 可以增大batch size
MINI_BATCH_SIZE=16
ROLLOUT_N=5                     # 保持5个采样
```

### 单个24GB+ GPU (如RTX 4090, A100)

```bash
# 修改训练脚本中的以下参数：
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
TRAIN_BATCH_SIZE=16
MINI_BATCH_SIZE=16
ROLLOUT_N=8                     # 可以增加到8个采样

# gpu_memory_utilization可以提高
actor_rollout_ref.rollout.gpu_memory_utilization=0.6

# 可以关闭offload提升速度
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
```

## 📊 训练监控

### 使用Weights & Biases (推荐)

1. 安装wandb：
```bash
pip install wandb
```

2. 登录wandb：
```bash
wandb login
```

3. 训练时会自动上传数据到wandb，你可以在浏览器中实时查看：
   - 训练损失曲线
   - 奖励分数变化
   - GPU使用率
   - 等等

### 查看日志文件

```bash
# 实时查看训练日志
tail -f training_<时间戳>.log

# 搜索关键指标
grep "reward" training_<时间戳>.log
grep "loss" training_<时间戳>.log
```

## 💾 Checkpoint位置

训练的checkpoint默认保存在：
```
/home/user/verl/checkpoints/<project_name>/<experiment_name>/
```

每个checkpoint包含：
- LoRA权重
- 优化器状态
- 训练配置

## 🎓 GRPO算法原理

GRPO是一种简化的强化学习算法，相比PPO的优势：

1. **无需Critic网络**：不需要训练一个单独的价值函数网络，节省显存和计算
2. **组内对比学习**：对同一个问题生成多个答案（n=5），通过对比学习
3. **相对奖励**：每个答案的奖励是相对于该组平均奖励的，而不是绝对值

### GRPO工作流程

```
1. 采样阶段：
   问题1 -> [答案1.1, 答案1.2, 答案1.3, 答案1.4, 答案1.5]
   问题2 -> [答案2.1, 答案2.2, 答案2.3, 答案2.4, 答案2.5]
   ...

2. 评估阶段：
   每个答案获得奖励分数（GSM8K中，答案正确=1，错误=0）

3. 计算优势：
   对于问题1的5个答案，计算它们的平均分
   每个答案的优势 = (该答案分数 - 平均分) / std

4. 策略更新：
   高于平均分的答案 -> 增加生成概率
   低于平均分的答案 -> 降低生成概率
```

## 🐛 常见问题

### 1. CUDA Out of Memory

**解决方案**：
- 减小 `TRAIN_BATCH_SIZE` (如从8降到4)
- 减小 `ROLLOUT_N` (如从5降到3)
- 启用 `param_offload` 和 `optimizer_offload`
- 降低 `gpu_memory_utilization`

### 2. 训练速度太慢

**解决方案**：
- 如果显存充足，增加 `gpu_memory_utilization`
- 关闭offload: `param_offload=False`, `optimizer_offload=False`
- 减少 `max_response_length` (如从1024降到512)

### 3. 模型路径找不到

**检查**：
```bash
# 确认模型路径正确
ls /mnt/data/Qwen2.5-1.5B-Instruct/

# 如果路径不对，修改脚本中的MODEL_PATH变量
```

### 4. 数据集下载失败

**解决方案**：
- 检查网络连接
- 使用镜像站点：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

### 5. 想要使用全量微调而不是LoRA

修改训练脚本，移除LoRA相关参数：
```bash
# 注释掉或删除这些行：
# actor_rollout_ref.model.lora_rank=${LORA_RANK} \
# actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
# actor_rollout_ref.model.target_modules=all-linear \
```

注意：全量微调需要更多显存（约3-4倍）

## 📚 进阶配置

### 多轮对话训练

如果想要训练多轮对话的数学推理，可以参考：
```bash
examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

### 自定义奖励函数

GSM8K的奖励函数定义在：
```
verl/utils/reward_score/gsm8k.py
```

### 使用自己的数据集

参考 `examples/data_preprocess/gsm8k.py` 创建自己的数据预处理脚本。

数据格式要求：
```python
{
    "prompt": [{"role": "user", "content": "问题"}],
    "reward_model": {"style": "rule", "ground_truth": "答案"},
    ...
}
```

## 📖 相关资源

- **GRPO原始论文**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/pdf/2402.03300)
- **verl文档**: https://verl.readthedocs.io/
- **GRPO算法文档**: `/home/user/verl/docs/algo/grpo.md`
- **更多示例**: `/home/user/verl/examples/grpo_trainer/`

## ✅ 训练完成后

### 评估模型

训练脚本会自动在验证集上评估，查看日志中的：
- `test/reward_mean`: 平均奖励分数
- `test/accuracy`: 准确率（如果适用）

### 导出模型

训练完成后，可以合并LoRA权重：
```bash
python3 -m verl.utils.merge_lora \
    --base_model /mnt/data/Qwen2.5-1.5B-Instruct \
    --lora_model checkpoints/<your_checkpoint> \
    --output_model merged_model
```

### 测试模型

使用vLLM进行推理：
```python
from vllm import LLM, SamplingParams

llm = LLM(model="merged_model")
prompts = ["Solve: 1+1=?"]
outputs = llm.generate(prompts, SamplingParams(temperature=0))
print(outputs[0].outputs[0].text)
```

---

祝训练顺利！如有问题，请查看 https://github.com/volcengine/verl/issues
