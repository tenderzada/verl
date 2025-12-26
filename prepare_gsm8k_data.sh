#!/bin/bash
# GSM8K数据准备脚本

set -e

echo "正在准备GSM8K数据集..."

# 创建数据目录
mkdir -p ~/data/gsm8k
mkdir -p data/gsm8k

# 运行数据预处理脚本
python3 examples/data_preprocess/gsm8k.py \
    --local_save_dir ~/data/gsm8k

# 创建软链接到data/gsm8k（某些脚本可能使用相对路径）
ln -sf ~/data/gsm8k/train.parquet data/gsm8k/train.parquet
ln -sf ~/data/gsm8k/test.parquet data/gsm8k/test.parquet

echo "✓ GSM8K数据集准备完成！"
echo "  - 训练集: ~/data/gsm8k/train.parquet"
echo "  - 测试集: ~/data/gsm8k/test.parquet"
