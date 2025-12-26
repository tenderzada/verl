#!/bin/bash
# GSM8K数据准备脚本 - 适配 /mnt/data/GSM8K
# 支持多种输入格式

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=================================="
echo "  GSM8K数据准备"
echo "=================================="
echo ""

# 数据路径
SOURCE_DATA_DIR="/mnt/data/GSM8K"
TARGET_DATA_DIR="$HOME/data/gsm8k"

echo -e "${YELLOW}源数据目录: ${SOURCE_DATA_DIR}${NC}"
echo -e "${YELLOW}目标数据目录: ${TARGET_DATA_DIR}${NC}"
echo ""

# 创建目标目录
mkdir -p "$TARGET_DATA_DIR"
mkdir -p data/gsm8k

# 检查源数据目录
if [ ! -d "$SOURCE_DATA_DIR" ]; then
    echo -e "${RED}错误: 源数据目录不存在: ${SOURCE_DATA_DIR}${NC}"
    echo ""
    echo "请选择以下选项之一："
    echo "1. 如果你还没有下载GSM8K数据，运行以下命令从Hugging Face下载："
    echo "   python3 examples/data_preprocess/gsm8k.py --local_save_dir ${TARGET_DATA_DIR}"
    echo ""
    echo "2. 如果你的数据在其他位置，请将数据放到 ${SOURCE_DATA_DIR}"
    echo "   或修改此脚本中的 SOURCE_DATA_DIR 变量"
    exit 1
fi

echo -e "${GREEN}✓ 找到源数据目录${NC}"
echo ""

# 检查数据格式并处理
echo "检查数据格式..."

# 情况1: 已经有处理好的parquet文件
if [ -f "$SOURCE_DATA_DIR/train.parquet" ] && [ -f "$SOURCE_DATA_DIR/test.parquet" ]; then
    echo -e "${GREEN}✓ 发现parquet格式数据${NC}"
    echo "复制数据文件..."

    cp "$SOURCE_DATA_DIR/train.parquet" "$TARGET_DATA_DIR/"
    cp "$SOURCE_DATA_DIR/test.parquet" "$TARGET_DATA_DIR/"

    echo -e "${GREEN}✓ 数据文件已复制${NC}"

# 情况2: 有Hugging Face格式的数据集
elif [ -d "$SOURCE_DATA_DIR/train" ] || [ -f "$SOURCE_DATA_DIR/dataset_info.json" ]; then
    echo -e "${YELLOW}发现Hugging Face格式数据${NC}"
    echo "正在转换为parquet格式..."

    python3 examples/data_preprocess/gsm8k.py \
        --local_dataset_path "$SOURCE_DATA_DIR" \
        --local_save_dir "$TARGET_DATA_DIR"

    echo -e "${GREEN}✓ 数据转换完成${NC}"

# 情况3: 有JSON文件
elif [ -f "$SOURCE_DATA_DIR/train.json" ] || [ -f "$SOURCE_DATA_DIR/train.jsonl" ]; then
    echo -e "${YELLOW}发现JSON格式数据${NC}"
    echo "正在转换为parquet格式..."

    # 使用Python脚本转换
    python3 << 'EOF'
import json
import os
from datasets import Dataset

source_dir = "/mnt/data/GSM8K"
target_dir = os.path.expanduser("~/data/gsm8k")

# 读取JSON文件
for split in ['train', 'test']:
    json_file = None
    for ext in ['.json', '.jsonl']:
        candidate = os.path.join(source_dir, f"{split}{ext}")
        if os.path.exists(candidate):
            json_file = candidate
            break

    if json_file:
        print(f"处理 {split} 数据...")

        # 读取数据
        data = []
        with open(json_file, 'r') as f:
            if json_file.endswith('.jsonl'):
                for line in f:
                    data.append(json.loads(line))
            else:
                data = json.load(f)

        # 转换为Dataset并保存
        dataset = Dataset.from_list(data)
        output_file = os.path.join(target_dir, f"{split}.parquet")
        dataset.to_parquet(output_file)
        print(f"✓ 保存到 {output_file}")
    else:
        print(f"⚠ 未找到 {split} 数据")

print("\n数据转换完成！")
EOF

    echo -e "${GREEN}✓ 数据转换完成${NC}"

# 情况4: 目录为空或格式不识别，从Hugging Face下载
else
    echo -e "${YELLOW}未识别到数据格式，将从Hugging Face下载${NC}"
    echo ""

    # 询问是否使用镜像
    read -p "是否使用Hugging Face镜像（国内用户推荐）? [Y/n]: " USE_MIRROR
    if [ "$USE_MIRROR" != "n" ] && [ "$USE_MIRROR" != "N" ]; then
        export HF_ENDPOINT=https://hf-mirror.com
        echo "使用Hugging Face镜像..."
    fi

    python3 examples/data_preprocess/gsm8k.py \
        --local_save_dir "$TARGET_DATA_DIR"

    echo -e "${GREEN}✓ 数据下载和转换完成${NC}"
fi

echo ""

# 创建软链接
echo "创建软链接..."
ln -sf "$TARGET_DATA_DIR/train.parquet" data/gsm8k/train.parquet
ln -sf "$TARGET_DATA_DIR/test.parquet" data/gsm8k/test.parquet
echo -e "${GREEN}✓ 软链接创建完成${NC}"

echo ""

# 验证数据
echo "验证数据..."
python3 << 'EOF'
import os
import pandas as pd

target_dir = os.path.expanduser("~/data/gsm8k")

for split in ['train', 'test']:
    parquet_file = os.path.join(target_dir, f"{split}.parquet")
    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)
        print(f"✓ {split}.parquet: {len(df)} 条数据")

        # 显示一条样例
        if len(df) > 0:
            print(f"  样例: {df.iloc[0].to_dict()}")
    else:
        print(f"✗ {split}.parquet 不存在")
EOF

echo ""
echo "=================================="
echo -e "${GREEN}数据准备完成！${NC}"
echo "=================================="
echo ""
echo "数据位置："
echo "  - 训练集: ${TARGET_DATA_DIR}/train.parquet"
echo "  - 测试集: ${TARGET_DATA_DIR}/test.parquet"
echo ""
echo "下一步: 运行训练脚本"
echo "  bash run_qwen2_5-1.5b_gsm8k_grpo.sh"
echo ""
