#!/bin/bash
# 检查自定义路径的模型和数据

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=================================="
echo "  路径和环境检查"
echo "=================================="
echo ""

# ============ 检查模型路径 ============
echo "1. 检查模型路径..."
MODEL_PATH="/mnt/data/Qwen2.5-1.5B-Instruct"

if [ -d "$MODEL_PATH" ]; then
    echo -e "${GREEN}✓${NC} 模型目录存在: ${MODEL_PATH}"

    # 检查关键文件
    REQUIRED_FILES=("config.json" "tokenizer_config.json")
    MODEL_FILES=("model.safetensors" "pytorch_model.bin" "model-00001-of-00002.safetensors")

    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$MODEL_PATH/$file" ]; then
            echo -e "  ${GREEN}✓${NC} $file"
        else
            echo -e "  ${RED}✗${NC} $file (缺失)"
        fi
    done

    # 检查模型权重（至少有一种格式）
    MODEL_FOUND=0
    for file in "${MODEL_FILES[@]}"; do
        if ls "$MODEL_PATH"/$file* 1> /dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} 模型权重文件"
            MODEL_FOUND=1
            break
        fi
    done

    if [ $MODEL_FOUND -eq 0 ]; then
        echo -e "  ${RED}✗${NC} 模型权重文件 (未找到)"
    fi

    # 显示目录内容
    echo ""
    echo "  模型目录内容（前10个文件）："
    ls -lh "$MODEL_PATH" | head -11 | tail -10 | sed 's/^/    /'

else
    echo -e "${RED}✗${NC} 模型目录不存在: ${MODEL_PATH}"
    echo ""
    echo "  请将Qwen2.5-1.5B-Instruct模型下载到该路径"
    echo "  或修改训练脚本中的MODEL_PATH变量"
    echo ""
    echo "  下载模型的方法："
    echo "    方法1 - 使用huggingface-cli:"
    echo "      huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ${MODEL_PATH}"
    echo ""
    echo "    方法2 - 使用git:"
    echo "      git lfs install"
    echo "      git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct ${MODEL_PATH}"
fi
echo ""

# ============ 检查数据路径 ============
echo "2. 检查数据路径..."
DATA_SOURCE="/mnt/data/GSM8K"
DATA_TARGET="$HOME/data/gsm8k"

# 检查源数据
if [ -d "$DATA_SOURCE" ]; then
    echo -e "${GREEN}✓${NC} 源数据目录存在: ${DATA_SOURCE}"
    echo ""
    echo "  数据目录内容:"
    ls -lh "$DATA_SOURCE" | sed 's/^/    /'
    echo ""

    # 检查数据格式
    if [ -f "$DATA_SOURCE/train.parquet" ] && [ -f "$DATA_SOURCE/test.parquet" ]; then
        echo -e "  ${GREEN}✓${NC} 发现parquet格式数据 (已处理)"
    elif [ -f "$DATA_SOURCE/train.json" ] || [ -f "$DATA_SOURCE/train.jsonl" ]; then
        echo -e "  ${YELLOW}⚠${NC} 发现JSON格式数据 (需要转换)"
        echo "    请运行: bash prepare_gsm8k_data_custom.sh"
    elif [ -d "$DATA_SOURCE/train" ] || [ -f "$DATA_SOURCE/dataset_info.json" ]; then
        echo -e "  ${YELLOW}⚠${NC} 发现Hugging Face格式数据 (需要转换)"
        echo "    请运行: bash prepare_gsm8k_data_custom.sh"
    else
        echo -e "  ${YELLOW}⚠${NC} 数据格式未识别"
        echo "    请运行: bash prepare_gsm8k_data_custom.sh"
    fi
else
    echo -e "${RED}✗${NC} 源数据目录不存在: ${DATA_SOURCE}"
    echo ""
    echo "  选项1: 如果数据在其他位置，请移动到 ${DATA_SOURCE}"
    echo "  选项2: 运行数据准备脚本从Hugging Face下载:"
    echo "    bash prepare_gsm8k_data_custom.sh"
fi
echo ""

# 检查处理后的数据
echo "3. 检查处理后的数据..."
if [ -f "$DATA_TARGET/train.parquet" ] && [ -f "$DATA_TARGET/test.parquet" ]; then
    echo -e "${GREEN}✓${NC} 处理后的数据已准备: ${DATA_TARGET}"

    # 显示数据统计
    python3 << EOF 2>/dev/null || echo "  (无法读取数据详情)"
import pandas as pd
import os

data_dir = os.path.expanduser("~/data/gsm8k")
try:
    train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
    print(f"  训练集: {len(train_df)} 条数据")
    print(f"  测试集: {len(test_df)} 条数据")
except Exception as e:
    print(f"  读取数据时出错: {e}")
EOF

else
    echo -e "${YELLOW}⚠${NC} 处理后的数据不存在"
    echo "    请运行: bash prepare_gsm8k_data_custom.sh"
fi
echo ""

# ============ 检查Python环境 ============
echo "4. 检查Python环境..."

# Python版本
PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "${GREEN}✓${NC} ${PYTHON_VERSION}"

# 关键包
echo ""
echo "  关键Python包:"

PACKAGES=("torch:PyTorch" "vllm:vLLM" "transformers:Transformers" "verl:verl" "ray:Ray" "flash_attn:Flash Attention")

for pkg_info in "${PACKAGES[@]}"; do
    IFS=':' read -r pkg name <<< "$pkg_info"
    if python3 -c "import $pkg" 2>/dev/null; then
        if [ "$pkg" = "torch" ]; then
            VERSION=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
            CUDA_AVAILABLE=$(python3 -c "import $pkg; print($pkg.cuda.is_available())" 2>/dev/null)
            echo -e "  ${GREEN}✓${NC} $name $VERSION (CUDA: $CUDA_AVAILABLE)"
        else
            VERSION=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "")
            echo -e "  ${GREEN}✓${NC} $name $VERSION"
        fi
    else
        echo -e "  ${RED}✗${NC} $name (未安装)"
    fi
done
echo ""

# ============ 检查GPU ============
echo "5. 检查GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} 找到 ${GPU_COUNT} 个GPU"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | while read line; do
        echo "  $line"
    done
else
    echo -e "${RED}✗${NC} 未找到nvidia-smi"
fi
echo ""

# ============ 总结 ============
echo "=================================="
echo "  检查总结"
echo "=================================="
echo ""

# 判断是否准备好训练
READY=1

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}✗${NC} 模型未准备好"
    READY=0
else
    echo -e "${GREEN}✓${NC} 模型已准备"
fi

if [ ! -f "$DATA_TARGET/train.parquet" ] || [ ! -f "$DATA_TARGET/test.parquet" ]; then
    echo -e "${YELLOW}⚠${NC} 数据需要处理"
    READY=0
else
    echo -e "${GREEN}✓${NC} 数据已准备"
fi

if ! python3 -c "import verl" 2>/dev/null; then
    echo -e "${RED}✗${NC} verl未安装"
    READY=0
else
    echo -e "${GREEN}✓${NC} verl已安装"
fi

echo ""

if [ $READY -eq 1 ]; then
    echo -e "${GREEN}✓✓✓ 环境已准备就绪，可以开始训练！${NC}"
    echo ""
    echo "运行训练命令:"
    echo "  bash run_qwen2_5-1.5b_gsm8k_grpo_custom.sh"
else
    echo -e "${YELLOW}还有一些准备工作需要完成:${NC}"
    echo ""

    if [ ! -d "$MODEL_PATH" ]; then
        echo "1. 下载模型到 ${MODEL_PATH}"
    fi

    if [ ! -f "$DATA_TARGET/train.parquet" ]; then
        echo "2. 准备数据: bash prepare_gsm8k_data_custom.sh"
    fi

    if ! python3 -c "import verl" 2>/dev/null; then
        echo "3. 安装verl: bash quick_install_verl.sh"
    fi
fi

echo ""
