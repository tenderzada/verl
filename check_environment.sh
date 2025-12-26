#!/bin/bash
# 环境检查脚本

echo "=================================="
echo "  Qwen2.5-1.5B GRPO 环境检查"
echo "=================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查GPU
echo "1. 检查GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}✓${NC} 找到 ${GPU_COUNT} 个GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | nl
else
    echo -e "${RED}✗${NC} 未找到nvidia-smi命令"
    exit 1
fi
echo ""

# 2. 检查Python和关键库
echo "2. 检查Python环境..."
PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "${GREEN}✓${NC} Python: ${PYTHON_VERSION}"

# 检查PyTorch
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    echo -e "${GREEN}✓${NC} PyTorch: ${TORCH_VERSION} (CUDA: ${CUDA_AVAILABLE})"
else
    echo -e "${RED}✗${NC} PyTorch未安装"
fi

# 检查transformers
if python3 -c "import transformers" 2>/dev/null; then
    TRANSFORMERS_VERSION=$(python3 -c "import transformers; print(transformers.__version__)")
    echo -e "${GREEN}✓${NC} Transformers: ${TRANSFORMERS_VERSION}"
else
    echo -e "${YELLOW}⚠${NC} Transformers未安装"
fi

# 检查vLLM
if python3 -c "import vllm" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} vLLM已安装"
else
    echo -e "${YELLOW}⚠${NC} vLLM未安装"
fi
echo ""

# 3. 检查磁盘空间
echo "3. 检查磁盘空间..."
HOME_SPACE=$(df -h ~ | tail -1 | awk '{print $4}')
echo -e "${GREEN}✓${NC} 主目录可用空间: ${HOME_SPACE}"

if [ -d "/mnt/data" ]; then
    MNT_SPACE=$(df -h /mnt/data | tail -1 | awk '{print $4}')
    echo -e "${GREEN}✓${NC} /mnt/data可用空间: ${MNT_SPACE}"
fi
echo ""

# 4. 检查模型文件
echo "4. 检查模型文件..."
MODEL_PATH="/mnt/data/Qwen2.5-1.5B-Instruct"
if [ -d "$MODEL_PATH" ]; then
    echo -e "${GREEN}✓${NC} 模型目录存在: ${MODEL_PATH}"

    # 检查关键文件
    if [ -f "$MODEL_PATH/config.json" ]; then
        echo -e "  ${GREEN}✓${NC} config.json"
    else
        echo -e "  ${RED}✗${NC} config.json"
    fi

    if [ -f "$MODEL_PATH/model.safetensors" ] || [ -f "$MODEL_PATH/pytorch_model.bin" ]; then
        echo -e "  ${GREEN}✓${NC} 模型权重文件"
    else
        echo -e "  ${RED}✗${NC} 模型权重文件"
    fi

    if [ -f "$MODEL_PATH/tokenizer.json" ] || [ -f "$MODEL_PATH/tokenizer_config.json" ]; then
        echo -e "  ${GREEN}✓${NC} tokenizer文件"
    else
        echo -e "  ${YELLOW}⚠${NC} tokenizer文件"
    fi
else
    echo -e "${RED}✗${NC} 模型目录不存在: ${MODEL_PATH}"
    echo -e "  请先下载模型到该路径，或修改训练脚本中的MODEL_PATH"
fi
echo ""

# 5. 检查数据集
echo "5. 检查GSM8K数据集..."
DATA_DIR="$HOME/data/gsm8k"
if [ -d "$DATA_DIR" ]; then
    if [ -f "$DATA_DIR/train.parquet" ] && [ -f "$DATA_DIR/test.parquet" ]; then
        TRAIN_SIZE=$(ls -lh "$DATA_DIR/train.parquet" | awk '{print $5}')
        TEST_SIZE=$(ls -lh "$DATA_DIR/test.parquet" | awk '{print $5}')
        echo -e "${GREEN}✓${NC} 数据集已准备"
        echo -e "  训练集: ${TRAIN_SIZE}"
        echo -e "  测试集: ${TEST_SIZE}"
    else
        echo -e "${YELLOW}⚠${NC} 数据集不完整，请运行: bash prepare_gsm8k_data.sh"
    fi
else
    echo -e "${YELLOW}⚠${NC} 数据集未准备，请运行: bash prepare_gsm8k_data.sh"
fi
echo ""

# 6. 检查verl安装
echo "6. 检查verl..."
if python3 -c "import verl" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} verl已安装"
else
    echo -e "${RED}✗${NC} verl未安装"
    echo "  请运行: pip install -e ."
fi
echo ""

# 总结
echo "=================================="
echo "  检查完成"
echo "=================================="
echo ""
echo "下一步操作："
echo "1. 如果数据集未准备，运行: bash prepare_gsm8k_data.sh"
echo "2. 检查并调整训练参数: nano run_qwen2_5-1.5b_gsm8k_grpo.sh"
echo "3. 开始训练: bash run_qwen2_5-1.5b_gsm8k_grpo.sh"
echo ""
