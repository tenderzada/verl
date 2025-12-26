#!/bin/bash
# verl 快速安装脚本
# 适用于已有CUDA 12.8+ 和 Python 3.10+ 的环境

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=================================="
echo "  verl 快速安装脚本"
echo "=================================="
echo ""

# 检查Python版本
echo -e "${YELLOW}检查Python版本...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.10"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}错误: Python版本需要 >= 3.10，当前版本: ${PYTHON_VERSION}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python版本: ${PYTHON_VERSION}${NC}"
echo ""

# 检查CUDA
echo -e "${YELLOW}检查CUDA...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    echo -e "${GREEN}✓ CUDA版本: ${CUDA_VERSION}${NC}"
else
    echo -e "${YELLOW}⚠ 未找到nvcc，请确保已安装CUDA 12.8+${NC}"
fi
echo ""

# 询问安装选项
echo "请选择安装方式："
echo "1) 仅FSDP（推荐，快速安装，适合大多数场景）"
echo "2) FSDP + Megatron（完整安装，适合大规模训练）"
read -p "请输入选择 [1/2]: " INSTALL_CHOICE

if [ "$INSTALL_CHOICE" = "2" ]; then
    USE_MEGATRON=1
    echo -e "${YELLOW}将安装完整版本（包含Megatron），预计需要30-60分钟${NC}"
else
    USE_MEGATRON=0
    echo -e "${YELLOW}将安装FSDP版本，预计需要15-30分钟${NC}"
fi
echo ""

# 询问是否安装SGLang
read -p "是否安装SGLang（用于多轮对话）? [y/N]: " INSTALL_SGLANG
if [ "$INSTALL_SGLANG" = "y" ] || [ "$INSTALL_SGLANG" = "Y" ]; then
    USE_SGLANG=1
else
    USE_SGLANG=0
fi
echo ""

# 确认继续
read -p "准备开始安装，是否继续? [Y/n]: " CONFIRM
if [ "$CONFIRM" = "n" ] || [ "$CONFIRM" = "N" ]; then
    echo "安装已取消"
    exit 0
fi
echo ""

# 开始安装
echo -e "${GREEN}开始安装...${NC}"
echo ""

# 1. 安装推理框架
echo -e "${YELLOW}[1/5] 安装推理框架...${NC}"
if [ $USE_SGLANG -eq 1 ]; then
    echo "安装SGLang..."
    pip install "sglang[all]==0.5.2" --no-cache-dir
    pip install torch-memory-saver --no-cache-dir
fi

echo "安装vLLM..."
pip install --no-cache-dir "vllm==0.11.0"
echo -e "${GREEN}✓ 推理框架安装完成${NC}"
echo ""

# 2. 安装基础依赖
echo -e "${YELLOW}[2/5] 安装基础依赖包...${NC}"
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel \
    pytest tensorboard

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo -e "${GREEN}✓ 基础依赖安装完成${NC}"
echo ""

# 3. 安装Flash Attention
echo -e "${YELLOW}[3/5] 安装Flash Attention...${NC}"

# 检查Python版本以选择正确的wheel
PYTHON_MINOR=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f2)
if [ "$PYTHON_MINOR" = "12" ]; then
    FLASH_ATTN_WHEEL="flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
elif [ "$PYTHON_MINOR" = "11" ]; then
    FLASH_ATTN_WHEEL="flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
elif [ "$PYTHON_MINOR" = "10" ]; then
    FLASH_ATTN_WHEEL="flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
else
    echo -e "${YELLOW}⚠ Python 3.${PYTHON_MINOR}可能需要手动安装Flash Attention${NC}"
    FLASH_ATTN_WHEEL=""
fi

if [ -n "$FLASH_ATTN_WHEEL" ]; then
    wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/${FLASH_ATTN_WHEEL} && \
        pip install --no-cache-dir ${FLASH_ATTN_WHEEL} && \
        rm ${FLASH_ATTN_WHEEL}
    echo -e "${GREEN}✓ Flash Attention安装完成${NC}"
else
    echo -e "${YELLOW}⚠ 跳过Flash Attention安装${NC}"
fi

# FlashInfer
pip install --no-cache-dir flashinfer-python==0.3.1

echo ""

# 4. 安装Megatron相关（如果选择）
if [ $USE_MEGATRON -eq 1 ]; then
    echo -e "${YELLOW}[4/5] 安装Megatron和TransformerEngine...${NC}"
    echo "注意：这个过程可能需要20-40分钟，请耐心等待..."

    pip install "onnxscript==0.3.1"

    echo "安装TransformerEngine..."
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6

    echo "安装Megatron-LM..."
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1

    echo -e "${GREEN}✓ Megatron安装完成${NC}"
else
    echo -e "${YELLOW}[4/5] 跳过Megatron安装${NC}"
fi
echo ""

# 5. 修复OpenCV
echo -e "${YELLOW}[5/5] 修复OpenCV...${NC}"
pip install opencv-python
pip install opencv-fixer
python -c "from opencv_fixer import AutoFix; AutoFix()" 2>/dev/null || true
echo -e "${GREEN}✓ OpenCV修复完成${NC}"
echo ""

# 安装cudnn（如果安装了Megatron）
if [ $USE_MEGATRON -eq 1 ]; then
    echo -e "${YELLOW}安装cuDNN Python包...${NC}"
    pip install nvidia-cudnn-cu12==9.10.2.21
    echo -e "${GREEN}✓ cuDNN安装完成${NC}"
    echo ""
fi

# 安装verl
echo -e "${YELLOW}安装verl...${NC}"
if [ -f "setup.py" ]; then
    pip install --no-deps -e .
    echo -e "${GREEN}✓ verl安装完成${NC}"
else
    echo -e "${YELLOW}⚠ 请在verl目录下运行此脚本${NC}"
    echo "或手动运行: pip install --no-deps -e ."
fi
echo ""

# 验证安装
echo "=================================="
echo "  验证安装"
echo "=================================="
echo ""

# 检查verl
if python3 -c "import verl" 2>/dev/null; then
    echo -e "${GREEN}✓ verl${NC}"
else
    echo -e "${RED}✗ verl${NC}"
fi

# 检查PyTorch
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}✓ PyTorch ${TORCH_VERSION} (CUDA可用)${NC}"
else
    echo -e "${YELLOW}⚠ PyTorch (CUDA不可用或未安装)${NC}"
fi

# 检查vLLM
if python3 -c "import vllm" 2>/dev/null; then
    echo -e "${GREEN}✓ vLLM${NC}"
else
    echo -e "${RED}✗ vLLM${NC}"
fi

# 检查Transformers
if python3 -c "import transformers" 2>/dev/null; then
    TRANS_VERSION=$(python3 -c "import transformers; print(transformers.__version__)")
    echo -e "${GREEN}✓ Transformers ${TRANS_VERSION}${NC}"
else
    echo -e "${RED}✗ Transformers${NC}"
fi

# 检查Flash Attention
if python3 -c "import flash_attn" 2>/dev/null; then
    echo -e "${GREEN}✓ Flash Attention${NC}"
else
    echo -e "${YELLOW}⚠ Flash Attention${NC}"
fi

# 检查SGLang
if [ $USE_SGLANG -eq 1 ]; then
    if python3 -c "import sglang" 2>/dev/null; then
        echo -e "${GREEN}✓ SGLang${NC}"
    else
        echo -e "${RED}✗ SGLang${NC}"
    fi
fi

# 检查Megatron
if [ $USE_MEGATRON -eq 1 ]; then
    if python3 -c "import megatron" 2>/dev/null; then
        echo -e "${GREEN}✓ Megatron-LM${NC}"
    else
        echo -e "${YELLOW}⚠ Megatron-LM${NC}"
    fi
fi

echo ""
echo "=================================="
echo -e "${GREEN}安装完成！${NC}"
echo "=================================="
echo ""
echo "下一步："
echo "1. 运行环境检查: bash check_environment.sh"
echo "2. 准备数据集: bash prepare_gsm8k_data.sh"
echo "3. 开始训练: bash run_qwen2_5-1.5b_gsm8k_grpo.sh"
echo ""
