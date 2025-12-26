# verl 环境安装指南

本指南提供两种安装方式：**Docker安装（推荐）** 和 **从零开始手动安装**。

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU（支持CUDA）
  - 推荐：RTX 3090, RTX 4090, A100, H100等
  - 最低：16GB显存
- **内存**: 32GB+ RAM
- **存储**: 50GB+ 可用空间

### 软件要求
- **Python**: >= 3.10 (推荐 3.12)
- **CUDA**: >= 12.8 (推荐 12.8.1)
- **操作系统**: Ubuntu 20.04/22.04 或类似Linux发行版

---

## 🐳 方法一：Docker安装（推荐）

Docker方式最简单，所有依赖都已预装。

### 1. 安装Docker和NVIDIA Container Toolkit

如果还没有安装Docker：

```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. 拉取verl Docker镜像

verl提供两种基础镜像，选择其一即可：

**选项A：基于vLLM的镜像（推荐用于GRPO训练）**
```bash
docker pull verlai/verl:vllm011.latest
```

**选项B：基于SGLang的镜像（适用于多轮对话）**
```bash
docker pull verlai/verl:sgl055.latest
```

### 3. 启动Docker容器

```bash
# 创建并启动容器
docker create --runtime=nvidia --gpus all --net=host \
    --shm-size="10g" --cap-add=SYS_ADMIN \
    -v /mnt/data:/mnt/data \
    -v ~/verl:/workspace/verl \
    --name verl \
    verlai/verl:vllm011.latest \
    sleep infinity

docker start verl
docker exec -it verl bash
```

### 4. 在容器内安装verl

```bash
# 克隆verl仓库（如果没有的话）
cd /workspace
git clone https://github.com/volcengine/verl.git
cd verl

# 安装verl（无依赖模式，因为Docker镜像已包含所有依赖）
pip install --no-deps -e .
```

### 5. 验证安装

```bash
python3 -c "import verl; print('verl installed successfully!')"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import vllm; print(f'vLLM installed successfully!')"
```

**✅ Docker安装完成！可以跳到"验证环境"部分。**

---

## 🔧 方法二：从零开始手动安装

如果你需要自定义环境或无法使用Docker，可以按照以下步骤手动安装。

### 步骤1: 安装CUDA和cuDNN

#### 1.1 安装CUDA 12.8

```bash
# 下载CUDA 12.8.1安装包
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.1-570.124.06-1_amd64.deb

# 安装
sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

# 设置CUDA路径
sudo update-alternatives --set cuda /usr/local/cuda-12.8

# 添加到环境变量
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

验证CUDA安装：
```bash
nvcc --version
nvidia-smi
```

#### 1.2 安装cuDNN 9.10

```bash
# 下载cuDNN 9.10
wget https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb

# 安装
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12
```

### 步骤2: 创建Python环境

使用Conda创建虚拟环境（推荐）：

```bash
# 安装Miniconda（如果没有conda）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建新环境
conda create -n verl python=3.12
conda activate verl
```

### 步骤3: 安装依赖包

verl提供了自动安装脚本。根据你的需求选择：

#### 选项A：仅使用FSDP训练（不需要Megatron）

```bash
# 克隆verl仓库
git clone https://github.com/volcengine/verl.git
cd verl

# 运行安装脚本（不安装Megatron）
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

#### 选项B：使用FSDP + Megatron训练（推荐，功能更完整）

```bash
# 克隆verl仓库
git clone https://github.com/volcengine/verl.git
cd verl

# 运行安装脚本（安装所有组件）
bash scripts/install_vllm_sglang_mcore.sh
```

**注意**：这个过程可能需要30-60分钟，请耐心等待。

#### 安装脚本会自动安装：

1. **推理框架**:
   - vLLM 0.11.0
   - SGLang 0.5.2（可选）

2. **基础依赖**:
   - PyTorch（vLLM依赖的版本）
   - Transformers >= 4.51.0
   - Accelerate, Datasets, PEFT
   - Ray, Hydra-core
   - 等等...

3. **Flash Attention 2.8.1**

4. **Megatron-LM**（如果选择安装）:
   - TransformerEngine v2.6
   - Megatron-LM core v0.13.1

### 步骤4: 安装verl

```bash
cd verl
pip install --no-deps -e .
```

### 步骤5: [可选] 安装NVIDIA Apex

如果使用Megatron-LM训练，推荐安装Apex：

```bash
# 克隆Apex
git clone https://github.com/NVIDIA/apex.git
cd apex

# 安装（设置MAX_JOBS避免内存溢出）
MAX_JOBS=8 pip install -v --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
```

**注意**：Apex编译可能需要30-60分钟。

### 步骤6: 安装后检查

确保关键包没有被覆盖：

```bash
pip list | grep -E "torch|vllm|sglang|pyarrow|tensordict|cudnn"
```

预期输出示例：
```
torch                    2.8.0
vllm                     0.11.0
sglang                   0.5.2
pyarrow                  19.0.0
tensordict               0.10.0
nvidia-cudnn-cu12        9.10.2.21
```

---

## ✅ 验证环境

运行以下检查脚本：

```bash
cd /home/user/verl
bash check_environment.sh
```

或手动验证：

```bash
# 1. 检查verl
python3 -c "import verl; print('✓ verl installed')"

# 2. 检查PyTorch和CUDA
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')"

# 3. 检查vLLM
python3 -c "import vllm; print('✓ vLLM installed')"

# 4. 检查Transformers
python3 -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"

# 5. 检查Flash Attention
python3 -c "import flash_attn; print('✓ Flash Attention installed')"

# 6. 检查Ray
python3 -c "import ray; print('✓ Ray installed')"

# 7. [可选] 检查SGLang
python3 -c "import sglang; print('✓ SGLang installed')"

# 8. [可选] 检查Megatron
python3 -c "import megatron; print('✓ Megatron installed')"
```

---

## 🔍 常见问题

### 1. CUDA版本不匹配

**问题**：`RuntimeError: CUDA version mismatch`

**解决**：
```bash
# 检查CUDA版本
nvcc --version
python3 -c "import torch; print(torch.version.cuda)"

# 如果不匹配，重新安装PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. vLLM安装失败

**问题**：`Failed to build vllm`

**解决**：
```bash
# 确保CUDA和编译工具已安装
sudo apt-get install -y build-essential

# 使用预编译wheel
pip install vllm==0.11.0 --no-build-isolation
```

### 3. Flash Attention安装失败

**问题**：编译flash-attn时内存不足或耗时过长

**解决**：
```bash
# 使用预编译wheel（推荐）
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

### 4. 包版本冲突

**问题**：安装其他包后，vLLM或PyTorch被覆盖

**解决**：
```bash
# 重新安装被覆盖的包
pip install vllm==0.11.0 --force-reinstall --no-deps

# 或者锁定版本
pip install -r requirements.txt
```

### 5. Ray初始化失败

**问题**：`Ray failed to start`

**解决**：
```bash
# 增加共享内存
# 在Docker中：--shm-size="10g"
# 或手动设置
sudo mount -o remount,size=10G /dev/shm
```

### 6. Hugging Face下载慢

**解决**：使用镜像站
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 或
pip install hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## 📦 完整依赖列表

如果自动安装脚本失败，可以手动安装以下包：

### 核心依赖
```bash
pip install \
    transformers>=4.51.0 \
    accelerate \
    datasets \
    peft \
    torch \
    vllm==0.11.0 \
    numpy\<2.0.0 \
    pyarrow>=15.0.0 \
    pandas \
    tensordict>=0.8.0,\<=0.10.0,!=0.9.0 \
    torchdata \
    ray[default] \
    codetiming \
    hydra-core \
    pylatexenc \
    wandb \
    dill \
    pybind11 \
    liger-kernel
```

### 可选依赖
```bash
# SGLang（用于多轮对话）
pip install "sglang[all]==0.5.2"

# Flash Attention（加速训练）
pip install flash-attn

# Pre-commit（代码检查）
pip install pre-commit
```

---

## 🎯 安装完成后的下一步

1. **准备数据集**：
```bash
cd /home/user/verl
bash prepare_gsm8k_data.sh
```

2. **运行环境检查**：
```bash
bash check_environment.sh
```

3. **开始训练**：
```bash
bash run_qwen2_5-1.5b_gsm8k_grpo.sh
```

---

## 📚 参考资源

- **官方文档**: https://verl.readthedocs.io/
- **安装指南**: https://verl.readthedocs.io/en/latest/start/install.html
- **GitHub仓库**: https://github.com/volcengine/verl
- **Docker镜像**: https://hub.docker.com/r/verlai/verl
- **问题反馈**: https://github.com/volcengine/verl/issues

---

## 💡 推荐配置

### 开发/研究环境（单机单卡）
- **方式**: Docker
- **镜像**: `verlai/verl:vllm011.latest`
- **训练后端**: FSDP
- **推理引擎**: vLLM

### 生产环境（多机多卡）
- **方式**: 手动安装
- **训练后端**: Megatron-LM
- **推理引擎**: vLLM或SGLang
- **包含**: Apex, TransformerEngine

---

**祝安装顺利！** 🚀

如有问题，请查看常见问题部分或提交issue。
