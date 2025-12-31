# 安装与环境准备

本节说明 bolt-lab 的环境要求与推荐安装步骤（已在 CUDA 12.6 + PyTorch cu126 组合上验证）。

## 环境要求

- Linux + NVIDIA GPU
- Python 3.10
- 已安装 NVIDIA 驱动（可运行 nvidia-smi）
- 如需安装 flash-attn：需要编译环境，并保证临时目录空间充足（建议设置 TMPDIR）

## 安装步骤（按顺序执行）

1）安装 bolt-lab（wheel 包）
pip install bolt_lab-0.1.0-py3-none-any.whl

2）安装 PyTorch（CUDA 12.6 对应 cu126）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

3）安装 NVCC（仅此项使用 conda）
conda install -c nvidia cuda-nvcc -y

4）安装其余 Python 依赖
pip install -r requirements.txt

5）安装 flash-attn（单独安装，避免构建隔离与缓存导致失败）
mkdir -p ~/tmp/pip
TMPDIR=~/tmp/pip 
pip install --no-build-isolation --no-cache-dir flash-attn==2.8.3

## 可选快速自检（确认安装成功）

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
bolt-grid --help

## 常见问题

- torch.cuda.is_available() 为 False：
  优先检查 NVIDIA 驱动是否正常、以及 PyTorch 是否安装了 cu126 版本。
- flash-attn 安装失败：
  常见原因是临时目录空间不足（建议设置 TMPDIR）、torch/CUDA 不匹配、或编译环境缺失。
