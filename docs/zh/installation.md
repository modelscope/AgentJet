# 安装指南

本文档提供 AgentScope-Tuner 的逐步安装说明。

AgentScope Tuner 正处于快速开发与迭代阶段。我们建议从源码安装，以获得最新功能与 Bug 修复。

## 前置条件

* Python 3.10
* CUDA 12.8 或更高版本

## 从源码安装

### Step 1：克隆仓库

从 GitHub 克隆 AgentScope Tuner 仓库，并进入项目目录：

```bash
git clone https://github.com/agentscope-ai/agentscope-tuner.git
cd agentscope-tuner
```

### Step 2：安装依赖

#### 1）在本机系统中安装

AgentScope-Tuner 支持多种训练后端（backbone），目前包括 `verl` 和 `trinity`（推荐）。
你可以按需选择后端，并在训练时自由选择使用其中任意一个。
我们推荐使用 `uv` 来管理 Python 环境，因为它速度非常快。也可参考 [`uv` 安装文档](https://docs.astral.sh/uv/getting-started/installation/)。
当然，如果你更希望使用 `conda`，也可以通过 conda + pip 安装（只需将 `uv pip` 替换为 `pip` 即可）。

* 使用 `trinity` 训练后端安装（推荐）

  ```bash
  uv venv --python=3.10
  source .venv/bin/activate
  uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[trinity]
  uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache
  ```

* 使用 `verl` 训练后端安装

  ```bash
  source .venv/bin/activate
  uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[verl]
  uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache  # 提示：flash-attn 必须在其他依赖安装完成后再安装，你可以 (1) 通过 export MAX_JOBS=${N_CPU} 加快编译速度，或者 (2) 通过确保 Github 访问畅通来直接安装预编译轮子
  ```

#### 2）通过 Docker 一键安装

如果你希望一键完成依赖安装，我们提供了镜像用于快速上手。

在继续之前，请确保你的系统已安装 `nvidia docker`。
我们的 Docker 容器内部需要使用 CUDA，因此必须依赖 Nvidia 的工具链来提供 GPU 支持。
请在宿主机的 Ubuntu 系统上安装 nvidia docker runtime。具体步骤可参考 [nvidia official document](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)。
基于上述链接，我们也编写了一份在 Ubuntu Linux 环境中安装 nvidia docker runtime 的说明文档，请阅读 [set up nvidia docker environment](./setup_ubuntu.md)。

以下命令会将你当前工作目录（agentscope-tuner 的根目录）挂载到容器内的 `/workspace`，并将你的数据目录挂载到容器内的 `/data`。

```bash
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v /path/to/your/checkpoint/and/data:/data \
  agentscope-tuner:latest
```
