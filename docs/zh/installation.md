# 安装指南

```{tip}
AgentScope Tuner 正在积极开发和迭代中。我们建议从源码安装，以获取最新功能和修复。
```

## 安装前提

- Python 3.10 至 3.12
- CUDA 12.8 或更高版本

## 源码安装

### 步骤 1：克隆仓库

首先，从 GitHub 克隆 AgentScope Tuner 仓库并进入项目目录：

```bash
git clone https://github.com/agentscope-ai/agentscope-tuner.git
cd agentscope-tuner
```

### 步骤 2：安装依赖

#### LLM 训练新手用户

如果你是 LLM 训练和环境配置的新手，建议使用 Docker 简化安装流程。

在继续之前，请确保你的系统已安装 Docker，并已经安装了 Nvidia GPU 相关支持。

你可以直接下载我们的镜像，或在本地构建。

##### 选项 1：下载预构建的镜像

可以使用以下命令拉取预构建的镜像：

```bash
docker pull ghcr.io/agentscope-ai/agentscope-tuner:latest
```

##### 选项 2：本地构建 Docker 镜像

我们提供了 Dockerfile，用于搭建 AgentScope Tuner 所需环境。

可使用以下命令构建 Docker 镜像：

```bash
# 请确保你在 agentscope-tuner 根目录下
docker build -f scripts/docker/dockerfile -t agentscope-tuner:latest .
```

##### 运行 Docker 容器

镜像拉取或构建完成后，可用以下命令运行容器。请将 `/path/to/your/checkpoint/and/data` 替换为你实际存放模型检查点和数据的路径。
该命令会将你当前工作目录（agentscope-tuner 根目录）挂载到容器内的 `/workspace`，并将数据目录挂载到 `/data`。

```bash
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v /path/to/your/checkpoint/and/data:/data \
  agentscope-tuner:latest
```

#### 熟悉 LLM 训练的用户

推荐使用 `uv` 管理你的 Python 环境。

如果尚未安装 `uv`，可通过 pip 安装：

```bash
pip install uv
uv venv --python=3.10.16
```

然后，激活虚拟环境并安装所需依赖：

```bash
source .venv/bin/activate
uv pip install -e.[dev]
uv pip install flash-attn==2.8.1 --no-build-isolation --no-cache-dir
```

## 从 PyPI 安装（不推荐）

你也可以通过 pip 直接从 PyPI 安装 AgentScope Tuner。此方法适合希望快速安装且无需最新开发功能的用户。

```bash
pip install agentscope-tuner
# 请在安装 agentscope-tuner 完成后安装 flash-attn，因为 flash-attn 依赖于部分已安装的包
pip install flash-attn --no-build-isolation --no-cache-dir
```
