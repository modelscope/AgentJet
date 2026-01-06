# 安装指南

本文档提供 AgentScope-Tuner 的逐步安装说明。

!!! tip "推荐从源码安装"
    AgentScope Tuner 正处于快速开发与迭代阶段。我们建议从源码安装，以获得最新功能与 Bug 修复。

---

## 前置条件

| 要求 | 版本 |
|------|------|
| **Python** | 3.10 |
| **CUDA** | 12.8 或更高版本 |

---

## 从源码安装

### Step 1：克隆仓库

从 GitHub 克隆 AgentScope Tuner 仓库，并进入项目目录：

```bash
git clone https://github.com/agentscope-ai/agentscope-tuner.git
cd agentscope-tuner
```

### Step 2：安装依赖

AgentScope-Tuner 支持多种训练后端（backbone），目前包括 `verl` 和 `trinity`（推荐）。

!!! info "包管理器"
    我们推荐使用 `uv` 来管理 Python 环境，因为它速度非常快。参考 [`uv` 安装文档](https://docs.astral.sh/uv/getting-started/installation/)。

    如果您更希望使用 `conda`，也可以通过 conda + pip 安装（只需将 `uv pip` 替换为 `pip` 即可）。

=== "Trinity（推荐）"

    使用 `trinity` 训练后端安装，支持全异步 RFT：

    ```bash
    uv venv --python=3.10
    source .venv/bin/activate
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[trinity]
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache
    ```

=== "Verl"

    使用 `verl` 训练后端安装：

    ```bash
    uv venv --python=3.10
    source .venv/bin/activate
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[verl]
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache
    ```

    !!! warning "flash-attn 安装说明"
        `flash-attn` 必须在其他依赖安装完成后再安装。为加快编译速度，可通过 `export MAX_JOBS=${N_CPU}` 设置并行编译数，或确保 GitHub 访问畅通以直接安装预编译轮子。

---

## 通过 Docker 一键安装

如果您希望一键完成依赖安装，我们提供了镜像用于快速上手。

!!! warning "前置条件"
    在继续之前，请确保您的系统已安装 **nvidia docker**。我们的 Docker 容器内部需要使用 CUDA，因此必须依赖 Nvidia 的工具链来提供 GPU 支持。

### 设置 Nvidia Docker Runtime

请在宿主机的 Ubuntu 系统上安装 nvidia docker runtime。详细步骤请参考：

- [Nvidia 官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)
- [Ubuntu 环境设置指南](./setup_ubuntu.md)（我们的详细安装说明）

### 运行 Docker 容器

以下命令会将您当前工作目录（agentscope-tuner 的根目录）挂载到容器内的 `/workspace`，并将您的数据目录挂载到容器内的 `/data`：

```bash
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v /path/to/your/checkpoint/and/data:/data \
  agentscope-tuner:latest
```

---

## 验证安装

安装完成后，验证一切是否正常工作：

```python
import ajet
print(ajet.__version__)
```

---

## 常见问题

??? note "flash-attn 安装失败"
    **问题**：`flash-attn` 安装失败

    **解决方案**：确保已安装 CUDA 工具包，并设置 `MAX_JOBS` 环境变量：
    ```bash
    export MAX_JOBS=4
    uv pip install flash-attn --no-build-isolation
    ```

??? note "GPU 未检测到"
    **问题**：Docker 容器无法识别 GPU

    **解决方案**：确保 nvidia-docker 已正确安装：
    ```bash
    nvidia-smi  # 应该显示 GPU 信息
    docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
    ```

---

## 下一步

<div class="card-grid">
<a href="../quickstart/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:lightning-bolt.svg" class="card-icon card-icon-agent" alt=""><h3>快速开始</h3></div><p class="card-desc">运行您的第一个训练命令并探索示例。</p></a>
<a href="../tune_your_first_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:rocket-launch.svg" class="card-icon card-icon-tool" alt=""><h3>调优你的第一个智能体</h3></div><p class="card-desc">从零开始构建和训练您自己的智能体的详细指南。</p></a>
</div>
