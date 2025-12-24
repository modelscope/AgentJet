# Installation Guide

This document provides a step-by-step guide to installing AgentScope-Tuner.

```{tip}
AgentScope Tuner is under active development and iteration. We recommend installing from source to get the latest futures and bug fixes.
```

## Prerequisites

- Python 3.10

- CUDA 12.8 or higher


## Install from Source

### Step 1: Clone the Repository

Clone the AgentScope Tuner repository from GitHub and navigate into the project directory:

```bash
git clone https://github.com/agentscope-ai/agentscope-tuner.git
cd agentscope-tuner
```


### Step 2: Install Dependencies

#### 1. Install in your native system

AgentScope-Tuner supports multiple backbones, currently we have `verl` and `trinity` (recommended).
You can choose you backbone as you wish, and choose any one of them during training as you wish.
We recommend using `uv` to manage your Python environment as it is incredibly fast.  See also [`uv` installation document](https://docs.astral.sh/uv/getting-started/installation/). However, if you wish to use `conda`, you can also install it via conda and pip (simply change to `uv pip` to `pip`):



- Install with `trinity` training backbone (Recommended).
  ```bash
  uv venv --python=3.10
  source .venv/bin/activate
  uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[trinity]
  uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache  # Hint: flash-attn must be installed after other deps
  ```


- Install with `verl` training backbone.
  ```bash
  source .venv/bin/activate
  uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[verl]
  uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache  # Hint: flash-attn must be installed after other deps, export MAX_JOBS=${N_CPU} to build faster
  ```

#### 2. Install one-click via docker

If you prefer one-click dependency installation, we provide image to jump start!

However, before proceeding, ensure you have `nvidia docker` installed on your system.
Cuda is needed inside our docker container, which need toolkits from Nvidia for GPU support.
Please install nvidia docker runtime on the host ubuntu system.
For details, refer to [nvidia official document](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian).
According the link above, we also write a manual about installing nvidia docker runtime in ubuntu linux environment, please read [set up nvidia docker environment](./setup_ubuntu.md).


This command mounts your current working directory (the root directory of agentscope-tuner) to `/workspace` and your data directory to `/data` inside the container.

```bash
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v /path/to/your/checkpoint/and/data:/data \
  agentscope-tuner:latest
```
