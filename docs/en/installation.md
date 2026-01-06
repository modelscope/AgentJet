# Installation Guide

This document provides a step-by-step guide to installing AgentScope-Tuner.

!!! tip "Latest Version Recommended"
    AgentScope Tuner is under active development and iteration. We recommend installing from source to get the latest features and bug fixes.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| **Python** | 3.10 |
| **CUDA** | 12.8 or higher |

---

## Install from Source

### Step 1: Clone the Repository

Clone the AgentScope Tuner repository from GitHub and navigate into the project directory:

```bash
git clone https://github.com/agentscope-ai/agentscope-tuner.git
cd agentscope-tuner
```

### Step 2: Install Dependencies

AgentScope-Tuner supports multiple backbones. Currently we have `verl` and `trinity` (recommended).

!!! info "Package Manager"
    We recommend using `uv` to manage your Python environment as it is incredibly fast. See also [`uv` installation document](https://docs.astral.sh/uv/getting-started/installation/).
    
    If you prefer `conda`, you can also install via conda and pip (simply change `uv pip` to `pip`).

=== "Trinity (Recommended)"

    Install with `trinity` training backbone for fully asynchronous RFT:

    ```bash
    uv venv --python=3.10
    source .venv/bin/activate
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[trinity]
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache
    ```

=== "Verl"

    Install with `verl` training backbone:

    ```bash
    uv venv --python=3.10
    source .venv/bin/activate
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[verl]
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache
    ```

    !!! warning "flash-attn Installation"
        `flash-attn` must be installed after other dependencies. To build faster, export `MAX_JOBS=${N_CPU}`, or ensure a healthy connection to GitHub to install pre-compiled wheels.

---

## Install via Docker

If you prefer one-click dependency installation, we provide a Docker image to jump start!

!!! warning "Prerequisites"
    Before proceeding, ensure you have **nvidia docker** installed on your system. CUDA is needed inside our docker container, which requires toolkits from Nvidia for GPU support.

### Setup Nvidia Docker Runtime

Please install nvidia docker runtime on the host Ubuntu system. For details, refer to:

- [Nvidia Official Document](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)
- [Setup Ubuntu Guide](./setup_ubuntu.md) (our step-by-step manual)

### Run Docker Container

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

---

## Verify Installation

After installation, verify that everything is working correctly:

```python
import agentscope_tuner
print(agentscope_tuner.__version__)
```

---

## Troubleshooting

??? note "Common Issues"
    **Issue**: `flash-attn` installation fails
    
    **Solution**: Make sure you have CUDA toolkit installed and `MAX_JOBS` environment variable set:
    ```bash
    export MAX_JOBS=4
    uv pip install flash-attn --no-build-isolation
    ```

??? note "GPU Not Detected"
    **Issue**: Docker container doesn't see GPU
    
    **Solution**: Ensure nvidia-docker is properly installed:
    ```bash
    nvidia-smi  # Should show GPU info
    docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
    ```

---

## Next Steps

<div class="card-grid">
<a href="../quickstart/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:lightning-bolt.svg" class="card-icon card-icon-agent" alt=""><h3>Quick Start</h3></div><p class="card-desc">Run your first training command and explore examples.</p></a>
<a href="../tune_your_first_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:rocket-launch.svg" class="card-icon card-icon-tool" alt=""><h3>Tune Your First Agent</h3></div><p class="card-desc">Step-by-step guide to build and train your own agent.</p></a>
</div>
