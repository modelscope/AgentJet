# Installation Guide

This document provides a step-by-step guide to installing AgentJet.

!!! tip "Latest Version Recommended:"

    AgentJet is under active development and iteration. We recommend installing from source to get the latest features and bug fixes.


## Prerequisites

| Requirement | Detail |
|-------------|---------|
| **Python**         | 3.10 |
| Package Management | `uv` or `conda` |


## Install from Source

### Step 1: Clone the Repository

Clone the AgentJet repository from GitHub and navigate into the project directory:

```bash
git clone https://github.com/modelscope/AgentJet.git
cd AgentJet
```

### Step 2: Install Dependencies

AgentJet supports multiple backbones. Currently we have `verl` and `trinity` (recommended).

!!! info "Package Manager"
    We recommend using `uv` to manage your Python environment as it is incredibly fast. See also [`uv` installation document](https://docs.astral.sh/uv/getting-started/installation/).

    And of course, if you prefer `conda`, you can also install via conda and pip (simply change `uv pip` to `pip`).

=== "VERL (uv)"

    ```bash
    # Install with `verl` training backbone:

    uv venv --python=3.10
    source .venv/bin/activate
    uv pip install -e .[verl]

    #`flash-attn` must be installed after other dependencies
    uv pip install --verbose flash-attn --no-deps --no-build-isolation --no-cache
    ```

    !!! warning "flash-attn Installation"
        `flash-attn` must be installed after other dependencies. To build faster, export `MAX_JOBS=${N_CPU}`, or ensure a healthy connection to GitHub to install pre-compiled wheels.

=== "VERL (conda)"

    ```bash
    # Install with `verl` training backbone:

    conda create -n ajet-verl python=3.10
    conda activate ajet-verl
    pip install -e .[verl]

    #`flash-attn` must be installed after other dependencies
    pip install --verbose flash-attn --no-deps --no-build-isolation --no-cache
    ```

    !!! warning "flash-attn Installation"
        `flash-attn` must be installed after other dependencies. To build faster, export `MAX_JOBS=${N_CPU}`, or ensure a healthy connection to GitHub to install pre-compiled wheels.


=== "VERL (aliyun)"


    ```bash
    # Install with `verl` training backbone:

    uv venv --python=3.10
    source .venv/bin/activate
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[verl]

    #`flash-attn` must be installed after other dependencies
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache
    ```

    !!! warning "flash-attn Installation"
        - `flash-attn` must be installed **after** other dependencies.
        - Ensure a healthy connection to GitHub to install pre-compiled wheels.
        - If you find your machine spend a long time installing flash-attn, ensure a healthy connection to GitHub.
        - To build faster, export `MAX_JOBS=${N_CPU}`.


=== "Trinity"

    ```bash
    # Install with `trinity` training backbone for fully asynchronous RFT:

    uv venv --python=3.10
    source .venv/bin/activate
    uv pip install -e .[trinity]
    uv pip install --verbose flash-attn --no-deps --no-build-isolation --no-cache
    ```


=== "Trinity (aliyun)"

    ```bash
    # Install with `trinity` training backbone for fully asynchronous RFT:

    uv venv --python=3.10
    source .venv/bin/activate
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[trinity]
    uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache
    ```


| Backbone  | VeRL     | Trinity-RFT     |
| -------- |--------  | -------------   |
| Core design   | Share-GPU actor-rollout engine (colocate) |   Async actor-rollout engine    |
| Speed         | ⭐⭐⭐⭐    |       ⭐⭐⭐⭐          |
| Scalability   |  ⭐⭐   |        ⭐⭐⭐⭐      |
| Minimum Required GPU Resource   |   1  |             2                |
| Training Stability   | ⭐⭐⭐⭐ |     ⭐⭐⭐            |
| vLLM Version   | 0.10.0 |     0.10.0         |





## Next Steps

<div class="card-grid">
<a href="../quickstart/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:lightning-bolt.svg" class="card-icon card-icon-agent" alt=""><h3>Quick Start</h3></div><p class="card-desc">Run your first training command and explore examples.</p></a>
<a href="../tune_your_first_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:rocket-launch.svg" class="card-icon card-icon-tool" alt=""><h3>Tune Your First Agent</h3></div><p class="card-desc">Step-by-step guide to build and train your own agent.</p></a>
</div>
