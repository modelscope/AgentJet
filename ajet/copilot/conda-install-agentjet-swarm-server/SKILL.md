---
name: conda-install-agentjet-swarm-server
description: Install agentjet swarm server with conda
license: Complete terms in LICENSE.txt
---


>
> when the user only need to run agentjet client, and do not have to run models locally (e.g. user in their laptop), ONLY install AgentJet basic requirements is enough (pip install -e .).
> see `install-agentjet-client` skill
>

# Prerequisites Check

Check Python version requirement (3.10) and conda availability.

Verify user has conda installed:
```bash
conda --version
```

# Step 1: Clone the Repository

Clone the AgentJet repository from GitHub and navigate into the project directory:

```bash
git clone https://github.com/modelscope/AgentJet.git
cd AgentJet
```

# Step 2: Create Conda Environment

Create a new conda environment with Python 3.10:

```bash
conda create -n ajet-verl python=3.10
conda activate ajet-verl
```

# Step 3: Install Dependencies

## Default Installation

Install AgentJet with the `verl` training backbone:

```bash
pip install -e .[verl]
```

## For Users in China

Use Aliyun PyPI mirror for faster downloads:

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[verl]
```

# Step 4: Test GitHub Connection

Before installing flash-attn, test the connection to GitHub:

```bash
curl -I --connect-timeout 10 https://github.com 2>/dev/null | head -n 1
```

If the connection test fails or times out:

!!! danger "IMPORTANT: GitHub Connection Issue Detected"
    ⚠️ **WARNING: You cannot connect to GitHub stably!**

    flash-attn requires downloading pre-compiled wheels from GitHub. Without a stable GitHub connection, pip will attempt to build flash-attn from source, which can take **30 minutes to several hours**.

    **Recommended actions:**
    1. Set up a stable connection to GitHub (VPN, proxy, etc.)
    2. Use a mirror that hosts flash-attn pre-compiled wheels
    3. If you must proceed, set `MAX_JOBS` to limit CPU usage during build:
       ```bash
       export MAX_JOBS=4  # Adjust based on your CPU cores
       ```

    **Do you want to proceed anyway?** (This may take a very long time)

# Step 5: Install flash-attn

`flash-attn` must be installed **after** other dependencies:

## Default Installation

```bash
pip install --verbose flash-attn --no-deps --no-build-isolation --no-cache
```

## For Users in China

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache
```

!!! warning "flash-attn Installation"
    - `flash-attn` must be installed **after** other dependencies.
    - Ensure a healthy connection to GitHub to install pre-compiled wheels.
    - If you find your machine spend a long time installing flash-attn, ensure a healthy connection to GitHub.
    - To build faster, export `MAX_JOBS=${N_CPU}`.

# Verification

Verify the installation:

```bash
python -c "import ajet; print(ajet.__version__)"
```