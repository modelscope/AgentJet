---
name: uv-install-agentjet-swarm-server
description: Install agentjet swarm server with uv package manager
license: Complete terms in LICENSE.txt
---

>
> when the user only need to run agentjet client, and do not have to run models locally (e.g. user in their laptop), ONLY install AgentJet basic requirements is enough (pip install -e .).
> see `install-agentjet-client` skill
>

# Prerequisites Check

Check Python version requirement (3.10) and uv availability.

Verify user has uv installed:
```bash
uv --version
```

If uv is not installed, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

---

# Step 1: Clone the Repository

Clone the AgentJet repository from GitHub and navigate into the project directory:

```bash
git clone https://github.com/modelscope/AgentJet.git
cd AgentJet
```

---

# Step 2: Create Virtual Environment

Create a new virtual environment with Python 3.10 using uv:

```bash
uv venv --python=3.10
source .venv/bin/activate
```

---

# Step 3: Install Dependencies

Install AgentJet with the `verl` training backbone:

```bash
uv pip install -e .[verl]
```

For users in China (faster with Aliyun mirror):
```bash
uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .[verl]
```

---

# Step 4: Test GitHub Connection (Before flash-attn)

Before installing flash-attn, test your connection to GitHub:

```bash
curl -I https://github.com --connect-timeout 10
```

or

```bash
git ls-remote https://github.com/Dao-AILab/flash-attention.git
```

!!! danger "⚠️ CRITICAL: Unstable GitHub Connection Detected"
    If the above command **fails** or **times out**, you will NOT be able to download pre-compiled flash-attn wheels from GitHub.

    This means flash-attn will need to be **compiled from source**, which can take:
    - **30+ minutes** on a fast machine
    - **1-2+ hours** on slower machines

    **RECOMMENDED**: Find a way to establish a stable GitHub connection before proceeding:
    - Use a VPN or proxy
    - Use GitHub mirrors if available in your region
    - Wait for better network conditions
    - Use a machine with better GitHub connectivity

    If you cannot establish a stable connection and must proceed with compilation:
    ```bash
    # Set to number of CPUs to speed up compilation
    export MAX_JOBS=$(nproc)
    ```

---

# Step 5: Install flash-attn

`flash-attn` must be installed **after** other dependencies:

```bash
uv pip install --verbose flash-attn --no-deps --no-build-isolation --no-cache
```

For users in China (faster with Aliyun mirror):
```bash
uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --verbose flash-attn --no-deps --no-build-isolation --no-cache
```

!!! warning "flash-attn Installation Notes"
    - `flash-attn` must be installed **after** other dependencies.
    - If installation takes a long time, ensure a healthy connection to GitHub.
    - To build faster, you can set: `export MAX_JOBS=${N_CPU}` (replace `${N_CPU}` with number of CPUs).

---

# Step 6: Verify Installation

Verify the installation by checking the AgentJet version:

```bash
python -c "import ajet; print(ajet.__version__)"
```

---

# Next Steps

After successful installation:

1. **Quick Start**: Run your first training command and explore examples
2. **Tune Your First Agent**: Follow the step-by-step guide to build and train your own agent