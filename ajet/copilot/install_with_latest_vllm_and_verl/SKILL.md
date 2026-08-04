---
name: install_with_latest_vllm_and_verl
description: Install AgentJet with the latest verl 0.8.0.dev + vllm 0.26 + torch 2.11(cu130) stack from a pinned requirements file. Handles venv creation on local disk, flash-attn source compilation against CUDA 13 (with nvidia wheel library symlinks and CUDA_HOME alignment), and the verl dev-version check. Use when the user wants the newest vllm/verl rather than the default pyproject extras.
license: Complete terms in LICENSE.txt
---

# Install AgentJet with latest vllm 0.26 + verl 0.8.0.dev (torch 2.11 / cu130)

This installs the **latest** stack (newer than `pyproject.toml`'s `[verl]` extra):
- torch 2.11.0+cu130, vllm 0.26.0, verl 0.8.0.dev0, transformers 5.14.1, flash-attn 2.8.3.post1

> For the **default/stable** stack (verl 0.7.1 / vllm 0.11), use the `uv-install-agentjet-swarm-server` skill instead (`uv pip install -e .[verl]`).

## When to use

- The user explicitly wants vllm 0.26 / torch 2.11 / CUDA 13.
- The default `[verl]` extra versions are too old.
- A pinned `requirements_stable_vllm_0_26.txt` is present in the repo root.

## Prerequisites

- NVIDIA GPU sm80+ (A100/H100), CUDA 13, Python 3.12, `uv` installed.
- The repo's `requirements_stable_vllm_0_26.txt` must contain:
  - `verl==0.8.0.dev0` (NOT `0.8.0` — the release wheel lacks `verl.experimental.dataset.sampler`)
  - `nvidia-cuda-nvcc==13.3.73` and `nvidia-cuda-runtime==13.3.29` (same major.minor 13.3 — required for flash-attn compile)
  - `flash-attn==2.8.3.post1`

Verify these lines exist; if not, fix the txt first.

---

# Step 1: Create venv on local disk

Create the venv on `/tmp` (local disk), NOT on a network filesystem:

```bash
export UV_CACHE_DIR=/tmp/uv-cache-local
export UV_LINK_MODE=copy
ulimit -n 65536

uv venv /tmp/_venv_local --python 3.12
source /tmp/_venv_local/bin/activate
uv pip install -r requirements_stable_vllm_0_26.txt --no-deps
```

`--no-deps` is required: the txt pins all 318 packages (including both cu12/cu13 nvidia libs), and re-resolving would reorder them causing same-path `.so` collisions.

Verify:

```bash
/tmp/_venv_local/bin/python -c "import torch,vllm,transformers; print(torch.__version__, vllm.__version__, transformers.__version__)"
# 2.11.0+cu130 0.26.0 5.14.1
```

---

# Step 2: Compile flash-attn from source

flash-attn 2.8.3.post1 has no prebuilt wheel for torch 2.11+cu130, and verl defaults to `flash_attention_2` — without it, model loading fails with `ImportError: FlashAttention2 ... not installed`.

## 2.1 Add library symlinks + link path

pip-installed nvidia wheels ship `libcudart.so.13` (for runtime dlopen) but NOT `libcudart.so` (for the linker `ld -lcudart`). Add the missing symlinks:

```bash
NVLIB=/tmp/_venv_local/lib/python3.12/site-packages/nvidia
for so in $NVLIB/cu13/lib/*.so.*; do ln -sf "$(basename "$so")" "${so%.*}" 2>/dev/null; done
export LD_LIBRARY_PATH=$(find $NVLIB -maxdepth 3 -type d -name 'lib' | tr '\n' ':')$NVLIB/cu13/cccl/lib:$LD_LIBRARY_PATH
```

## 2.2 Set CUDA toolchain to cu13

```bash
CU13=/tmp/_venv_local/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU13
export CUDA_PATH=$CU13
export PATH=$CU13/bin:$PATH
```

`CUDA_HOME` must point to the venv's `nvidia/cu13` (nvcc 13.3). Otherwise it defaults to system `/usr/local/cuda` (12.4), which mismatches torch cu130 and torch's `cpp_extension` refuses to compile.

## 2.3 Compile flags

```bash
export TORCH_CUDA_ARCH_LIST="8.0"    # A100=sm80; set to "9.0" for H100
export FLASH_ATTN_CUDA_ARCHS="80"
export MAX_JOBS=$(nproc)             # parallel jobs; 90-core machine can use 256
export NVCC_THREADS=1
ulimit -n 65536
```

`MAX_JOBS` controls parallelism — higher is much faster (256 → ~7 min; 4 → 40+ min).

## 2.4 Build

```bash
source /tmp/_venv_local/bin/activate
uv pip install flash-attn==2.8.3.post1 --no-build-isolation --no-deps --force-reinstall --no-cache
```

If GitHub/PyPI is slow, use proxychains:

```bash
proxychains4 -f /root/.proxychains/proxychains.conf \
  uv pip install flash-attn==2.8.3.post1 --no-build-isolation --no-deps --force-reinstall --no-cache
```

## 2.5 Monitor

```bash
BD=$(find /tmp -name '.ninja_log' -path '*flash*' | head -1)
tail -f "$BD"; wc -l < "$BD"   # compiled count, total 73
```

## 2.6 Verify

```bash
/tmp/_venv_local/bin/python -c "import flash_attn; print(flash_attn.__version__)"
# 2.8.3.post1
```

---

# Step 3: Verify verl dev submodule

```bash
/tmp/_venv_local/bin/python -c "from verl.experimental.dataset.sampler import AbstractSampler; print('OK')"
```

If this fails, `verl` is the release `0.8.0` wheel — fix `requirements_stable_vllm_0_26.txt` to `verl==0.8.0.dev0` and reinstall.

---

# Step 4: Run the benchmark test

```bash
cd <project root>
export VERL_PYTHON=/tmp/_venv_local/bin/python
export CUDA_HOME=/tmp/_venv_local/lib/python3.12/site-packages/nvidia/cu13
NVLIB=/tmp/_venv_local/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $NVLIB -maxdepth 3 -type d -name 'lib' | tr '\n' ':')$NVLIB/cu13/cccl/lib:$LD_LIBRARY_PATH
ulimit -n 65536

/tmp/_venv_local/bin/python -m pytest -s \
  tests/bench/benchmark_math/execute_benchmark_math.py::TestBenchmarkMath::test_01_begin_verl
```

Subprocess logs: `saved_experiments/companion_logs/companion/*.log`.

---

# Step 5: Daily usage

Set environment before training/compiling:

```bash
export VENV=/tmp/_venv_local
export CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
NVLIB=$VENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $NVLIB -maxdepth 3 -type d -name 'lib' | tr '\n' ':')$NVLIB/cu13/cccl/lib:$LD_LIBRARY_PATH
ulimit -n 65536
source $VENV/bin/activate
```

⚠️ `/tmp` is local disk and is wiped on machine restart. To persist, `cp -a /tmp/_venv_local .venv`.

---

# Troubleshooting

| Symptom | Fix |
|------|------|
| `ModuleNotFoundError: verl.experimental.dataset` | verl is `0.8.0`, change to `0.8.0.dev0` in txt |
| `FlashAttention2 ... not installed` | flash-attn not built, see Step 2 |
| `CUDA compiler and CUDA toolkit headers are incompatible` | nvcc/cudart minor mismatch — both must be 13.3 in txt |
| `cannot find -lcudart` | Step 2.1 symlinks + LD_LIBRARY_PATH |
| `detected CUDA version (12.4) mismatches PyTorch (13.0)` | Step 2.2 CUDA_HOME must point to cu13 |
| Compilation too slow | Step 2.3 raise MAX_JOBS |
| `verify_python_env` rejects env | add installed version to `ajet/utils/launch_utils.py:306` list |
| Test `Process with PGID ... is not running` | check companion log for subprocess crash |
