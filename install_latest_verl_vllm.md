# 安装 verl 0.8.0.dev + vllm 0.26 + torch 2.11(cu130)

依赖清单见 `requirements_stable_vllm_0_26.txt`(已含正确版本:verl 0.8.0.dev0、nvidia-cuda-nvcc/cuda-runtime 同为 13.3、flash-attn 2.8.3.post1)。

本文档只写 txt 无法表达的步骤:建 venv、编译 flash-attn、跑测试。

验证环境:A100 ×8、CUDA 13、Python 3.12.9。

---

## 1. 建 venv 装依赖

venv 装在 `/tmp`(本地盘):

```bash
export UV_CACHE_DIR=/tmp/uv-cache-local
export UV_LINK_MODE=copy
ulimit -n 65536

uv venv /tmp/_venv_local --python 3.12
source /tmp/_venv_local/bin/activate
uv pip install -r requirements_stable_vllm_0_26.txt --no-deps
```

`--no-deps`:txt 已 pin 死全部包,避免 uv 重解析打乱 cu12/cu13 顺序。

验证:

```bash
/tmp/_venv_local/bin/python -c "import torch,vllm,transformers; print(torch.__version__, vllm.__version__, transformers.__version__)"
# 2.11.0+cu130 0.26.0 5.14.1
```

---

## 2. 编译安装 flash-attn(源码)

flash-attn 2.8.3.post1 在 PyPI 只有源码包,且无 torch 2.11+cu130 预编译 wheel,必须源码编译。verl 默认 attention 是 `flash_attention_2`,没装会模型加载报错。

### 2.1 补库软链 + 链接路径

nvidia wheel 只有 `libcudart.so.13`,缺链接器要的 `libcudart.so`:

```bash
NVLIB=/tmp/_venv_local/lib/python3.12/site-packages/nvidia
for so in $NVLIB/cu13/lib/*.so.*; do ln -sf "$(basename "$so")" "${so%.*}" 2>/dev/null; done
export LD_LIBRARY_PATH=$(find $NVLIB -maxdepth 3 -type d -name 'lib' | tr '\n' ':')$NVLIB/cu13/cccl/lib:$LD_LIBRARY_PATH
```

### 2.2 设 CUDA 工具链

```bash
CU13=/tmp/_venv_local/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU13
export CUDA_PATH=$CU13
export PATH=$CU13/bin:$PATH
```

`CUDA_HOME` 必须指 cu13(nvcc 13.3),否则默认找系统 cuda 12.4 和 torch cu130 冲突。

### 2.3 编译参数

```bash
export TORCH_CUDA_ARCH_LIST="8.0"    # A100=sm80
export FLASH_ATTN_CUDA_ARCHS="80"
export MAX_JOBS=256                  # 90核机器;核少设成核数
export NVCC_THREADS=1
ulimit -n 65536
```

`MAX_JOBS=256` 编 73 个内核约 7 分钟;`=4` 要 40+ 分钟。

### 2.4 编译

```bash
source /tmp/_venv_local/bin/activate

# 直连:
uv pip install flash-attn==2.8.3.post1 --no-build-isolation --no-deps --force-reinstall --no-cache

# 走代理:
proxychains4 -f /root/.proxychains/proxychains.conf \
  uv pip install flash-attn==2.8.3.post1 --no-build-isolation --no-deps --force-reinstall --no-cache
```

### 2.5 监控进度

```bash
BD=$(find /tmp -name '.ninja_log' -path '*flash*' | head -1)
tail -f "$BD"; wc -l < "$BD"   # 已编译数,总 73
```

### 2.6 验证

```bash
/tmp/_venv_local/bin/python -c "import flash_attn; print(flash_attn.__version__)"
# 2.8.3.post1
```

---

## 3. 跑测试

```bash
cd /mnt/data_cpfs/qingxu.fu/agentjet
export VERL_PYTHON=/tmp/_venv_local/bin/python
export CUDA_HOME=/tmp/_venv_local/lib/python3.12/site-packages/nvidia/cu13
NVLIB=/tmp/_venv_local/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $NVLIB -maxdepth 3 -type d -name 'lib' | tr '\n' ':')$NVLIB/cu13/cccl/lib:$LD_LIBRARY_PATH
ulimit -n 65536

/tmp/_venv_local/bin/python -m pytest -s \
  tests/bench/benchmark_math/execute_benchmark_math.py::TestBenchmarkMath::test_01_begin_verl
```

子进程日志:`saved_experiments/companion_logs/companion/*.log`。

---

## 4. 日常使用

训练/编译前必设环境:

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

⚠️ `/tmp` 是本地盘,机器重启会丢,需持久化时 `cp -a` 迁到 CPFS 的 `.venv`。

---

## 5. 踩坑速查

| 症状 | 解法 |
|------|------|
| `ModuleNotFoundError: verl.experimental.dataset` | verl 装成 0.8.0,改 0.8.0.dev0 |
| `FlashAttention2 ... not installed` | 没装 flash-attn,第 2 节 |
| `CUDA compiler and CUDA toolkit headers are incompatible` | nvcc 和 cudart minor 不一致,txt 里改 13.3 |
| `cannot find -lcudart` | 第 2.1 补软链 + LD_LIBRARY_PATH |
| `detected CUDA version (12.4) mismatches PyTorch (13.0)` | 第 2.2 CUDA_HOME 指向 cu13 |
| 编译太慢 | 第 2.3 MAX_JOBS 调大 |
| `verify_python_env` 拒绝 | `ajet/utils/launch_utils.py:306` 版本列表加装的版本 |
