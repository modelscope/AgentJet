#!/usr/bin/env bash
# Launch Qwen3.6-35B-A3B on vLLM (TP=8) with a raised fd limit.
set -u
ulimit -n 1048576 2>/dev/null || ulimit -n 102400 2>/dev/null || true
echo "launched-nofile=$(ulimit -n)"
MODEL="/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen3___6-35B-A3B"
exec python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --dtype auto \
  --served-model-name Qwen3.6-35B-A3B \
  --port 2888
