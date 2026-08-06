#!/usr/bin/env bash
# e2b ATB coding-agent RL — swarm client 启动脚本 (8851).
# 前置 (在专用 aimux session 里, 先 source .venv 再提 ulimit):
#   1) judge 独立转发进程:  nohup python judge_forwarder.py &   (JUDGE_MODEL_SERVER=dashscope + JUDGE_DASHSCOPE_KEY)
#   2) swarm server session: ulimit -n 1048576; ajet-swarm start --swarm-port=10086
# 然后 bash run_e2b_atbench.sh
set -euo pipefail

HELLO=/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet2
MAT=$HELLO/tmp/coding-agent-material
V=$HELLO/.venv
cd "$HELLO"

# shellcheck disable=SC1091
source "$V/bin/activate"
ulimit -n 1048576 || true
# SwanLab 凭据 (可选)
for f in "$HELLO/load_research_env.sh" "$HELLO/tutorial/load_research_env.sh"; do
  [ -f "$f" ] && { source "$f" || true; break; }
done

# E2B 沙箱 (PAI-EAS template)
export E2B_DOMAIN=sandbox01.vpc.cn-hongkong.pai-eas.aliyuncs.com
export E2B_API_KEY=***REMOVED:E2B_API_KEY***
export E2B_VALIDATE_API_KEY=false
export SLIME_AGENT_E2B_TEMPLATE=agentscope-qwenpaw-0604

# 网络: 沙箱回连本机 + judge 独立转发进程 (judge_forwarder.py → dashscope)
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-10.29.255.115}"
export JUDGE_FORWARDER_PORT="${JUDGE_FORWARDER_PORT:-18005}"
export JUDGE_MODEL_SERVER="${JUDGE_MODEL_SERVER:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export JUDGE_DASHSCOPE_KEY="${JUDGE_DASHSCOPE_KEY:-}"

# coding-agent-material 路径 + 二进制
export E2B_ATBENCH_MATERIAL="$MAT"
export CC_CLAUDE_BIN="$MAT/claudecode_binary/claude"
export CC_TMUX_BIN="$MAT/tmux_binary/tmux"
export CC_TMUX_LIBEVENT="$MAT/tmux_binary/libevent_core-2.1.so.7"
export CC_DRIVER_DIR="$MAT/claudecode_py_driver"

# 模型: 策略 (被训练, solver 直连 interchange) + 固定 judge
export E2B_ATBENCH_POLICY_MODEL="${E2B_ATBENCH_POLICY_MODEL:-Qwen3.6-35B-A3B}"
export E2B_ATBENCH_JUDGE_MODEL="${E2B_ATBENCH_JUDGE_MODEL:-glm-5.2}"
export CC_SOLVER_MODEL="$E2B_ATBENCH_POLICY_MODEL"
export CC_JUDGE_MODEL="$E2B_ATBENCH_JUDGE_MODEL"
export CC_SOLVER_TIMEOUT="${CC_SOLVER_TIMEOUT:-1800}"
export CC_JUDGE_TIMEOUT="${CC_JUDGE_TIMEOUT:-1800}"
export E2B_ATBENCH_EPISODE_TIMEOUT="${E2B_ATBENCH_EPISODE_TIMEOUT:-3600}"

# swarm + 训练模型路径
export AJET_SWARM_URL="${AJET_SWARM_URL:-http://localhost:10086}"
export REMOTE_MODEL_PATH="${REMOTE_MODEL_PATH:-/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen3___6-35B-A3B}"
export FORCE_RESTART_SWARM_ENGINE="${FORCE_RESTART_SWARM_ENGINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

echo "=== e2b_atbench swarm client (8851) ==="
echo "  policy(solver): $E2B_ATBENCH_POLICY_MODEL   judge: $E2B_ATBENCH_JUDGE_MODEL"
echo "  solver: 直连 interchange (假设有 /v1/messages)"
echo "  judge:  judge_forwarder ${ADAPTER_PUBLIC_HOST}:${JUDGE_FORWARDER_PORT} -> $JUDGE_MODEL_SERVER"
echo "  batch_size=4  num_repeat=4  max_parallel=16  (in client)"
echo "  swarm: $AJET_SWARM_URL   model: $REMOTE_MODEL_PATH"
echo "  material: $MAT"
echo "------------------------------------------------------"

exec python -m tutorial.e2b_atbench.e2b_atbench_swarm_client
