#!/bin/bash
set -e  

#===============================================================================
# 配置区域
#===============================================================================
SUFFIX="cc_rm4_res2cit2fai2_30b_single"  # 实验后缀
PREFIX="open"                            # 实验前缀

ADDR="127.0.0.1"                         # 单机建议使用回环地址
MCP_PORT="8040"
export CONFIG_FILE_NAME="tutorial/example_finworld/finworld_single.yaml"
export AJET_ROOT="/mnt/data_cpfs/taoshuchang.tsc/deepresearch/AgentJet"
export BEYONDAGENT_ROOT="${AJET_ROOT}"   # 假设在同一目录下，若不同请手动修改

#===============================================================================
# 环境初始化
#===============================================================================
cd ${AJET_ROOT}

# 加载 .env
ENV_FILE="${AJET_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a && source "$ENV_FILE" && set +a
    echo -e "\033[32m已从 $ENV_FILE 加载环境变量\033[0m"
fi

# 1. 激活主虚拟环境 (uv)
source .venv/bin/activate

# 2. 动态获取 Conda 基础路径，用于解决 PTY 找不到 conda 的问题
CONDA_BASE_PATH=$(conda info --base)

#===============================================================================
# 服务与路径配置
#===============================================================================
# MongoDB 配置
export CACHE_TYPE="mongodb"
export MONGO_URI="mongodb://${ADDR}:27117/"
export MONGO_DB_NAME="finworld_cache"
export MONGO_COLLECTION_NAME="tool_cache"

# FinWorld 配置
LOG_DIR="${AJET_ROOT}/logs/${PREFIX}"
mkdir -p ${LOG_DIR}
export FINWORLD_MCP_CONFIG="${AJET_ROOT}/tutorial/example_finworld/config/mcp_finance_tool_generated.json"
export FINWORLD_TOOL_RESULT_MAX_CHARS=10000

# 动态生成 MCP 配置
cat > ${FINWORLD_MCP_CONFIG} << EOF
{
    "mcpServers": {
      "flowllm": {
        "transport": "sse",
        "url": "http://${ADDR}:${MCP_PORT}/sse",
        "timeout": 600,
        "sse_read_timeout": 1200
      }
    }
}
EOF

# 环境变量导出
export HF_ENDPOINT="https://hf-mirror.com"
export ES_HOSTS="http://11.160.132.46:8200"
export PYTHONPATH="${AJET_ROOT}:${BEYONDAGENT_ROOT}:${PYTHONPATH}"
export RAY_CLUSTER_MODE="single_node"

# 关键修复：在脚本中显式加载 conda.sh 以供 PTY 子进程使用
export FINWORLD_PATH="${BEYONDAGENT_ROOT}"
export FINWORLD_SCRIPT="source ${CONDA_BASE_PATH}/etc/profile.d/conda.sh && conda activate finworld_1209 && cd ${BEYONDAGENT_ROOT} && python -m env_service.env_service --env finworld --portal 0.0.0.0 --port 8080"

#===============================================================================
# 启动 Ray 本地集群
#===============================================================================
echo -e "\033[32m正在初始化单机 Ray 环境...\033[0m"
ray stop --force || true
sleep 2

# 启动单机 Head 节点，分配 8 张 GPU
ray start --head --num-gpus 8

# 等待 Ray 就绪
sleep 5
if ! ray status > /dev/null 2>&1; then
    echo -e "\033[31m错误: Ray 启动失败\033[0m"
    exit 1
fi

#===============================================================================
# 启动训练
#===============================================================================
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
CONFIG_FILE="${AJET_ROOT}/${CONFIG_FILE_NAME}"
TRAIN_LOG="${LOG_DIR}/train_${SUFFIX}_${CURRENT_TIME}.log"

# 设置训练所需的运行时变量
export RAY_ADDRESS="ray://localhost:10001"
export env_url="http://127.0.0.1:8080"
export env_type="finworld"

echo -e "\033[32m===================================\033[0m"
echo -e "\033[32m开始单机运行: ${SUFFIX}\033[0m"
echo -e "\033[32m日志文件: ${TRAIN_LOG}\033[0m"
echo -e "\033[32m===================================\033[0m"

# 启动 Launcher
python ajet/launcher.py \
    --with-finworld \
    --conf ${CONFIG_FILE} \
    --backbone="verl" \
    2>&1 | tee ${TRAIN_LOG}