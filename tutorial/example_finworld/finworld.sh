#!/bin/bash
set -e  
#===============================================================================
# 配置区域 - 用户只需修改这里
#===============================================================================
SUFFIX="ajet_finworld"     # 实验后缀，影响所有日志和实验名称
PREFIX="open"                        # 实验前缀，影响日志和实验所在文件夹


# OpenJudge 模型配置
OPENJUDGE_LLM='qwen-flash'        # OpenJudge 评分模型
RM_LLM='qwen-max'                 # RM Gallery 评分模型
JUDGE_CONCURRENCY=10

# 奖励权重配置
RM_WEIGHT=0.4
CITATION_AUDIT_WEIGHT=0.2
REPORT_RESOLUTION_WEIGHT=0.2
TRAJECTORY_FAITHFULNESS_WEIGHT=0.2

# API密钥配置（从 .env 文件加载，不要硬编码）
# 配置
NUM_REPEAT=4        # group size，每个query rollout NUM_REPEAT次
TRAIN_BATCH_SIZE=32 
NUM_STEPS=6         # 每个样本step轮数

ADDR="22.17.31.142"
MCP_PORT="8040"

# 修改：配置文件生成路径，现在动态生成到 yaml 目录下
export AJET_ROOT="/mnt/data_cpfs/taoshuchang.tsc/deepresearch/AgentJet"
CONFIG_FILE="${AJET_ROOT}/tutorial/example_finworld/yaml/finworld_${SUFFIX}.yaml"
CONFIG_TEMPLATE="tutorial/example_finworld/yaml_template/finworld_template.yaml"
#===============================================================================
# 环境配置区域
#===============================================================================

cd ${AJET_ROOT}
source .venv/bin/activate
# API密钥配置 - 从 .env 文件加载
ENV_FILE="${AJET_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo -e "\033[32m已从 $ENV_FILE 加载环境变量\033[0m"
else
    echo -e "\033[31m警告: 找不到 .env 文件: $ENV_FILE\033[0m"
fi

# MongoDB 缓存配置
CACHE_TYPE="mongodb"
MONGO_URI="mongodb://${ADDR}:27117/"
MONGO_DB_NAME="finworld_cache"
MONGO_COLLECTION_NAME="tool_cache"

# FinWorld MCP 配置
LOG_DIR="${AJET_ROOT}/logs/${PREFIX}"
FINWORLD_MCP_CONFIG="${AJET_ROOT}/tutorial/example_finworld/config/mcp_finance_tool_generated.json"

# 动态生成 MCP 配置文件
mkdir -p $(dirname ${FINWORLD_MCP_CONFIG})
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
FINWORLD_TOOL_RESULT_MAX_CHARS=10000

# 其他服务配置
HF_ENDPOINT="https://hf-mirror.com"
ES_HOSTS="http://11.160.132.46:8200"

#===============================================================================
# 多机训练参数配置
#===============================================================================
if [ -z "${WORLD_SIZE}" ]; then
    echo "ERROR: WORLD_SIZE environment variable is not set!"
    echo "Please ensure this script is run in a multi-node environment (e.g., PAI-DLC, SLURM)"
    exit 1
fi

NNODES=${WORLD_SIZE}
GPUS_PER_NODE=8
EXPECTED_WORKERS=$WORLD_SIZE

#===============================================================================
# NCCL 配置
#===============================================================================
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=23
export NCCL_ASYNC_ERROR_HANDLING=1

#===============================================================================
# 自动生成的变量
#===============================================================================
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")

MASTER_IP_FILE="${LOG_DIR}/master-ip_${SUFFIX}.log"
ENV_SERVICE_LOG="${LOG_DIR}/env_service_${SUFFIX}_${CURRENT_TIME}.log"
TRAIN_LOG="${LOG_DIR}/train_${SUFFIX}_${CURRENT_TIME}.log"

#===============================================================================
# 工具函数
#===============================================================================
print_green() {
    echo -e "\033[32m$1\033[0m"
}

print_red() {
    echo -e "\033[31m$1\033[0m"
}

log() {
    echo -e "\033[0;32m[$(date '+%Y-%m-%d %H:%M:%S')]\033[0m \033[0;34m[INFO]\033[0m $1"
}

check_workers() {
    local status_output=$(ray status 2>/dev/null)
    if [ -z "$status_output" ]; then echo 0; return; fi
    local node_count=$(echo "$status_output" | grep -E "^[[:space:]]*1[[:space:]]+node_" | wc -l)
    if [ "$node_count" -gt 0 ]; then echo $node_count; return; fi
    echo $(echo "$status_output" | grep -o "node_[0-9a-f]\+" | sort -u | wc -l)
}

check_gpu_resources() {
    gpu_count=$(ray status 2>/dev/null | grep -A 10 "Resources" | grep "GPU" | awk '{print $1}' | cut -d'/' -f2)
    if [ -z "$gpu_count" ]; then echo 0; else printf "%.0f" "$gpu_count"; fi
}

#===============================================================================
# 导出环境变量
#===============================================================================
export CACHE_TYPE MONGO_URI MONGO_DB_NAME MONGO_COLLECTION_NAME
export FINWORLD_MCP_CONFIG  FINWORLD_TOOL_RESULT_MAX_CHARS
export HF_ENDPOINT ES_HOSTS 
export PYTHONPATH="${AJET_ROOT}:${PYTHONPATH}"
export RAY_CLUSTER_MODE="multi_node"
# Directory paths
export ENV_SERVICE_ROOT="/mnt/data_cpfs/taoshuchang.tsc/deepresearch/mongodb/BeyondAgent_env"

export FINWORLD_PATH="${ENV_SERVICE_ROOT}" # AgentJet 内部可能使用此路径
export FINWORLD_SCRIPT="source /mnt/data/taoshuchang.tsc/anaconda3/etc/profile.d/conda.sh && conda activate finworld_1209  && cd ${ENV_SERVICE_ROOT} && FINWORLD_TOOL_RESULT_MAX_CHARS=${FINWORLD_TOOL_RESULT_MAX_CHARS} FINWORLD_MCP_CONFIG=${FINWORLD_MCP_CONFIG} CACHE_TYPE=${CACHE_TYPE} MONGO_URI=${MONGO_URI} MONGO_DB_NAME=${MONGO_DB_NAME} MONGO_COLLECTION_NAME=${MONGO_COLLECTION_NAME} python -m env_service.env_service --env finworld --portal 0.0.0.0 --port 8080"

#===============================================================================
# 主流程
#===============================================================================
log "开始多机多卡训练: ${SUFFIX}"
log "节点数: ${NNODES}, 每节点GPU数: ${GPUS_PER_NODE}"
mkdir -p ${LOG_DIR}
mkdir -p $(dirname ${CONFIG_FILE})

#===============================================================================
# Master 节点启动流程
#===============================================================================
if [[ $HOSTNAME == *"-master-"* ]]; then
    print_green "==> This is MASTER node: $HOSTNAME"

    #---------------------------------------------------------------------------
    # 1. 动态生成配置文件 (从模板注入参数)
    #---------------------------------------------------------------------------
    log "正在从模板生成配置文件..."
    sed -e "s|{{SUFFIX}}|${SUFFIX}|g" \
        -e "s|{{PREFIX}}|${PREFIX}|g" \
        -e "s|{{MODEL_PATH}}|${MODEL_PATH}|g" \
        -e "s|{{NNODES}}|${NNODES}|g" \
        -e "s|{{RM_WEIGHT}}|${RM_WEIGHT}|g" \
        -e "s|{{CITATION_AUDIT_WEIGHT}}|${CITATION_AUDIT_WEIGHT}|g" \
        -e "s|{{OPENJUDGE_LLM}}|${OPENJUDGE_LLM}|g" \
        -e "s|{{RM_LLM}}|${RM_LLM}|g" \
        -e "s|{{JUDGE_CONCURRENCY}}|${JUDGE_CONCURRENCY}|g" \
        -e "s|{{REPORT_RESOLUTION_WEIGHT}}|${REPORT_RESOLUTION_WEIGHT}|g" \
        -e "s|{{TRAJECTORY_FAITHFULNESS_WEIGHT}}|${TRAJECTORY_FAITHFULNESS_WEIGHT}|g" \
        -e "s|{{NUM_REPEAT}}|${NUM_REPEAT}|g" \
        -e "s|{{NUM_STEPS}}|${NUM_STEPS}|g" \
        -e "s|{{TRAIN_BATCH_SIZE}}|${TRAIN_BATCH_SIZE}|g" \
        -e "s|{{TRAIN_DATA_PATH}}|${TRAIN_DATA_PATH}|g" \
        -e "s|{{VAL_DATA_PATH}}|${VAL_DATA_PATH}|g" \
        -e "s|{{TRAIN_REF_ANS_PATH}}|${TRAIN_REF_ANS_PATH}|g" \
        -e "s|{{VAL_REF_ANS_PATH}}|${VAL_REF_ANS_PATH}|g" \
        ${AJET_ROOT}/${CONFIG_TEMPLATE} > ${CONFIG_FILE}
    
    print_green "配置文件已生成: ${CONFIG_FILE}"
    print_green "参数确认: RM=${RM_WEIGHT}, Citation=${CITATION_AUDIT_WEIGHT}, OpenJudge=${OPENJUDGE_LLM}, RM_LLM=${RM_LLM}"

    #---------------------------------------------------------------------------
    # 2. 清理和初始化 Ray
    #---------------------------------------------------------------------------
    rm -f "$MASTER_IP_FILE"
    ray stop --force || true
    sleep 3

    #---------------------------------------------------------------------------
    # 4. 启动 Ray Head
    #---------------------------------------------------------------------------
    print_green "Starting Ray head node at $MASTER_ADDR"
    ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8
    sleep 10
    echo $MASTER_ADDR > $MASTER_IP_FILE

    #---------------------------------------------------------------------------
    # 9. 启动训练任务
    #---------------------------------------------------------------------------
    print_green "Starting training job..."
    source .venv/bin/activate

    export RAY_ADDRESS="ray://localhost:10001"
    export env_url="http://${MASTER_ADDR}:8080"
    export env_type="finworld"

    print_green "==================================="
    print_green "Training Configuration"
    print_green "Total GPUs: $((NNODES * GPUS_PER_NODE))"
    print_green "Log: ${TRAIN_LOG}"
    print_green "==================================="

    # 启动训练任务
    python ajet/launcher.py \
        --with-finworld \
        --conf ${CONFIG_FILE} \
        --backbone="verl" \
        --debug="TAG_A" \
        2>&1 | tee ${TRAIN_LOG}
    
    # 保留原脚本末尾的 CLI 调用
    ajet --conf ${CONFIG_FILE} --backbone='verl'

#===============================================================================
# Worker 节点启动流程 (逻辑保持不变)
#===============================================================================
else
    print_green "==> This is WORKER node: $HOSTNAME"
    # [此处保留原脚本中 Worker 节点等待 Master IP 和连接 Ray 的完整逻辑]
    while [ ! -f $MASTER_IP_FILE ]; do sleep 5; done
    MASTER_ADDR=$(cat $MASTER_IP_FILE)
    ray stop || true
    ray start --address $MASTER_ADDR:6379 --num-gpus 8
    while true; do sleep 60; done
fi