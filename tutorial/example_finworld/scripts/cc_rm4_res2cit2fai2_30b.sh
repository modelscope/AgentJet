#!/bin/bash
set -e  
#===============================================================================
# 配置区域 - 用户只需修改这里
#===============================================================================
SUFFIX="cc_rm4_res2cit2fai2_30b"     # 实验后缀，影响所有日志和实验名称
PREFIX="open"                     # 实验前缀，影响日志和实验所在文件夹

ADDR="22.17.31.142"
MCP_PORT="8040"
export CONFIG_FILE_NAME="tutorial/example_finworld/finworld.yaml"
export AJET_ROOT="/mnt/data_cpfs/taoshuchang.tsc/deepresearch/AgentJet"
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



#===============================================================================
# 环境配置区域
#===============================================================================

# MongoDB 缓存配置
CACHE_TYPE="mongodb"
MONGO_URI="mongodb://${ADDR}:27117/"
MONGO_DB_NAME="finworld_cache"
MONGO_COLLECTION_NAME="tool_cache"

# FinWorld MCP 配置
LOG_DIR="${AJET_ROOT}/logs/${PREFIX}"
FINWORLD_MCP_CONFIG="${AJET_ROOT}/tutorial/example_finworld/config/mcp_finance_tool_generated.json"

# 动态生成 MCP 配置文件（使用 ADDR 变量）
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
# RAY_DEBUG_POST_MORTEM="1"
# DEBUG_TAGS="TAG_A"
#===============================================================================
# 自动生成的变量（不需要修改）
#===============================================================================
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
CONFIG_FILE="${AJET_ROOT}/${CONFIG_FILE_NAME}"

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

# 检查所有节点数量(包括head节点)
check_workers() {
    local status_output=$(ray status 2>/dev/null)
    if [ -z "$status_output" ]; then
        echo 0
        return
    fi
    # 统计 "1 node_" 这种格式的行数
    local node_count=$(echo "$status_output" | grep -E "^[[:space:]]*1[[:space:]]+node_" | wc -l)
    if [ "$node_count" -gt 0 ]; then
        echo $node_count
        return
    fi
    # 如果方法1失败,尝试统计包含node_的唯一ID
    node_count=$(echo "$status_output" | grep -o "node_[0-9a-f]\+" | sort -u | wc -l)
    echo $node_count
}

# 检查GPU资源是否完全就绪
check_gpu_resources() {
    gpu_count=$(ray status 2>/dev/null | grep -A 10 "Resources" | grep "GPU" | awk '{print $1}' | cut -d'/' -f2)
    if [ -z "$gpu_count" ]; then
        echo 0
    else
        printf "%.0f" "$gpu_count"
    fi
}

#===============================================================================
# 导出环境变量
# API密钥相关变量已通过 .env 文件加载并自动导出 (set -a)
#===============================================================================
export CACHE_TYPE MONGO_URI MONGO_DB_NAME MONGO_COLLECTION_NAME
export FINWORLD_MCP_CONFIG  FINWORLD_TOOL_RESULT_MAX_CHARS
export HF_ENDPOINT ES_HOSTS 
export PYTHONPATH="${AJET_ROOT}:${BEYONDAGENT_ROOT}:${PYTHONPATH}"
export RAY_CLUSTER_MODE="multi_node"



# 配置 finworld 环境服务（供 launcher.py --with-finworld 使用）
# 注意：这里可以自定义 env_service 的启动参数
export FINWORLD_PATH="${BEYONDAGENT_ROOT}"
# 如果需要传递额外参数，修改下面的命令行参数即可
# 例如：--env_file_name custom_config --debug true
# FINWORLD_SCRIPT: API密钥会从环境变量继承
export FINWORLD_SCRIPT="source /mnt/data/taoshuchang.tsc/anaconda3/etc/profile.d/conda.sh && conda activate finworld_1209 && cd ${BEYONDAGENT_ROOT} && FINWORLD_TOOL_RESULT_MAX_CHARS=${FINWORLD_TOOL_RESULT_MAX_CHARS} FINWORLD_MCP_CONFIG=${FINWORLD_MCP_CONFIG} CACHE_TYPE=${CACHE_TYPE} MONGO_URI=${MONGO_URI} MONGO_DB_NAME=${MONGO_DB_NAME} MONGO_COLLECTION_NAME=${MONGO_COLLECTION_NAME} FINWORLD_TASKS_DATA_PATH=${FINWORLD_TASKS_DATA_PATH} FINWORLD_TRAIN_REF_ANS_PATH=${FINWORLD_TRAIN_REF_ANS_PATH} python -m env_service.env_service --env finworld --portal 0.0.0.0 --port 8080"


#===============================================================================
# 主流程
#===============================================================================
log "开始多机多卡训练: ${SUFFIX}"
log "时间戳: ${CURRENT_TIME}"
log "节点数: ${NNODES}, 每节点GPU数: ${GPUS_PER_NODE}"
log "配置文件: ${CONFIG_FILE}"

# 确保日志目录存在
mkdir -p ${LOG_DIR}

#===============================================================================
# Master 节点启动流程
#===============================================================================
if [[ $HOSTNAME == *"-master-"* ]]; then
    print_green "==> This is MASTER node: $HOSTNAME"

    #---------------------------------------------------------------------------
    # 1. 清理和初始化
    #---------------------------------------------------------------------------
    rm -f "$MASTER_IP_FILE"
    print_green "Cleaned old master IP file"

    ray stop --force || true
    sleep 3
    print_green "Runtime env configuration created"

    #---------------------------------------------------------------------------
    # 4. 启动 Ray Head 节点（带 runtime_env）
    #---------------------------------------------------------------------------
    print_green "Starting Ray head node at $MASTER_ADDR with runtime_env"
    ray start --head \
        --node-ip-address $MASTER_ADDR \
        --num-gpus 8

    print_green "Waiting for Ray head to be fully ready..."
    sleep 10

    if ! ray status > /dev/null 2>&1; then
        print_red "ERROR: Ray head failed to start properly"
        exit 1
    fi
    print_green "Ray head is ready"

    # 写入 Master IP 到共享文件
    echo $MASTER_ADDR > $MASTER_IP_FILE
    print_green "Master IP written to $MASTER_IP_FILE: $MASTER_ADDR"

    #---------------------------------------------------------------------------
    # 5. 等待所有 Worker 节点加入
    #---------------------------------------------------------------------------
    print_green "Waiting for all nodes to join the Ray cluster..."
    print_green "Expected nodes: $EXPECTED_WORKERS (including head node)"

    TIMEOUT=1000
    INTERVAL=10
    ELAPSED=0

    while true; do
        current_nodes=$(check_workers)
        print_green "Current node count: $current_nodes/$EXPECTED_WORKERS"

        if [ "$current_nodes" -ge "$EXPECTED_WORKERS" ]; then
            print_green "All nodes have joined the cluster!"
            break
        fi

        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
            print_red "Timeout waiting for nodes. Only $current_nodes/$EXPECTED_WORKERS nodes joined."
            ray status
            exit 1
        fi

        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
    done

    #---------------------------------------------------------------------------
    # 6. 等待 GPU 资源就绪
    #---------------------------------------------------------------------------
    print_green "Waiting for GPU resources to be fully available..."
    EXPECTED_GPUS=$((WORLD_SIZE * 8))
    GPU_TIMEOUT=300
    GPU_ELAPSED=0

    while true; do
        current_gpus=$(check_gpu_resources)
        print_green "Current GPU count: $current_gpus/$EXPECTED_GPUS"

        if [ "$current_gpus" -eq "$EXPECTED_GPUS" ]; then
            print_green "All GPUs are available!"
            break
        fi

        if [ "$GPU_ELAPSED" -ge "$GPU_TIMEOUT" ]; then
            print_red "Timeout waiting for GPUs. Only $current_gpus/$EXPECTED_GPUS GPUs available."
            ray status
            exit 1
        fi

        sleep 5
        GPU_ELAPSED=$((GPU_ELAPSED + 5))
    done

    print_green "Final cluster status before training:"
    ray status

    #---------------------------------------------------------------------------
    # 7. 等待 Ray Dashboard 启动
    #---------------------------------------------------------------------------
    print_green "Waiting for Ray dashboard to be ready..."
    while ! curl -s http://127.0.0.1:8265 > /dev/null; do
        sleep 5
    done

    #---------------------------------------------------------------------------
    # 8. 确认 env_service 启动配置
    #---------------------------------------------------------------------------
    print_green "Environment service will be started by launcher.py --with-finworld"
    print_green "  FINWORLD_PATH: ${FINWORLD_PATH}"
    print_green "  FINWORLD_SCRIPT: ${FINWORLD_SCRIPT}"
    print_green "  Log file: ${ENV_SERVICE_LOG}"
    print_green "  Note: env_service will load .env internally from its conda environment"

    #---------------------------------------------------------------------------
    # 9. 启动训练任务
    #---------------------------------------------------------------------------
    print_green "Starting training job..."


    # 激活训练环境
    source .venv/bin/activate

    # 重新导出关键环境变量（conda activate 可能会重置）
    # API密钥已通过 .env 加载
    export CACHE_TYPE="${CACHE_TYPE}"
    export MONGO_URI="${MONGO_URI}"
    export MONGO_DB_NAME="${MONGO_DB_NAME}"
    export MONGO_COLLECTION_NAME="${MONGO_COLLECTION_NAME}"

    # 设置训练环境变量
    export RAY_ADDRESS="ray://localhost:10001"
    export env_url="http://${MASTER_ADDR}:8080"
    export env_type="finworld"
    export PYTHONPATH="${AJET_ROOT}:${PYTHONPATH}"

    # 输出配置信息
    print_green "==================================="
    print_green "Training Configuration"
    print_green "==================================="
    print_green "NNODES: $NNODES"
    print_green "GPUS_PER_NODE: $GPUS_PER_NODE"
    print_green "Total GPUs: $((NNODES * GPUS_PER_NODE))"
    print_green "env_url: $env_url"
    print_green "RAY_ADDRESS: $RAY_ADDRESS"
    print_green "Python: $(which python)"
    print_green "训练日志: ${TRAIN_LOG}"
    print_green "==================================="

    # 启动训练（多机模式下不需要 --with-ray，因为 Ray 集群已在脚本中手动启动）
    # 使用 --with-finworld 让 launcher.py 统一管理 env_service 的启动和生命周期
    python ajet/launcher.py \
        --with-finworld \
        --conf ${CONFIG_FILE} \
        --backbone="verl" \
        2>&1 | tee ${TRAIN_LOG}
    ajet --conf ${CONFIG_FILE} --backbone='verl'

#===============================================================================
# Worker 节点启动流程
#===============================================================================
else
    print_green "==> This is WORKER node: $HOSTNAME"

    #---------------------------------------------------------------------------
    # 1. 等待 Master IP 文件
    #---------------------------------------------------------------------------
    export PYTHONPATH="${AJET_ROOT}:${PYTHONPATH}"

    while [ ! -f $MASTER_IP_FILE ]; do
        print_green "Waiting for master node IP file..."
        sleep 5
    done
    sleep 2

    MASTER_ADDR=$(cat $MASTER_IP_FILE)
    print_green "Found master node at $MASTER_ADDR"

    #---------------------------------------------------------------------------
    # 2. 连接到 Ray 集群
    #---------------------------------------------------------------------------
    ray stop || true

    MAX_RETRIES=3
    RETRY_COUNT=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if ray start --address $MASTER_ADDR:6379 --num-gpus 8; then
            print_green "Worker node started successfully"
            break
        fi

        RETRY_COUNT=$((RETRY_COUNT + 1))
        print_red "Failed to start worker node, attempt $RETRY_COUNT of $MAX_RETRIES"
        sleep 10
    done

    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_red "Failed to start worker node after $MAX_RETRIES attempts"
        exit 1
    fi

    #---------------------------------------------------------------------------
    # 4. 保持连接状态
    #---------------------------------------------------------------------------
    print_green "Worker node is running, keeping alive..."
    while true; do
        sleep 60
        if ! ray status > /dev/null 2>&1; then
            print_red "Lost connection to Ray cluster, exiting..."
            break
        fi
    done
fi
