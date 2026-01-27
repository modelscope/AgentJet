#!/bin/bash
set -e
#===============================================================================
# 1. 配置区域 - 用户只需修改这里
#===============================================================================
SUFFIX="ajet_deep_finance"     # 实验后缀，影响所有日志和实验名称
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

# 训练参数配置
NUM_REPEAT=4        # group size，每个query rollout NUM_REPEAT次
TRAIN_BATCH_SIZE=32  # 训练batchsize
NUM_STEPS=6         # 每个样本step轮数
DEEPFINANCE_TOOL_RESULT_MAX_CHARS=10000

# 主目录

NNODES=${WORLD_SIZE}

# 涉密的配置（API_KEY以及模型、数据位置）从.env读取
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
# 2. 动态生成配置文件 (从yaml template生成yaml)
#===============================================================================
# 修改：配置文件生成路径，现在动态生成到 yaml 目录下
CONFIG_TEMPLATE="tutorial/example_deep_finance/yaml_template/deep_finance_template.yaml"
CONFIG_FILE="${AJET_ROOT}/tutorial/example_deep_finance/yaml/${SUFFIX}.yaml"
mkdir -p $(dirname ${CONFIG_FILE})

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
    -e "s|{{ENV_SERVICE_URL}}|${ENV_SERVICE_URL}|g" \
    -e "s|{{TRAIN_REF_ANS_PATH}}|${TRAIN_REF_ANS_PATH}|g" \
    -e "s|{{VAL_REF_ANS_PATH}}|${VAL_REF_ANS_PATH}|g" \
    -e "s|{{CKPT_SAVE_PATH}}|${CKPT_SAVE_PATH}|g" \
    ${AJET_ROOT}/${CONFIG_TEMPLATE} > ${CONFIG_FILE}

echo "配置文件已生成: ${CONFIG_FILE}"
echo "参数确认: RM=${RM_WEIGHT}, Citation=${CITATION_AUDIT_WEIGHT}, OpenJudge=${OPENJUDGE_LLM}, RM_LLM=${RM_LLM}"

#===============================================================================
# 3. 环境配置
#===============================================================================
# MongoDB 缓存配置
CACHE_TYPE="mongodb"
MONGO_URI="mongodb://${ADDR}:27117/"
MONGO_DB_NAME="finworld_cache"
MONGO_COLLECTION_NAME="tool_cache"
export CACHE_TYPE MONGO_URI MONGO_DB_NAME MONGO_COLLECTION_NAME

# DeepFinance MCP 配置
DEEPFINANCE_MCP_CONFIG="${AJET_ROOT}/tutorial/example_deep_finance/config/mcp_finance_tool_generated.json"

# 动态生成 MCP 配置文件
mkdir -p $(dirname ${DEEPFINANCE_MCP_CONFIG})
cat > ${DEEPFINANCE_MCP_CONFIG} << EOF
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
export DEEPFINANCE_MCP_CONFIG  DEEPFINANCE_TOOL_RESULT_MAX_CHARS

# 其他服务配置
HF_ENDPOINT="https://hf-mirror.com"
ES_HOSTS="http://11.160.132.46:8200"
export HF_ENDPOINT ES_HOSTS

# log 文件位置
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="${AJET_ROOT}/logs/${PREFIX}"
MASTER_IP_FILE="${LOG_DIR}/master-ip_${SUFFIX}.log"
ENV_SERVICE_LOG="${LOG_DIR}/env_service_${SUFFIX}_${CURRENT_TIME}.log"
TRAIN_LOG="${LOG_DIR}/train_${SUFFIX}_${CURRENT_TIME}.log"

# 多机训练参数配置
GPUS_PER_NODE=8
EXPECTED_WORKERS=$WORLD_SIZE


#===============================================================================
# 4. 工具函数 以及 NCCL 配置（固定）
#===============================================================================
print_green() {
    echo -e "\033[32m$1\033[0m"
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


export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=23
export NCCL_ASYNC_ERROR_HANDLING=1

#===============================================================================
# 5. 工具envservice 环境变量
#===============================================================================

export PYTHONPATH="${AJET_ROOT}:${PYTHONPATH}"
export RAY_CLUSTER_MODE="multi_node"


#===============================================================================
# 6. 主流程
#===============================================================================
log "开始多机多卡训练: ${SUFFIX}"
log "节点数: ${NNODES}, 每节点GPU数: ${GPUS_PER_NODE}"
mkdir -p ${LOG_DIR}
mkdir -p $(dirname ${CONFIG_FILE})

#===============================================================================
#  6.1 Master 节点启动流程
#===============================================================================
# 启动训练任务（最核心）
python ajet/launcher.py \
    --conf ${CONFIG_FILE} \
    --backbone="debug" \
    2>&1 | tee ${TRAIN_LOG}
