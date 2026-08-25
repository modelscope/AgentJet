#!/usr/bin/env bash
# ============================================================================
# gpu_monitor.sh — 多节点 GPU 状态一览 (master 本地 + 3 workers)
#
# 用法:
#   bash scripts/gpu_monitor.sh              # 单次快照 (默认本地+3 workers)
#   bash scripts/gpu_monitor.sh -w 5         # watch 模式, 每 5s 刷新 (Ctrl-C 退出)
#   bash scripts/gpu_monitor.sh -w 5 -n 10   # watch 模式, 刷 10 次后退出
#   bash scripts/gpu_monitor.sh --json       # 单次快照, JSON 输出 (供程序消费)
#   NODES="local 10.29.255.112" bash scripts/gpu_monitor.sh   # 只看部分节点
#     ("local" 关键字代表本机; 其余为可 SSH 免密到达的 IP/hostname)
#
# 依赖: 到 worker 节点的 SSH 免密 (同 activate.sh 的 WORKERS 列表), nvidia-smi.
# 输出列: ID GPU MEM(used/total) MEM% UTIL% TEMP POWER PROCS
#   MEM% >=80 红, >=50 黄; 卡上计算进程数 PROCS 便于发现泄漏/残留进程.
# ============================================================================

set -uo pipefail

# ---- 配置 ---------------------------------------------------------------
LOCAL_LABEL="$(hostname -s)"
# 可用环境变量 NODES 覆盖 (空格分隔, "local" = 本机)
DEFAULT_NODES="local 10.29.255.112 10.29.255.114 10.29.255.116"

INTERVAL=5
MAX_COUNT=0          # watch 模式下刷新次数上限, 0 = 无限
WATCH=0
USE_JSON=0

# ---- 参数解析 -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--watch)     WATCH=1; INTERVAL="${2:?}"; shift 2 ;;
        -n|--count)     MAX_COUNT="${2:?}"; shift 2 ;;
        --json)         USE_JSON=1; shift ;;
        -h|--help)      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              echo "unknown arg: $1 (see -h)" >&2; exit 1 ;;
    esac
done

read -r -a NODES_ARR <<< "${NODES:-$DEFAULT_NODES}"

# ---- 单节点采集片段 (本地 eval / 远程 ssh 执行同一份代码) -----------------
# nvidia-smi CSV 以 ", " 分隔, GPU 名含空格 → 先逐字段去空白、名字空格换 _,
# 输出竖线分隔: idx|name|mem_used|mem_total|mem_pct|util|temp|power|fan|procs
REMOTE_SNIP='
csv=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,fan.speed --format=csv,noheader,nounits 2>/dev/null)
if [[ -z "$csv" ]]; then echo "ERR|nvidia-smi failed or no GPU"; exit 0; fi
nprocs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
echo "$csv" | while IFS="," read -r idx name mu mt util temp power fan; do
    idx=$(echo $idx); mu=$(echo $mu); mt=$(echo $mt)
    util=$(echo $util); temp=$(echo $temp); power=$(echo $power); fan=$(echo $fan)
    name=$(echo $name | tr " " "_")
    mp=0; [[ "$mt" =~ ^[0-9]+$ && "$mt" -gt 0 ]] && mp=$((mu * 100 / mt))
    echo "$idx|$name|$mu|$mt|$mp|$util|$temp|$power|$fan|$nprocs"
done
'

probe_one() {  # $1 = "local" 或 IP/hostname
    local target="$1"
    if [[ "$target" == "local" ]]; then
        eval "$REMOTE_SNIP"
    else
        timeout 10 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
            "$target" "$REMOTE_SNIP" 2>/dev/null || echo "ERR|ssh failed / unreachable"
    fi
}

# 颜色 (非 tty 或 --json 时禁用)
if [[ -t 1 && $USE_JSON -eq 0 ]]; then
    C_RESET=$'\e[0m'; C_RED=$'\e[31m'; C_YEL=$'\e[33m'; C_GRN=$'\e[32m'
    C_DIM=$'\e[2m'; C_BLD=$'\e[1m'; C_CYA=$'\e[36m'
else
    C_RESET=""; C_RED=""; C_YEL=""; C_GRN=""; C_DIM=""; C_BLD=""; C_CYA=""
fi

# ---- 汇总输出 -----------------------------------------------------------
render() {
    local ts target lbl out
    ts=$(date '+%F %T')

    if [[ $USE_JSON -eq 1 ]]; then
        echo -n "{\"timestamp\":\"$ts\",\"nodes\":["
        local first_node=1
        for target in "${NODES_ARR[@]}"; do
            [[ $first_node -eq 0 ]] && echo -n ","
            first_node=0
            [[ "$target" == "local" ]] && lbl="$LOCAL_LABEL" || lbl="$target"
            echo -n "{\"node\":\"$lbl\",\"gpus\":["
            out="$(probe_one "$target")"
            if [[ "$out" == ERR* || -z "$out" ]]; then
                echo -n "{\"error\":\"${out#ERR|}\"}"
            else
                local first=1 idx name mu mt mp util temp power fan nprocs
                while IFS="|" read -r idx name mu mt mp util temp power fan nprocs; do
                    [[ $first -eq 0 ]] && echo -n ","
                    first=0
                    echo -n "{\"index\":$idx,\"name\":\"$name\",\"mem_used_mib\":$mu,\"mem_total_mib\":$mt,\"mem_pct\":$mp,\"util_pct\":$util,\"temp_c\":$temp,\"power_w\":\"$power\",\"fan\":\"$fan\",\"procs\":$nprocs}"
                done <<< "$out"
            fi
            echo -n "]}"
        done
        echo "]}"
        return
    fi

    # ---- 文本模式 ----
    echo "${C_BLD}═══════════════════════════════════════════════════════════════════════${C_RESET}"
    echo "${C_BLD} GPU 集群监控  ${C_DIM}$ts${C_RESET}"
    echo "${C_BLD}═══════════════════════════════════════════════════════════════════════${C_RESET}"

    for target in "${NODES_ARR[@]}"; do
        [[ "$target" == "local" ]] && lbl="$LOCAL_LABEL" || lbl="$target"
        echo "${C_CYA}▸ $lbl${C_RESET}"
        out="$(probe_one "$target")"
        if [[ "$out" == ERR* || -z "$out" ]]; then
            echo "  ${C_RED}✗ ${out:-unreachable}${C_RESET}"
            continue
        fi
        printf "  ${C_DIM}%-3s %-20s %15s %5s %5s %5s %8s %5s${C_RESET}\n" \
               "ID" "GPU" "MEM" "MEM%" "UT%" "TEMP" "POWER" "PROC"
        local idx name mu mt mp util temp power fan nprocs
        while IFS="|" read -r idx name mu mt mp util temp power fan nprocs; do
            local m_color="$C_GRN" u_color="$C_GRN"
            [[ "$mp" =~ ^[0-9]+$ ]] && { [[ "$mp" -ge 80 ]] && m_color="$C_RED" || { [[ "$mp" -ge 50 ]] && m_color="$C_YEL"; }; }
            [[ "$util" =~ ^[0-9]+$ ]] && { [[ "$util" -lt 10 ]] && u_color="$C_DIM"; }
            printf "  %-3s %-20s %7s/%sMiB ${m_color}%4s%%%s  ${u_color}%4s%%%s  %3sC %7sW %5s\n" \
                   "$idx" "$name" "$mu" "$mt" "$mp" "$C_RESET" "$util" "$C_RESET" "$temp" "$power" "$nprocs"
        done <<< "$out"
    done
    echo ""
}

# ---- 主循环 -------------------------------------------------------------
if [[ $WATCH -eq 0 ]]; then
    render
    exit 0
fi

count=0
while true; do
    render
    count=$((count + 1))
    if [[ $MAX_COUNT -gt 0 && $count -ge $MAX_COUNT ]]; then break; fi
    # 区块间用空行分隔而非 clear, 保留历史便于 tmux scrollback 溯源
    sleep "$INTERVAL"
done
