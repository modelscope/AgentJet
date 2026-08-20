
# ── 干净重启: 先停 tmux(swarm server/client/forwarder 不再派发新 rollout),
#    再逐节点彻底释放 GPU + ray, 最后验证显存已清空才启动 ──
tmux kill-session -t e2b_train 2>/dev/null

# 非交互 shell source ~/.bashrc 会被其交互守卫拦截 (密钥在文件末尾), 用 grep 精确导出
eval "$(grep -E '^export (E2B_API_KEY|JUDGE_DASHSCOPE_KEY|SWANLAB_API_KEY|SWANLAB_API_HOST|SWANLAB_WEB_HOST)=' /root/.bashrc)"

WORKERS="10.29.255.112 10.29.255.114 10.29.255.116"

# head(本机) 先停 ray: 杀掉 GCS 后 worker raylet 失去协调, 不会再拉起新 actor
/tmp/conda_venv/bin/ray stop --force 2>/dev/null
pkill -9 -x raylet 2>/dev/null
pkill -9 -x gcs_server 2>/dev/null
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null

# worker 释放 GPU + ray. 关键: 占显存的 VLLM::Worker_TP* 进程名被 setproctitle
# 改写, 不含 "ray::" -> ray stop / pkill ray:: 打不到, 显存不释放.
# 用 nvidia-smi compute-apps 直杀所有占 GPU 的 pid(专用节点, 重启时全清)才彻底.
for ip in $WORKERS; do
  echo "--- 清理 worker $ip ---"
  ssh -o ConnectTimeout=10 root@$ip "
    /tmp/conda_venv/bin/ray stop --force 2>/dev/null
    pkill -9 -x raylet 2>/dev/null
    pkill -9 -x gcs_server 2>/dev/null
    sleep 1
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
  "
done

# 验证 worker 显存已释放(干净 <100MiB); 任一卡 >1GiB 则告警
sleep 3
for ip in $WORKERS; do
  ssh -o ConnectTimeout=10 root@$ip "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null" | awk -F', ' -v ip="$ip" '{ if ($2+0 > 1024) printf "  ⚠ %s GPU%s 仍占用 %s\n", ip, $1, $2 }'
done

/tmp/conda_venv/bin/python start_formal_training.py
tmux attach -t e2b_train
