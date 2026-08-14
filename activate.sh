source /root/miniconda3/etc/profile.d/conda.sh
conda activate /tmp/conda_venv
beast_logger


ajet-swarm top

-------------- start training --------------

# ── 干净重启: 先停 tmux(swarm server/client/forwarder 不再派发新 rollout),
#    再逐节点彻底释放 GPU + ray, 最后验证显存已清空才启动 ──
tmux kill-session -t e2b_train 2>/dev/null

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

----------------------------


/tmp/conda_venv/bin/python get_swarm_client_apikey_and_url.py




export PYTHONNOUSERSITE="1"
export FLASHINFER_DISABLE_VERSION_CHECK="1"
export TRITON_CACHE_DIR="/dev/shm/triton_e2b"
export VLLM_CACHE_ROOT="/dev/shm/vllm_cache_e2b"
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/torchinductor_e2b"
source /tmp/e2b_env_new.sh
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /tmp/conda_venv
python -m vllm.entrypoints.cli.main serve \
    /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen3___6-35B-A3B \
    --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --dtype auto \
    --served-model-name Qwen3.6-35B-A3B --port 2888



任务：验证claudecode->vllm的token序列，和claudecode->agentjet->vllm的token序列格式是否一致

claudecode->agentjet->vllm的token序列已经记录在（问题是一个简单的 【list all files】)
ccd6e8336d99440a818a125aeb701f72_cached_sample.pkl

你的任务：验证claudecode->vllm的token序列
1. 修改 launch_vllm.sh，启动 vllm 并设法记录 token 序列
2. 修改 ~/.claude/settings.vllm.json
3. 运行 claudecode->vllm，使用settings.vllm.json运行claudecode，记录 token 序列
