source /root/miniconda3/etc/profile.d/conda.sh
conda activate /tmp/conda_venv
beast_logger


ajet-swarm top



/tmp/conda_venv/bin/ray stop --force
tmux kill-session -t e2b_train
# worker 也清（它们可能还连着旧 GCS）
for ip in 10.29.255.112 10.29.255.114 10.29.255.116; do
  ssh root@$ip "/tmp/conda_venv/bin/ray stop --force 2>/dev/null; pkill -9 -f raylet; pkill -9 -f gcs_server; pkill -9 -f ray::"
done



/tmp/conda_venv/bin/python start_formal_training.py
tmux attach -t e2b_train



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