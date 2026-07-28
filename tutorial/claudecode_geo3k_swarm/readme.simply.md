
tmux window one:

```bash
tmux new -s a_session_running_swarm_server
source .venv/bin/activate

tmux new -s a_session_running_swarm_client
source .venv/bin/activate
export AJET_SWARM_URL=http://localhost:10086
export REMOTE_MODEL_PATH=/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-VL-7B-Instruct
export GEO3K_DATASET_PATH=/mnt/data_cpfs/model_cache/modelscope/dataset/hiyouga/geometry3k
export FORCE_RESTART_SWARM_ENGINE=1
python -m tutorial.claudecode_geo3k_swarm.geo3k
```
