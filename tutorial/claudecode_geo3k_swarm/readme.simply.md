
## Run the swarm client

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
python -m tutorial.claudecode_geo3k_swarm.geo3k_swarm_client
```

## Test the agent standalone (no swarm server)

The per-episode agent logic lives in `geo3k_agent.py` and can be tested
directly against any OpenAI-compatible vision endpoint. If
`GEO3K_DATASET_PATH` is unset it falls back to a synthetic right-triangle
task so the multimodal path still runs.

DashScope (Aliyun compatible-mode):

```bash
source .venv/bin/activate
export DASHSCOPE_API_KEY=sk-xxxx
python -m tutorial.claudecode_geo3k_swarm.geo3k_agent \
    --backend dashscope --model qwen-vl-max-latest
```

Local vLLM OpenAI server:

```bash
source .venv/bin/activate
python -m tutorial.claudecode_geo3k_swarm.geo3k_agent \
    --backend vllm --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-VL-7B-Instruct
```
