
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

## Verify the image reaches training (token-id check)

`test_token_ids.py` proves an image survives the whole swarm pipeline
and lands in the *training* tokenization as real `<|image_pad|>` (151655)
tokens, by comparing:

- `token_id_io.md` — the vLLM-side request tokenization (case A = with
  image = 189 prompt tokens, 54 image tokens).
- `token_id_io.debug.md` — the training-side batch dumped at
  `ajet/backbone/trainer_verl.py:599`, written only when the
  `AJET_CAPTURE_TOKEN_IDS` env var is set in the swarm-server process.

Server tmux (note the extra export):

```bash
source .venv/bin/activate
export AJET_CAPTURE_TOKEN_IDS=$PWD/token_id_io.debug.md
ajet-swarm start --swarm-port=10086
```

Client tmux:

```bash
source .venv/bin/activate
export AJET_SWARM_URL=http://localhost:10086
export REMOTE_MODEL_PATH=/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-VL-7B-Instruct
export FORCE_RESTART_SWARM_ENGINE=1
python -m tutorial.claudecode_geo3k_swarm.test_token_ids
```

Only the prompt token pattern is compared; the model's answer (response
tokens) may differ between runs — the test passes as long as the prompt
tokenization matches (54 image tokens + vision span). Once both md files
exist you can re-run the comparison alone with `--compare-only`.
