
## Run the swarm client

tmux window one:

```bash
tmux new -s a_session_running_swarm_server
source .venv/bin/activate
ajet-swarm start

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

## Multi-case token-id check (multiple images, cross-turn, tool, pure text)

`test_multimodal_cases.py` extends the single-image check to four cases
(defined once in `multimodal_cases.py`):

1. `two_images_one_msg` — two images in ONE user turn (150 image tokens).
   Positive control: passes even before the multi-message fix.
2. `image_text_image_turns` — images interleaved across turns
   (user img / assistant text / user img+text; 150 image tokens).
3. `pure_text` — zero images (40 prompt tokens).
4. `img_text_img_text_tool` — image / assistant tool_call / tool text /
   image+text; tool turns are text-only (150 image tokens).

Two fixes make cases 2 and 4 work end-to-end (they crashed/corrupted before):

- `ajet/context_tracker/single_agent_tracking.py::merge_multi_modal_inputs`
  concatenates each message's `multi_modal_inputs` (in message order) so the
  training-side `image_grid_thw`/`pixel_values` align with every
  `<|image_pad|>` span — not just the first image's, matching a single
  whole-conversation processor call (the vLLM request path).
- `ajet/task_rollout/native_parallel_worker.py::samples_to_dataproto`
  promotes text-only samples' 1-D position ids to the 4-channel M-RoPE
  layout when any sample in the batch is multimodal, so a mixed
  text+image batch no longer crashes when stacking `position_ids`.

Step 1 — vLLM ground truth (`vllm serve` a VL model, then):

```bash
source .venv/bin/activate
python -m tutorial.claudecode_geo3k_swarm.gen_ground_truth \
    --base-url http://localhost:8000/v1 --model <served-vl-model>
# writes token_id/<case>.md for each case
```

Step 2 — live swarm capture. Point `AJET_CAPTURE_TOKEN_IDS` at the
`token_id/` **directory** (not a file); the trainer then writes one
`token_id/<task_id>.debug.md` per case.

Server tmux:

```bash
source .venv/bin/activate
export SETUPTOOLS_USE_DISTUTILS=local
export AJET_CAPTURE_TOKEN_IDS=$PWD/token_id     # a DIRECTORY -> per-case files
ajet-swarm start --swarm-port=10086
```

Client tmux:

```bash
source .venv/bin/activate
export AJET_SWARM_URL=http://localhost:10086
export REMOTE_MODEL_PATH=/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-VL-7B-Instruct
export FORCE_RESTART_SWARM_ENGINE=1
python -m tutorial.claudecode_geo3k_swarm.test_multimodal_cases
```

Compare-only (both md sets already exist):

```bash
python -m tutorial.claudecode_geo3k_swarm.test_multimodal_cases --compare-only
```

Pass = for every case the training capture matches the vLLM ground truth on
prompt length, image-token count (150/150/0/150) and vision-span markers.

## Offline sanity check (no GPU, no swarm)

`offline_merge_check.py` proves the merge fix reproduces a single
whole-conversation processor call for a multi-message conversation
(byte-identical `pixel_values`/`image_grid_thw`), yields 4-channel
`position_ids` whose length equals `input_ids`, and places every
`<|image_pad|>` where the loss mask is 0. Runs on CPU:

```bash
source .venv/bin/activate
python -m tutorial.claudecode_geo3k_swarm.offline_merge_check
```
