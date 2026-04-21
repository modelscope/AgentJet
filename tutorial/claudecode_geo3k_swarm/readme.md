# Geo3K Multimodal Swarm Tutorial

This tutorial is migrated from rllm's `examples/geo3k` to AgentJet's
swarm architecture. The agent reads a geometry problem with an
accompanying figure (image) and must output the final answer in
`\boxed{}`.

## What's new in AgentJet

Geo3K needs multimodal (vision-language) inputs. AgentJet exposes a
small helper at `ajet/utils/multimodal.py` that converts dataset rows
containing PIL images / bytes / data URLs into OpenAI-compatible
chat-completions messages (with `image_url` content blocks), suitable
for forwarding to a VL-capable vLLM backend (e.g. Qwen2.5-VL).

```python
from ajet.utils.multimodal import build_multimodal_messages, extract_image
messages = build_multimodal_messages(
    system_prompt="...",
    user_text="Question...",
    image=extract_image(task.metadata),
)
```

## 1. Download the dataset (via proxychains in tmux)

The dataset is hosted on Hugging Face (`hiyouga/geometry3k`). Download
it in a tmux window so it can continue in the background:

```bash
tmux new-session -d -s geo3k_download \
  'proxychains huggingface-cli download hiyouga/geometry3k \
     --repo-type dataset \
     --local-dir /mnt/data_cpfs/model_cache/modelscope/dataset/hiyouga/geometry3k'
tmux attach -t geo3k_download   # optional — watch progress
```

Make sure you have a Qwen2.5-VL checkpoint locally too, e.g.:

```bash
tmux new-session -d -s qwen25vl \
  'proxychains huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
     --local-dir /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen2.5-VL-7B-Instruct'
```

## 2. Start the AgentJet swarm server (in tmux)

```bash
tmux new-session -d -s SWARM_SERVER_GEO3K \
  'ajet-swarm start --swarm-port=10086 2>&1 | tee /tmp/swarm-server-geo3k.log'
# Optional: watch the overwatch panel
tmux new-session -d -s SWARM_OVERWATCH_GEO3K \
  'ajet-swarm overwatch --swarm-url=http://localhost:10086'
```

## 3. Run the swarm client (in tmux)

```bash
export AJET_SWARM_URL=http://localhost:10086
export REMOTE_MODEL_PATH=/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen2.5-VL-7B-Instruct
export GEO3K_DATASET_PATH=/mnt/data_cpfs/model_cache/modelscope/dataset/hiyouga/geometry3k

tmux new-session -d -s SWARM_CLIENT_GEO3K \
  'cd /mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet2 && \
   python -m tutorial.claudecode_geo3k_swarm.geo3k 2>&1 | tee /tmp/swarm-client-geo3k.log'
```

## 4. Debug a single rollout (no training)

If you only want to sanity-check the multimodal pipeline against an
existing OpenAI-compatible VL endpoint, you can import the agent
function directly:

```python
from tutorial.claudecode_geo3k_swarm.geo3k import run_agent_and_compute_reward
# task is an ajet.schema.task.Task whose metadata has 'question',
# 'image'/'images', and 'ground_truth'/'answer'.
out = run_agent_and_compute_reward(task, base_url="http://.../v1", api_key="...")
print(out.reward, out.metadata["final_answer"])
```

## Notes on multimodal support

Server-side multimodal plumbing (not just client inference) was added:

- `ajet/utils/multimodal.py` — image normalization, OpenAI vision
  message construction, and PIL loading helpers.
- `ajet/schema/extended_msg.py` — `ExtendedMessage` carries `images`
  and `multi_modal_inputs`; messages with images tokenize through the
  HF `AutoProcessor` so `<|image_pad|>` tokens are expanded to match
  the image grid (verified: concat of per-message token_arrs equals a
  full processor call on the same conversation).
- `ajet/context_tracker/multiagent_tracking.py` — vision content
  blocks survive `disable_toolcalls=True` and are routed into
  ExtendedMessage with the images preserved.
- `ajet/task_rollout/async_llm_bridge.py` — `llm_chat_verl` detects
  vision blocks, tokenizes via the processor, and passes `image_data`
  to `async_rollout_manager.generate`; returned `multi_modal_inputs`
  propagate to the tracker.
- `ajet/task_rollout/native_parallel_worker.py` — `samples_to_dataproto`
  puts `multi_modal_inputs` (list of per-sample dicts with
  `pixel_values`, `image_grid_thw`, ...) into `non_tensor_batch`,
  which VERL's actor (`dp_actor.py`, `megatron_actor.py`) already
  consumes.
- Processor is plumbed from `AjetRayPPOTrainer` → `VerlRolloutManager`
  → `BaseRolloutManager` → `AsyncLlmBridge` / `SwarmRunner` /
  `GeneralRunner` → `MultiAgentContextTracker` → `ExtendedMessage`.

### Known limitation

- **Position IDs remain 1D.** Qwen2-VL uses 3D M-RoPE position ids
  via `verl.models.transformers.qwen2_vl.get_rope_index`. AgentJet's
  current `samples_to_dataproto` pads 1D position_ids; promoting to
  3D requires refactoring pad_sequence paths across the whole batch
  pipeline. For single-turn VL RL this still trains (vision tower
  gradients flow through `multi_modal_inputs`), but for best fidelity
  to the VL checkpoint's original position encoding, a follow-up
  patch to emit and pad 3D position_ids when `multi_modal_inputs` is
  present is recommended.

### Client-only guarantees

- Image payloads are encoded as data URLs and forwarded to the
  vLLM-served VL endpoint via OpenAI `image_url` content blocks; they
  never leave the client host.
