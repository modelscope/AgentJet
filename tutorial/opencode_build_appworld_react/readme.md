# AppWorld React Agent Training

A React-style agent for learning to interact with AppWorld environment using GRPO reinforcement learning.

## Configuration

Edit `agent_roll.py` to set:
- `REMOTE_TRAIN_MODEL`: Model path (default: Qwen2.5-7B-Instruct)
- `REMOTE_BATCH_SIZE`: 32
- `REMOTE_ALLOCATE_GPU_PER_NODE`: 8
- `LOCAL_GRPO_N`: 4

## Training

1. Start swarm server on GPU machine:
```bash
ajet-swarm start
```

2. Start environment service:
```bash
# Make sure AppWorld is installed at /tmp/pack_all_in_one
cd /tmp/pack_all_in_one
bash EnvService/env_sandbox/appworld.sh
```

3. Run training client:
```bash
python -m tutorial.opencode_build_appworld_react.agent_roll
```

## Monitor

```bash
ajet-swarm overwatch
```
