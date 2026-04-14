# AIME Math Agent - Swarm Training

This tutorial demonstrates how to train a math problem-solving agent using AgentJet Swarm with the DAPO-Math-17k dataset and AIME-2024 test set.

## Overview

- **Training Dataset**: [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) (~17k math problems)
- **Test Dataset**: [AIME-2024](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024) (American Invitational Mathematics Examination)
- **Reward Function**: DAPO-style binary reward (+1/-1) with boxed answer extraction
- **Algorithm**: GRPO (Group Relative Policy Optimization)

## Quick Start

### Step 1: Download Datasets

```bash
# Use proxychains to download from HuggingFace
proxychains python -m tutorial.opencode_build_aime.download_data
```

This will download:
- `tutorial/opencode_build_aime/data/dapo-math-17k.parquet` (training)
- `tutorial/opencode_build_aime/data/aime-2024.parquet` (test)

### Step 2: Start Swarm Server

On your GPU server, start the swarm server:

```bash
# Start swarm server (default port: 10086)
ajet-swarm start

# Or with logging and monitoring:
(ajet-swarm start &> ajet-swarm-server.log) & (ajet-swarm overwatch)
```

### Step 3: Run Training

```bash
# Set environment variables (optional)
export AJET_SWARM_URL="http://localhost:10086"
export REMOTE_MODEL_PATH="/path/to/your/model"
export REMOTE_BATCH_SIZE=32
export REMOTE_ALLOCATE_GPU_PER_NODE=8

# Start training
python -m tutorial.opencode_build_aime.agent_roll
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AJET_SWARM_URL` | `http://localhost:10086` | Swarm server URL |
| `REMOTE_MODEL_PATH` | Qwen2.5-7B-Instruct | Model to train |
| `REMOTE_BATCH_SIZE` | 32 | Training batch size |
| `REMOTE_ALLOCATE_GPU_PER_NODE` | 8 | GPUs per node |

### Training Parameters

Edit `agent_roll.py` to modify:
- `LOCAL_GRPO_N`: Number of rollouts per task (default: 4)
- `LOCAL_NUM_EPOCH`: Number of training epochs (default: 10000)

## Reward Function

The reward function follows the DAPO (Direct Alignment from Preferences Optimization) style:

1. **Answer Extraction**:
   - Primary: Extract from `\boxed{...}` (last 300 characters)
   - Fallback: Minerva-style `Answer: ...` pattern

2. **Normalization**:
   - Remove units (cm, degrees, dollars, etc.)
   - Normalize LaTeX expressions
   - Handle comma-separated numbers

3. **Scoring**:
   - Correct answer: +1.0
   - Incorrect answer: -1.0

## Files

```
tutorial/opencode_build_aime/
├── agent_roll.py      # Training entry point
├── agent_run.py       # Agent execution & reward computation
├── download_data.py   # Dataset download script
├── readme.md          # This file
└── data/              # Downloaded datasets (created by download_data.py)
    ├── dapo-math-17k.parquet
    └── aime-2024.parquet
```

## Monitoring

Monitor training progress:

```bash
# In a separate terminal
ajet-swarm overwatch --swarm-url=http://localhost:10086
```

Or use the Python API:

```python
from ajet.tuner_lib.experimental.swarm_client import SwarmClient

client = SwarmClient("http://localhost:10086")
client.print_rollout_stat()
```

## Debugging

To debug without affecting training:

```python
# In agent_roll.py, change:
swarm_worker.end_episode(task, episode_uuid, workflow_output)

# To:
swarm_worker.abort_episode(episode_uuid)
```

This discards the episode without contributing to training.

## References

- [DAPO Paper](https://arxiv.org/abs/2503.14476)
- [verl Repository](https://github.com/volcengine/verl)
- [AgentJet Swarm Documentation](../../docs/en/swarm.md)
