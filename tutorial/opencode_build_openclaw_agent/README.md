# OpenClaw Agent Training - Extraversion Personality

Train an LLM agent to exhibit more extraverted personality traits using reinforcement learning.

## Overview

This training program uses GRPO (Group Relative Policy Optimization) to train Qwen2.5-7B-Instruct to respond with more extraverted characteristics:
- Outgoing, energetic, enthusiastic tone
- Social engagement and excitement
- Positive, upbeat language
- Action-oriented expressions

## Architecture

```
User Query → fake_vllm_endpoint.py → Swarm Server (8 GPUs)
                ↓
        Generate N=4 responses in parallel
                ↓
        Evaluate with ExtraversionGrader (OpenJudge)
                ↓
        Compute rewards & update model (GRPO)
                ↓
        Return best response to user
```

## Prerequisites

```bash
pip install py-openjudge datasets
```

## Setup

### 1. Download Dataset

```bash
cd tutorial/opencode_build_openclaw_agent
python download_dataset.py
```

This downloads the `holistic-ai/personality_manipulation` dataset and extracts extraversion examples.

### 2. Configure API Key

Edit `on_compute_relative_reward.py` and set your API key for the judge model:

```python
model = OpenAIChatModel(
    model="qwen-plus",
    api_key="YOUR_API_KEY_HERE",  # Change this
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

## Training

### Step 1: Start Swarm Server

On your GPU server (with 8 GPUs available):

```bash
ajet-swarm start
```

Or with monitoring:

```bash
(ajet-swarm start &> ajet-swarm-server.log) & (ajet-swarm overwatch)
```

### Step 2: Start Fake vLLM Endpoint

In a new terminal:

```bash
cd tutorial/opencode_build_openclaw_agent
export AJET_SWARM_URL="http://localhost:10086"
export NUM_REPEAT=4
python fake_vllm_endpoint.py
```

This starts the training proxy on `http://localhost:8090`.

### Step 3: Configure OpenClaw to Use Training Endpoint

OpenClaw needs to connect to the fake vLLM endpoint.

Configure it to use `http://localhost:8090` as the LLM backend.

### Step 4: Send Training Requests

Option A - Manual testing via OpenClaw Web / Cli:

```bash
openclaw agent --message "What are your thoughts on Paris?" --thinking high
```

Option B - Automated dataset iteration:

```bash
python mock_user_request.py
```

This will iterate through the personality_manipulation dataset and send each question via OpenClaw CLI.

## Configuration

Key parameters in `fake_vllm_endpoint.py`:

- `n_gpu=8` - Number of GPUs for training
- `batch_size=32` - Training batch size
- `num_repeat=4` - GRPO N parameter (responses per query)
- `model` - Base model path

## Reward Function

The `ExtraversionGrader` evaluates responses on a 1-10 scale:
- 1 = Highly introverted (reserved, quiet)
- 10 = Highly extraverted (energetic, enthusiastic)

Scores are normalized to [-1, 1] for GRPO training.

## Monitoring

Check training progress:

```bash
# View swarm status
ajet-swarm overwatch

# Check request history
curl http://localhost:8090/requests

# Health check
curl http://localhost:8090/health
```

## Files

- `fake_vllm_endpoint.py` - Main training server
- `on_compute_relative_reward.py` - Extraversion reward function
- `on_user_submit_new_requests.py` - Request handler
- `download_dataset.py` - Dataset downloader
- `mock_user_request.py` - Automated testing client

## Troubleshooting

**Import errors**: LSP warnings about unresolved imports are normal - dependencies will be available at runtime.

**Connection refused**: Ensure swarm server is running on port 10086.

**All episodes failed**: Check GPU availability and swarm server logs.

## Notes

- Training is passive - the endpoint waits for requests rather than iterating a dataset
- Each request generates N=4 responses, evaluates them, and trains on the best
- The model gradually learns to produce more extraverted responses over time
