# OpenClaw Reward Cheatsheet

## Run the test

```bash
cd agentjet/tutorial/opencode_build_openclaw_agent

# pointwise (default)
DASHSCOPE_API_KEY=your_key python test_reward.py

# listwise
REWARD_MODE=listwise DASHSCOPE_API_KEY=your_key python test_reward.py
```

## Run the training endpoint

```bash
# pointwise (default)
AJET_SWARM_URL=http://localhost:10086 \
DASHSCOPE_API_KEY=your_key \
REWARD_MODE=pointwise \
python fake_vllm_endpoint.py

# listwise
AJET_SWARM_URL=http://localhost:10086 \
DASHSCOPE_API_KEY=your_key \
REWARD_MODE=listwise \
python fake_vllm_endpoint.py
```

## Reward modes

| Mode | Description |
|------|-------------|
| `pointwise` | Each response scored independently (0.0–1.0) |
| `listwise` | All responses ranked together (best=1.0, worst=0.0) |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REWARD_MODE` | `pointwise` | `pointwise` or `listwise` |
| `DASHSCOPE_API_KEY` | — | DashScope API key (required) |
| `JUDGE_MODEL` | `qwen-plus` | Judge model name |
| `JUDGE_BASE_URL` | DashScope endpoint | Judge model base URL |
| `AJET_SWARM_URL` | `http://localhost:10086` | Swarm server URL |
| `NUM_REPEAT` | `4` | GRPO N (responses per query) |
