# Vibe Training with AgentJet Swarm: Training Werewolves Agents by Just Prompting

> The original document is [Chinese Version](https://modelscope.github.io/AgentJet/en/blog_vibe_training_werewolves.zh/)

## Abstract

AgentJet's Swarm Mode is a game-changer — want to train something? Just prompt the AI. I call it **Vibe Training**. In this blog, we demonstrate how to set up a full Werewolves game training pipeline using AgentJet Swarm, where the entire training code is only **70 lines**. We configure a heterogeneous multi-agent system with trainable werewolves and a seer sharing a 7B model, while the hunter, witch, and villagers are powered by a static Qwen3-235B-A22B. The result? Smooth training curves and rising win rates — even after two CPFS disk crashes mid-training.

## Game Setup: Heterogeneous Multi-Agent Configuration

The Werewolves social deduction game is a classic POMDP (Partially Observable Markov Decision Process) problem. In this setup, we configure a 9-player game with **two tiers of intelligence**:

| Role | Count | Model | Mode |
|------|-------|-------|------|
| Werewolf | 3 | Qwen2.5-7B | **Trainable** (shared parameters) |
| Seer | 1 | Qwen2.5-7B | **Trainable** (shared parameters) |
| Hunter | 1 | Qwen3-235B-A22B | Static |
| Witch | 1 | Qwen3-235B-A22B | Static |
| Villager | 3 | Qwen3-235B-A22B | Static |

This is a powerful configuration: the trainable agents (werewolves + seer) share a lightweight 7B model, learning to play against formidable opponents powered by Qwen3-235B-A22B. The shared-parameter design means the 7B model must learn **role-specific behaviors** from the same weights — werewolves must learn deception and coordination, while the seer must learn to investigate and communicate findings effectively.

## The 70-Line Training Code

Here's the remarkable part: the entire training orchestration under Swarm Mode fits in roughly 70 lines of Python. No boilerplate, no framework gymnastics — just clean, readable logic:

```python
import os
from ajet.schema.task import Task, WorkflowTask
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.default_config.ajet_config_schema import AjetTaskReader
from ajet.tuner_lib.experimental.swarm_client import SwarmClient

NUM_EPOCH = 10000
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")

def main():
    dataset = RouterTaskReader(
        reader_type="random_dummy",
        reader_config=AjetTaskReader()
    )

    ajet_job = AgentJetJob(
        base_yaml_config="tutorial/example_werewolves_swarm/werewolves.yaml",
        algorithm="grpo",
        experiment_name="werewolves_swarm",
        max_env_worker=128,
    )

    # Handshake with remote swarm server
    swarm_worker = SwarmClient(AJET_SWARM_URL)
    swarm_worker.auto_sync_train_config_and_start_engine(ajet_job)

    GRPO_N = ajet_job.num_repeat
    REMOTE_BATCH_SIZE = ajet_job.batch_size

    def rollout(task):
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(
            discard_episode_timeout=240
        )
        workflow_output = execute_agent(task, api_baseurl_key)
        swarm_worker.end_episode(task, episode_uuid, workflow_output)

    executor = PeriodicDrainThreadPoolExecutor(
        workers=GRPO_N * REMOTE_BATCH_SIZE, max_parallel=64, auto_retry=True
    )
    for _ in range(NUM_EPOCH):
        for _, task in enumerate(dataset.generate_training_tasks()):
            for _ in range(GRPO_N):
                executor.submit_with_periodic_drain(fn=rollout, task=task)


def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
    import asyncio
    from tutorial.example_werewolves.start import ExampleWerewolves
    game = ExampleWerewolves(
        trainable_targets=["werewolf"],
        big_external_opponent_llm_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
        big_external_opponent_llm_url="http://22.14.116.243:2888/v1",
    )
    res = asyncio.run(game.execute(WorkflowTask(task=task), api_baseurl_key))
    return res
```

[Full source on GitHub](https://github.com/modelscope/AgentJet/blob/main/tutorial/example_werewolves_swarm/agent_roll.py)

The code follows a simple three-step pattern that defines all Swarm Client programs:

1. **Handshake**: `SwarmClient` connects to the Swarm Server, syncs the training configuration (model path, algorithm, hyperparameters), and starts the training engine.
2. **Rollout Loop**: For each task, the client calls `begin_episode()` to get a temporary API endpoint backed by the training model, runs the full game simulation, then calls `end_episode()` to report the reward back.
3. **Parallel Execution**: `PeriodicDrainThreadPoolExecutor` manages concurrent game simulations, with up to 64 parallel episodes feeding the training sample pool.

## Architecture: How It Works

```
Swarm Server (GPU Cluster, 8x GPUs)
    ├── Qwen2.5-7B-Instruct (trainable)
    ├── vLLM inference engine
    ├── Policy gradient computation (GRPO)
    └── Checkpoint management
         │
         │  ◄── OpenAI-compatible API ──►
         │
Swarm Client (any device)
    ├── Game simulation engine
    ├── 9-player Werewolves game
    │    ├── 4 trainable agents → Swarm Server API (7B)
    │    └── 5 static agents → External Qwen3-235B API
    ├── Reward computation (win/loss)
    └── Episode submission to Swarm Server
```

The key insight of the Swarm architecture is **complete decoupling of training and sampling**. The Swarm Server handles all GPU-intensive work (inference + gradient updates), while the Swarm Client can run anywhere — even on your laptop. The client simply plays games using the API endpoints and reports rewards. When the sample pool reaches the configured batch size, the server automatically triggers a weight update.

## Training Configuration

The training is configured via `werewolves.yaml`:

```yaml
ajet:
  model:
    path: /path/to/Qwen2.5-7B-Instruct
  rollout:
    temperature: 0.7
    num_repeat: 6           # GRPO rollouts per task
    max_env_worker: 64
    max_response_length_in_one_turn: 1024
    max_model_len: 22000
  data:
    train_batch_size: 32
    max_prompt_length: 4000
    max_response_length: 18000
  trainer_common:
    save_freq: 5            # Checkpoint every 5 steps
    total_training_steps: 25
    n_gpus_per_node: 8
  enable_swarm_mode: True
```

Key design decisions:
- **`save_freq: 5`** — Frequent checkpointing proved critical when our CPFS disk crashed twice during training. With a small save interval, we lost minimal progress each time.
- **`num_repeat: 6`** — GRPO generates 6 candidate responses per task for relative reward comparison.
- **`train_batch_size: 32`** — The server waits until 32 completed tasks accumulate before triggering a gradient update.

## Training Curve

During training, the CPFS shared disk was filled up by other colleagues — twice — causing two crashes. Fortunately, the small checkpoint save interval (`save_freq: 5`) meant we could recover quickly each time without losing significant progress.

Despite these interruptions, the reward curve shows a clear upward trend, demonstrating that the agents are successfully learning:

- **Werewolves** learn to coordinate kills, maintain cover during discussions, and avoid self-exposure when voted out
- **The Seer** learns when to reveal investigation results and how to build trust with villagers

## Getting Started

### Step 1: Start the Swarm Server

```bash
# Using Docker (recommended)
docker run --rm -it \
  -v ./swarmlog:/workspace/log \
  -v ./swarmexp:/workspace/saved_experiments \
  -p 10086:10086 --gpus=all --shm-size=32GB \
  ghcr.io/modelscope/agentjet:main bash -c \
  "(ajet-swarm overwatch) & (NO_COLOR=1 LOGURU_COLORIZE=NO ajet-swarm start &>/workspace/log/swarm_server.log)"
```

### Step 2: Run the Training Client

```bash
git clone https://github.com/modelscope/agentjet.git && cd agentjet
pip install -e .
python tutorial/example_werewolves_swarm/agent_roll.py
```

### Step 3: Monitor Training

```bash
ajet-swarm overwatch
```

## About AgentJet

[AgentJet](https://modelscope.github.io/AgentJet/) features an original **Swarm Training Mode** that lets you comfortably do RL in any environment.

AgentJet is built on top of [VERL](https://github.com/volcengine/verl). For researchers familiar with VERL, virtually all algorithms implemented in VERL can be seamlessly applied to AgentJet. On top of that, AgentJet adds a swarm communication layer and timeline merging optimizations, while keeping the core training logic fully consistent. Low migration cost, reliable performance.

For a deeper dive into the Swarm architecture, see the [Swarm Training Mode Introduction](https://modelscope.github.io/AgentJet/en/swarm_intro_blog_en/).
