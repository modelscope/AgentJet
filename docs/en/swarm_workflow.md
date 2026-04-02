# Swarm Workflow

This tutorial introduces how to define a trainable workflow in **swarm mode** — where a GPU server runs the training loop and one or more client machines (potentially without GPUs) run the agent rollout. For single-machine training, see [Classic Workflow](../classic_workflow/).

## Classic vs. Swarm Workflow

```
 Classic Mode                              Swarm Mode
 ────────────                              ──────────
 ┌──────────────────────┐                  ┌──────────────────────┐
 │    Single Machine    │                  │    GPU Server        │
 │  ┌────────────────┐  │                  │  ┌────────────────┐  │
 │  │ Training Loop  │  │                  │  │ Training Loop  │  │
 │  │ (weight update)│  │                  │  │ (weight update)│  │
 │  └───────┬────────┘  │                  │  └───────┬────────┘  │
 │          │           │                  │          │           │
 │  ┌───────▼────────┐  │                  │  ┌───────▼────────┐  │
 │  │ Rollout        │  │                  │  │vLLM Engine     │  │
 │  │ (vLLM + agent  │  │                  │  │(inference only)│  │
 │  │  + environment)│  │                  │  └───────┬────────┘  │
 │  └────────────────┘  │                  │          │OpenAI API │
 └──────────────────────┘                  └──────────┼───────────┘
                                                      │
                                           ┌──────────▼────────────┐
                                           │   Client Machine(s)   │
                                           │   (no GPU needed)     │
                                           │  ┌────────────────┐   │
                                           │  │ Agent Rollout  │   │
                                           │  │ + Environment  │   │
                                           │  │ + Reward Calc  │   │
                                           │  └────────────────┘   │
                                           └───────────────────────┘
```

| | Classic Workflow | Swarm Workflow |
|---|---|---|
| **Launch** | `ajet --conf your.yaml` | `ajet-swarm start` + `python your_client.py` |
| **Workflow class** | Inherits `ajet.Workflow`, uses `tuner.as_agentscope_model()` | Plain Python function, uses `SwarmClient` + OpenAI SDK |
| **Where agent runs** | On the GPU machine inside the training loop | On any client machine, communicating via HTTP |
| **GPU requirement** | All on one cluster | Server needs GPUs; clients do not |
| **Scaling** | Limited to one machine's parallelism | Add more client machines to scale rollout |
| **Config** | YAML config file | `AgentJetJob` Python object (or YAML) |

## How Swarm Workflow Works

<div align="center">
<img width="600" alt="image" src="https://serve.gptacademic.cn/publish/shared/Image/swarming.gif"/>
</div>



A swarm workflow has three phases per training step:

```
 Client                             Server (ajet-swarm)
 ──────                             ───────────────────
 1. begin_episode()        ───►     Registers episode, returns
                           ◄───     base_url + api_key

 2. Run your agent logic            Agent calls LLM via
    (any framework/language) ──►    OpenAI-compatible API
    Compute reward locally  ◄───    (proxied to vLLM engine)

 3. end_episode(task,       ───►    Collects reward + trajectory
    episode_uuid,                   When enough episodes collected,
    workflow_output)                 triggers weight update
```

**Key idea**: the client calls `begin_episode()` to get OpenAI-compatible credentials (`base_url` and `api_key`), uses them to call the model during agent execution, computes the reward locally, and sends it back with `end_episode()`. The server handles all training — the client never touches model weights.


## Step-by-Step Guide

### 1. Start the Swarm Server

```bash
# Start server + TUI monitor (in two terminals, or combined):
ajet-swarm start &
ajet-swarm overwatch
```

The server starts with `ajet_swarm_default.yaml` (see [Swarm Mode Config Chain](../configuration/#swarm-mode-config-chain)). No `--conf` is needed for the default setup.


### 2. Configure and Activate the Server

From your client script, connect to the server and send training parameters:

```python
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient, run_episodes_until_all_complete

swarm_worker = SwarmClient("http://localhost:10086")

yaml_job = AgentJetJob(
    algorithm="grpo",
    project_name="my-project",
    experiment_name="my-experiment",
    n_gpu=8,                                          # GPUs on the server
    model="/path/to/Qwen2.5-7B-Instruct",            # model to train
    batch_size=32,                                     # tasks per training step
    num_repeat=4,                                      # GRPO group size
)

# Send config to server and start the vLLM engine
swarm_worker.auto_sync_train_config_and_start_engine(yaml_job)
```

!!! tip "Inspecting the full config"
    Call `yaml_job.dump_job_as_yaml('./config.yaml')` to see all resolved configuration keys, or `yaml_job.build_job_from_yaml('./config.yaml')` to load overrides from a YAML file.


### 3. Write Your Agent and Reward

Write a function that takes a task and OpenAI-compatible credentials, runs the agent, and returns a `WorkflowOutput` with the reward:

```python
import requests
from ajet.schema.task import Task, WorkflowOutput

def execute_agent(task: Task, api_baseurl_key) -> WorkflowOutput:
    base_url, api_key = api_baseurl_key.base_url, api_baseurl_key.api_key

    # Read task data
    query = task.main_query
    reference_answer = task.metadata["answer"]

    # Call the model via OpenAI-compatible API
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    response = requests.post(
        f"{base_url}/chat/completions",
        json={"model": "any", "messages": messages},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    answer = response.json()["choices"][0]["message"]["content"]

    # Compute reward locally
    is_success = check_answer(answer, reference_answer)
    reward = 1.0 if is_success else 0.0

    return WorkflowOutput(reward=reward, is_success=is_success, metadata={"answer": answer})
```

!!! info "Use any framework"
    The model is exposed as a standard OpenAI-compatible endpoint. You can use the OpenAI Python SDK, LangChain, AgentScope, raw HTTP, or any language/framework that speaks OpenAI API. See [With Frameworks](../support_oaisdk/) for examples.


### 4. Run the Training Loop

Wire the agent into a rollout loop — the server handles the rest:

```python
def rollout(task):
    # Get credentials for this episode
    episode_uuid, api_baseurl_key = swarm_worker.begin_episode()
    # Run agent
    workflow_output = execute_agent(task, api_baseurl_key)
    # Report reward back to server
    swarm_worker.end_episode(task, episode_uuid, workflow_output)
    return workflow_output.reward

# Training loop
next_batch = []
for task in dataset.generate_training_tasks():
    for _ in range(LOCAL_GRPO_N):
        next_batch.append(task)
        if len(next_batch) >= BATCH_SIZE * LOCAL_GRPO_N:
            results = run_episodes_until_all_complete(next_batch, func=rollout, auto_retry=True)
            print(results)
            next_batch.clear()
```

That's it — the server collects episodes, computes advantages, and updates model weights automatically. Your client just runs agents in a loop.


### 5. Episode Lifecycle

Each episode follows this lifecycle:

| Step | Client Call | What Happens |
|---|---|---|
| **Begin** | `swarm_worker.begin_episode()` | Server registers episode, returns `(episode_uuid, api_baseurl_key)` |
| **Execute** | Your agent code | Agent calls model via `base_url`/`api_key`; server tracks the conversation |
| **End** | `swarm_worker.end_episode(task, uuid, output)` | Server receives reward and trajectory |
| **Abort** | `swarm_worker.abort_episode(uuid)` | Discard this episode (bad reward, API error, debugging, etc.) |

!!! warning "Thread safety"
    `SwarmClient` is thread-safe and stateless per episode. You can safely call `begin_episode()` from multiple threads concurrently.


## Advanced: Multi-Model Training

Swarm mode supports training multiple models simultaneously by running multiple swarm servers. Each server trains one model, and the client orchestrates them:

```python
# Two servers, two models
swarm_7b = SwarmClient("http://server-1:10086")   # trains 7B model
swarm_14b = SwarmClient("http://server-2:10086")  # trains 14B model

# Configure each server with its own AgentJetJob
swarm_7b.auto_sync_train_config_and_start_engine(job_7b)
swarm_14b.auto_sync_train_config_and_start_engine(job_14b)

def rollout(task):
    # Get credentials from both servers
    uuid_7b, api_7b = swarm_7b.begin_episode()
    uuid_14b, api_14b = swarm_14b.begin_episode()

    # Run multi-model agent (agent 1 uses 7B, agent 2 uses 14B, ...)
    output_7b, output_14b = execute_multi_model_agent(task, api_7b, api_14b)

    # Report rewards to each server separately
    swarm_7b.end_episode(task, uuid_7b, output_7b)
    swarm_14b.end_episode(task, uuid_14b, output_14b)
```

See the [Multi-Model Training](../example_train_multi_model/) example for a complete walkthrough.


## Advanced: Distributed Rollout

Scale rollout across many machines by running the same client script on multiple nodes. Each node handles a share of the batch:

```python
N_CLIENTS = 4  # total number of client machines
next_batch = []
for task in dataset.generate_training_tasks():
    for _ in range(LOCAL_GRPO_N):
        next_batch.append(task)
        # Each client processes 1/N_CLIENTS of the batch
        if len(next_batch) >= (BATCH_SIZE // N_CLIENTS * LOCAL_GRPO_N):
            results = run_episodes_until_all_complete(next_batch, func=rollout, auto_retry=True)
            next_batch.clear()
```

```bash
# On each client machine (no GPU needed):
python swarm_client_roll.py
```

The swarm server is resilient — if a client crashes, other clients continue. The server waits for enough episodes before triggering a training step, regardless of which clients provided them.


## Next Steps

<div class="card-grid">
<a href="../classic_workflow/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/lucide:workflow.svg" class="card-icon card-icon-general" alt=""><h3>Classic Workflow</h3></div><p class="card-desc">Single-machine training with the ajet CLI.</p></a>
<a href="../swarm_best_practice/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/lucide:rocket.svg" class="card-icon card-icon-general" alt=""><h3>Swarm Best Practice</h3></div><p class="card-desc">Four demo scenarios: single, multi-model, distributed, and multi-mission.</p></a>
<a href="../configuration/#swarm-mode" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/lucide:settings.svg" class="card-icon card-icon-general" alt=""><h3>Swarm Configuration</h3></div><p class="card-desc">All swarm-related configuration keys explained.</p></a>
</div>
