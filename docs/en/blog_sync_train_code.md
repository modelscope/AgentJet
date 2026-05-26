# From Laptop to GPU Cluster: Remote-Controlled Swarm Training with `sync_train_code`

> `sync_train_code` lets an AgentJet Swarm client upload the local AgentJet source snapshot to a remote Swarm Server before training starts. In practice, this means a laptop, or an AI research agent running on that laptop, can modify training code, sync it to a GPU cluster, restart the engine, and continue the experiment loop without manual deployment.

AgentJet Swarm already made rollout distributed: the training engine can run on a GPU cluster, while clients on laptops or workstations run agent workflows and send rewards back.

`sync_train_code` extends that idea from rollout control to **training-code control**. The client can now decide not only what samples to send, but also which AgentJet source code the remote training engine should run.

## Why This Matters

In agent RL research, code iteration is part of the experiment:

- reward processing changes;
- trajectory recording changes;
- config conversion changes;
- launcher behavior changes;
- backend defaults change;
- bugs appear only after the remote training engine boots.

Without code sync, every server-side change becomes an operations task: SSH into the cluster, patch files, rebuild an image, or manually keep the laptop checkout and server checkout aligned.

With `sync_train_code`, the workflow becomes:

```text
edit locally -> sync source snapshot -> restart remote engine -> observe result -> iterate
```

This is especially important for automated research. A research agent can now run the same loop without human deployment steps: inspect results, modify AgentJet code, call `sync_train_code_from_dir()`, restart the Swarm engine, wait for metrics, and decide the next experiment. The GPU cluster becomes a controlled execution backend rather than a machine that must be manually operated.

## Minimal Usage

Start the Swarm Server on the GPU cluster:

```bash
ajet-swarm start --swarm-port=10086
```

Then run the client from your local AgentJet checkout:

```python
import os
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient

swarm_worker = SwarmClient("http://gpu-cluster-host:10086")

if os.getenv("SYNC_CODE", "0") == "1":
    swarm_worker.sync_train_code_from_dir(os.getcwd(), force_restart=True)

swarm_worker.auto_sync_train_config_and_start_engine(
    AgentJetJob(
        experiment_name="remote_controlled_grpo",
        algorithm="grpo",
        n_gpu=8,
        model="/mnt/models/Qwen2.5-7B-Instruct",
        batch_size=32,
        num_repeat=4,
    ),
    force_restart=True,
)
```

Run with code sync enabled:

```bash
SYNC_CODE=1 python your_swarm_client.py
```

For local development, it can also be combined with automatic local server startup:

```python
swarm_worker = SwarmClient(
    "http://localhost:10086",
    auto_start_swarm_server=True,
)
swarm_worker.sync_train_code_from_dir(os.getcwd(), force_restart=True)
```

For a remote GPU cluster, start `ajet-swarm start` on the server first, then connect to it from the client.

## What Gets Controlled

`sync_train_code` controls the AgentJet source code used by the training engine process. It works together with existing Swarm Client APIs:

| Operation | Client API | Remote effect |
|---|---|---|
| Sync code | `sync_train_code_from_dir()` | Uploads a timestamped `ajet/` source snapshot. |
| Sync config | `auto_sync_train_config_and_start_engine()` | Sends model path, algorithm, GPU count, batch size, and other training parameters. |
| Start engine | `start_engine()` | Starts training with the synced code and config. |
| Stop engine | `stop_engine()` | Stops the active engine and returns to `ENGINE.OFFLINE`. |
| Run rollout | `begin_episode()` / `end_episode()` | Sends samples and rewards to the remote trainer. |

Together, these APIs make the Swarm Client a practical control plane for remote training.

## How It Works

The implementation is intentionally simple and safe:

```text
local checkout
  -> git ls-files -- ajet
  -> create /tmp/ajet_train_code_*.zip
  -> POST /sync_train_code
  -> server extracts to ./ajet_temp/<timestamp>/ajet
  -> start_engine sets ISOLATED_AGENTJET_BASE_DIR
  -> training subprocess imports ajet from the synced snapshot first
```

The server checkout is not overwritten. Each sync creates an isolated timestamped source copy. When training starts successfully, the server logs a line like:

```text
[start_engine] Using synced training code from ./ajet_temp/20260526_120000_123456/ajet
```

## Important Rules

Only Git-tracked files under `ajet/` are packaged.

- Modified tracked files are included.
- New untracked files are not included until staged or tracked.
- Files outside `ajet/` are not included.
- Datasets, checkpoints, virtual environments, and ignored files are not included.

If you add a new module, stage it before syncing:

```bash
git add ajet/path/to/new_module.py
```

Code sync is accepted only when the server is `ENGINE.OFFLINE`. Use `force_restart=True` during development when you intentionally want to stop the current engine and restart with new code.

`sync_train_code` does not sync Python dependencies. If the new code imports a new package, install that package on the GPU server environment.

## Why It Helps Automated Research

Automated research systems need to close the loop:

```text
plan -> modify code/config -> launch training -> wait -> analyze -> adjust -> relaunch
```

The weak point is usually deployment. If every code change requires a human to log into the GPU server, the loop is not truly automated.

With AgentJet Swarm plus `sync_train_code`, an automated researcher can operate at the experiment level:

- generate a new hypothesis;
- patch AgentJet training logic or config mapping;
- sync the patched code to the cluster;
- restart training;
- monitor results;
- decide the next patch or experiment.

This turns the GPU cluster into a programmable research instrument. The laptop or automation agent owns the research loop; the cluster provides reliable compute.

That is the key value of `sync_train_code`: **remote Swarm training is no longer just distributed rollout. It becomes remotely programmable training.**
