---
name: swarm-configuration
description: How `max_env_worker` caps the "Running Episodes" gauge, and how `AgentJetJob` relates to the YAML config.
license: Complete terms in LICENSE.txt
---

## Running-episodes cap

The `Running Episodes (Episodes: N)` number in `swarm_overwatch` is bounded by the **engine-side** `max_env_worker` (set on the job config, e.g. `CocktailV2Config.max_env_worker`, then forwarded into `AgentJetJob` and read at `ajet/backbone/trainer_verl.py` as `max_parallel`). In `ajet/task_rollout/native_parallel_worker.py::rollout_swarm`, the engine spawns `ceil(max_env_worker / grpo_n) * grpo_n` long-lived worker threads, each looping `register_episode` → wait-for-claim → repeat, so the total in-flight episodes (summed across **all** swarm clients) cannot exceed that count. `total_batch_size`, per-client `max_env_worker`, `grpo_n`, and the number of clients do **not** raise this cap , to lift it, raise the engine's `max_env_worker` (keep it divisible by `grpo_n`) and restart.

## AgentJetJob ↔ YAML

When using Agentjet Swarm, please first use `AgentJetJob` as the primary configuration interface.

If there are fields you want to set that are not exposed as `AgentJetJob` kwargs, use yaml as the primary configuration interface.

In general, you should place most configuration in a place (either `AgentJetJob` or yaml), and MUST NOT place configuration here and there at the same time.

`AgentJetJob` (`ajet/copilot/job.py`) is a thin **YAML overlay**, not a separate config system. On `__init__` it loads a base YAML (default `ajet/default_config/ajet_swarm_default.yaml`, or whatever path is passed via `base_yaml_config=`) into `self.config`, then walks an `overrides` table that maps each constructor kwarg to a deep YAML key (e.g. `max_env_worker` → `ajet.rollout.max_env_worker`, `batch_size` → `ajet.data.train_batch_size`, `model` → `ajet.model.path`). For each entry: if the kwarg is `None` the YAML value wins; if non-`None` it overwrites the YAML value in-place. Anything not listed in `overrides` (e.g. `rollout.temperature`, `rollout.multi_turn`, `trainer_common.save_freq`) has no kwarg shortcut and must be set by mutating `ajet_job.config.ajet.*` directly after construction , this is what `build_cocktail_ajet_job` does in the cocktail_rl_v2 tutorial. `dump_job_as_yaml(path)` serialises the merged result back out, and that dumped YAML is the file the engine subprocess actually consumes. Net effect: **YAML is the source of truth for defaults; `AgentJetJob` kwargs are sparse overrides; post-construction attribute writes are the escape hatch for fields without a kwarg.**
