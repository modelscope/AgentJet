# -*- coding: utf-8 -*-
"""
Single source of truth for example_cocktail_rl_v2.

Every config value used anywhere in this tutorial -- v2 schedule knobs, engine
knobs, per-domain knobs -- lives on `CocktailV2Config`. There are no YAMLs, no
hardcoded constants in the runner or clients, no `.get(key, default)` fallback
patterns that could drift. To change anything, edit a default here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List


SCHEDULE_TYPES = ("linear", "cos", "constant")


# ============================ Per-domain sub-configs ============================

@dataclass
class AppWorldConfig:
    env_url: str = "http://127.0.0.1:8080"
    env_type: str = "appworld"
    training_split: str = "train"
    validation_split: str = "dev"
    episode_timeout: int = 60


@dataclass
class AimeConfig:
    episode_timeout: int = 60
    # Filenames resolve under ../opencode_build_aime/data relative to this tutorial.
    train_dataset_filename: str = "dapo-math-17k.parquet"
    test_dataset_filenames: dict = field(default_factory=lambda: {
        "AIME-2026": "aime-2026.parquet",
        "DAPO-Math-Tiny-Val": "dapo-math-tiny-val.parquet",
    })


# ============================ Top-level config ============================

@dataclass
class CocktailV2Config:
    """Single source of truth. Both client_0 and client_1 must agree on these
    values, so the dataclass defaults ARE the canonical config.

    Schedule semantics for client_0's batch ratio:
      schedule_type == "constant": ratio is always `schedule_start`.
      schedule_type == "linear":   linear from `schedule_start` at step 0 to
                                   `schedule_end` at `schedule_end_step`,
                                   then stays at `schedule_end`.
      schedule_type == "cos":      cosine anneal from `schedule_start` to
                                   `schedule_end` over `schedule_end_step`,
                                   then stays at `schedule_end`.
    client_1's ratio is always 1 - client_0's ratio.
    """
    # ---- v2 batching / schedule ----
    total_batch_size: int = 32
    grpo_n: int = 8
    schedule_type: str = "linear"
    schedule_start: float = 0.5
    schedule_end: float = 0.0
    schedule_end_step: int = 200

    # ---- v2 client-side runtime ----
    max_env_worker: int = 64 * 8
    max_inference_tracker_threads: int = 256
    eval_interval: int = 10
    eval_k: int = 4
    total_training_steps: int = 200
    swarm_url: str = "http://localhost:10086"
    result_dir: str = "./cocktail_results_v2"

    # ---- engine-global per-rollout knobs (read by engine + per-client agents) ----
    max_response_length: int = 20000
    max_steps: int = 25

    # ---- engine-only knobs (consumed by build_cocktail_ajet_job) ----
    project_name: str = "cocktail_rl"
    experiment_name: str = "cocktail_rl_v2"
    experiment_dir: str = "auto"
    model_path: str = "/mnt/data_cpfs/xielipeng.xlp/models/Qwen3-8B-Keep-History"
    algorithm: str = "grpo"
    swarm_mode: bool = True
    swarm_mode_sample_collection_method: str = "rollout_until_all_clients_agree_sync_weight"
    logging: str = "swanlab"
    compute_madness_checklist: List[str] = field(default_factory=lambda: ["nonsense"])
    max_prompt_length: int = 3000
    max_response_length_in_one_turn: int = 12000
    max_model_len: int = 23000
    max_num_seqs: int = 128
    n_gpu: int = 8
    use_kl_loss: bool = True
    use_kl_in_reward: bool = False
    kl_penalty_type: str = "kl"

    # ---- engine knobs not exposed as AgentJetJob kwargs ----
    temperature: float = 0.9
    force_disable_toolcalls: bool = False
    agent_madness_reward: float = 0.0
    tensor_model_parallel_size: int = 1
    multi_turn_max_sample_per_task: int = 25
    save_freq: int = 1_000_000_000
    test_freq: int = 10
    total_epochs: int = 99_999
    nnodes: int = 1
    val_pass_n: int = 4
    val_before_train: bool = False
    debug_max_parallel: int = 1
    debug_first_n_tasks: int = 1

    # ---- per-domain ----
    appworld: AppWorldConfig = field(default_factory=AppWorldConfig)
    aime: AimeConfig = field(default_factory=AimeConfig)

    def __post_init__(self) -> None:
        assert self.total_batch_size >= 1, "total_batch_size must be >= 1"
        assert self.grpo_n >= 1, "grpo_n must be >= 1"
        assert self.schedule_type in SCHEDULE_TYPES, \
            f"schedule_type must be one of {SCHEDULE_TYPES}, got {self.schedule_type}"
        assert 0.0 <= self.schedule_start <= 1.0, "schedule_start must be in [0, 1]"
        assert 0.0 <= self.schedule_end <= 1.0, "schedule_end must be in [0, 1]"
        assert self.schedule_end_step >= 0, "schedule_end_step must be >= 0"

    def get_client_0_ratio(self, global_step: int) -> float:
        if self.schedule_type == "constant" or self.schedule_end_step <= 0:
            return self.schedule_start
        if global_step >= self.schedule_end_step:
            return self.schedule_end
        t = global_step / self.schedule_end_step
        if self.schedule_type == "linear":
            return self.schedule_start + t * (self.schedule_end - self.schedule_start)
        if self.schedule_type == "cos":
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 at t=0, 0 at t=1
            return self.schedule_end + (self.schedule_start - self.schedule_end) * cos_factor
        raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

    def split_local_batch_sizes(self, global_step: int) -> tuple[int, int]:
        """Return (client_0_local_batch_size, client_1_local_batch_size) -- the
        number of distinct prompts each client should contribute this round.
        Uses round() on client_0; client_1 = total - client_0. Sum == total exactly."""
        r0 = max(0.0, min(1.0, self.get_client_0_ratio(global_step)))
        client_0_local_batch_size = int(round(self.total_batch_size * r0))
        client_1_local_batch_size = self.total_batch_size - client_0_local_batch_size
        return client_0_local_batch_size, client_1_local_batch_size


def cocktail_v2_config_from_env() -> CocktailV2Config:
    """Build the v2 config and apply env-var overrides.

    Currently supported env vars:
      COCKTAIL_RATIO_SCHEDULE = linear | cos | constant
          Override schedule_type. The same value MUST be exported in both
          clients' shells, otherwise the two will compute different per-round
          local batch sizes.
      COCKTAIL_RESULT_DIR = <path>
          Override result_dir (default './cocktail_results_v2'). Both clients
          must export the same value; otherwise their logs will diverge.
      COCKTAIL_SCHEDULE_START = <float in [0, 1]>
          Override schedule_start (client_0's batch ratio at step 0; for
          schedule_type=constant this is the ratio at every step). Both clients
          must export the same value, or they will compute different local
          batch sizes.
    """
    cfg = CocktailV2Config()
    sched_type = os.getenv("COCKTAIL_RATIO_SCHEDULE")
    if sched_type is not None:
        cfg.schedule_type = sched_type
        # Re-validate since we mutated.
        cfg.__post_init__()
        print(f"[INFO] env override: COCKTAIL_RATIO_SCHEDULE = {sched_type!r}")
    result_dir = os.getenv("COCKTAIL_RESULT_DIR")
    if result_dir is not None:
        cfg.result_dir = result_dir
        print(f"[INFO] env override: COCKTAIL_RESULT_DIR = {result_dir!r}")
    sched_start = os.getenv("COCKTAIL_SCHEDULE_START")
    if sched_start is not None:
        cfg.schedule_start = float(sched_start)
        cfg.__post_init__()
        print(f"[INFO] env override: COCKTAIL_SCHEDULE_START = {cfg.schedule_start!r}")
    return cfg
