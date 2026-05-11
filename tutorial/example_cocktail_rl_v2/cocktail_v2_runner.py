# -*- coding: utf-8 -*-
"""
Shared base class for example_cocktail_rl_v2.

Each per-domain client (AppWorld / AIME) subclasses CocktailSwarmRunner and
implements four methods (setup_data, rollout, eval_rollout, is_success).
The driver subclass additionally overrides `build_ajet_job()`. The follower
inherits the default (returns None) and waits for the engine to roll.

All configuration lives in `cocktail_v2_config.CocktailV2Config` -- this file
contains zero config defaults.
"""

from __future__ import annotations

import os
import time
import statistics
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from tqdm import tqdm

from ajet.copilot.job import AgentJetJob
from ajet.schema.task import Task
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor

from tutorial.example_cocktail_rl_v2.cocktail_v2_config import CocktailV2Config


class CocktailSwarmRunner(ABC):
    ROLE: str = ""           # "client_0" | "client_1"
    IS_DRIVER: bool = False  # whether this client drives engine startup
    CLIENT_LABEL: str = ""   # e.g. "appworld" | "aime", used in subdir + log lines
    EPISODE_TIMEOUT: int = 60

    def __init__(self, v2_config: CocktailV2Config):
        assert self.ROLE in ("client_0", "client_1"), \
            f"subclass must set ROLE; got {self.ROLE!r}"
        assert self.CLIENT_LABEL, "subclass must set CLIENT_LABEL"

        self.config = v2_config
        self.swarm_worker: Optional[SwarmClient] = None
        self.dataset = None  # must have generate_training_tasks() method
        self.eval_tasks_by_set: dict[str, list[Task]] = {}

        self.client_result_dir = os.path.join(
            v2_config.result_dir, f"results_{self.CLIENT_LABEL}"
        )
        os.makedirs(self.client_result_dir, exist_ok=True)

    # ---------------- to override ----------------

    @abstractmethod
    def setup_data(self) -> None:
        """Populate self.dataset (with generate_training_tasks() method) and self.eval_tasks_by_set."""

    @abstractmethod
    def rollout(self, task: Task) -> float:
        """Train rollout: begin_episode -> execute -> end_episode -> return reward."""

    @abstractmethod
    def eval_rollout(self, task: Task) -> float:
        """Eval rollout: begin_episode(episode_type='eval') -> execute -> abort_episode."""

    @abstractmethod
    def is_success(self, reward: float) -> bool:
        """Domain-specific success threshold for logging."""

    def build_ajet_job(self) -> Optional[AgentJetJob]:
        """Driver-only hook. Return a configured AgentJetJob; followers return None."""
        return None

    # ---------------- shared lifecycle ----------------

    def setup(self) -> None:
        self.swarm_worker = SwarmClient(self.config.swarm_url, verbose=False)
        if self.IS_DRIVER:
            ajet_job = self.build_ajet_job()
            assert ajet_job is not None, f"{type(self).__name__}.build_ajet_job() must return AgentJetJob (IS_DRIVER=True)"
            self.swarm_worker.auto_sync_train_config_and_start_engine(ajet_job)
        else:
            print("[INFO] Waiting for swarm server (ENGINE.ROLLING)...")
            self.swarm_worker._wait_until_status_change_to(desired_status="ENGINE.ROLLING")
            print("[INFO] Swarm server is ready.")

        self.setup_data()

    def run(self) -> None:
        self.setup()
        self.run_eval(n_global_step=0)
        self.train_loop()

    # ---------------- shared training ----------------

    def _get_local_batch_size(self, step: int) -> int:
        client_0_batch, client_1_batch = self.config.split_local_batch_sizes(step)
        return client_0_batch if self.ROLE == "client_0" else client_1_batch

    def train_loop(self) -> None:
        assert self.swarm_worker is not None and self.dataset is not None

        train_log_path = os.path.join(
            self.client_result_dir, f"train_results_{self.CLIENT_LABEL}.log"
        )
        last_eval_step = 0

        num_epochs = 10000
        for epoch in range(num_epochs):
            step = self.swarm_worker.get_global_step()
            local_batch_size = self._get_local_batch_size(step)

            executor = PeriodicDrainThreadPoolExecutor(
                workers=local_batch_size * self.config.grpo_n,
                max_parallel=self.config.max_env_worker,
                auto_retry=True,
            )

            for _, task in enumerate(self.dataset.generate_training_tasks()):
                for _ in range(self.config.grpo_n):
                    _, drained_results = executor.submit_with_periodic_drain(   # ✨✨✨✨
                        fn=self.rollout, task=task
                    )
                    if drained_results:
                        rewards = [r for r in drained_results if r is not None]
                        step = self.swarm_worker.get_global_step()
                        if rewards:
                            avg_reward = sum(rewards) / len(rewards)
                            std_reward = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
                            success_rate = sum(1 for r in rewards if self.is_success(r)) / len(rewards)
                            line = (
                                f"[TRAIN @ step {step}] client={self.CLIENT_LABEL}  "
                                f"batch_size={len(rewards)}  mean_reward={avg_reward:.4f}  "
                                f"std_reward={std_reward:.4f}  success_rate={success_rate*100:.2f}%"
                            )
                            print(line)
                            with open(train_log_path, "a") as f:
                                f.write(line + "\n")

                        self.swarm_worker.agree_sync_weight()
                        if step >= last_eval_step + self.config.eval_interval:
                            self.run_eval(step)
                            last_eval_step = step

                if step >= self.config.total_training_steps:
                    break

            executor.shutdown(wait=False)
            if self.swarm_worker.get_global_step() >= self.config.total_training_steps:
                break

        finish_flag = os.path.join(self.client_result_dir, "finish.flag")
        with open(finish_flag, "w") as f:
            f.write(f"Training completed at {time.time()}\n")
        print(f"[INFO] {self.CLIENT_LABEL} training complete.")

    # ---------------- shared eval ----------------

    def run_eval(self, n_global_step: int) -> None:
        if not self.eval_tasks_by_set:
            return
        eval_log_path = os.path.join(
            self.client_result_dir, f"eval_results_{self.CLIENT_LABEL}.log"
        )
        for label, eval_tasks in self.eval_tasks_by_set.items():
            self._run_eval_one(n_global_step, label, eval_tasks, eval_log_path)

    def _run_eval_one(
        self,
        n_global_step: int,
        label: str,
        eval_tasks: List[Task],
        eval_log_path: str,
    ) -> None:
        k = self.config.eval_k
        total_rollouts = len(eval_tasks) * k
        print(
            f"\n[EVAL @ step {n_global_step}] {self.CLIENT_LABEL}/{label}: "
            f"{len(eval_tasks)} tasks x {k} (pass@{k})..."
        )
        per_task_rewards: List[List[float]] = [[] for _ in eval_tasks]
        pbar = tqdm(total=total_rollouts, desc=f"EVAL {label} @ step {n_global_step}")

        with ThreadPoolExecutor(max_workers=self.config.max_env_worker) as eval_executor:
            future_to_idx = {
                eval_executor.submit(self.eval_rollout, t): i
                for i, t in enumerate(eval_tasks)
                for _ in range(k)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    per_task_rewards[idx].append(fut.result())
                except Exception as e:
                    print(f"[EVAL] future error: {e}")
                pbar.update(1)
        pbar.close()

        flat = [r for rs in per_task_rewards for r in rs if r is not None]
        if not flat:
            print(f"[EVAL @ step {n_global_step}] {self.CLIENT_LABEL}/{label}  no valid rewards")
            return

        avg = sum(flat) / len(flat)
        std = statistics.pstdev(flat) if len(flat) > 1 else 0.0
        pass1 = sum(1 for r in flat if self.is_success(r)) / len(flat)
        num_all_success_tasks = sum(
            1
            for rs in per_task_rewards
            if rs and all((r is not None and self.is_success(r)) for r in rs)
        )
        num_pass_n_tasks = sum(
            1
            for rs in per_task_rewards
            if any((r is not None and self.is_success(r)) for r in rs)
        )
        passk = num_pass_n_tasks / len(per_task_rewards)
        summary = (
            f"[EVAL @ step {n_global_step}] {self.CLIENT_LABEL}/{label}  "
            f"mean_reward={avg:.4f}  std_reward={std:.4f}  "
            f"task_pass_rate@1={pass1*100:.2f}%  task_pass_rate@{k}={passk*100:.2f}%  "
            f"n_tasks={len(per_task_rewards)}  n_rollouts={len(flat)}"
        )
        print(summary)
        with open(eval_log_path, "a") as f:
            f.write(summary + "\n")

        val_result_path = os.path.join(
            self.client_result_dir, f"val_results_{self.CLIENT_LABEL}.md"
        )
        with open(val_result_path, "a") as f:
            f.write(f"\n## Step {n_global_step} ({label})\n")
            f.write(f"- pass_n: {k}\n")
            f.write(f"- total_tasks: {len(per_task_rewards)}\n")
            f.write(f"- num_all_success_tasks: {num_all_success_tasks}\n")
            f.write(f"- num_pass_n_tasks: {num_pass_n_tasks}\n")
            f.write(f"- task_pass_rate@1: {pass1*100:.2f}%\n")
            f.write(f"- task_pass_rate@{k}: {passk*100:.2f}%\n")
            f.write(f"- mean_reward: {avg:.4f}\n")
            f.write(f"- std_reward: {std:.4f}\n")
            f.write(f"- n_rollouts: {len(flat)}\n")
