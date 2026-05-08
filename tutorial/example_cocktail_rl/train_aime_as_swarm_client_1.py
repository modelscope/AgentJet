# -*- coding: utf-8 -*-
"""
AIME Math Swarm Training - Client 1 (Follower)
This client does NOT control training parameters - it only connects to the swarm server
started by client_0 and contributes rollouts.

python -m tutorial.example_cocktail_rl.train_aime_as_swarm_client_1
"""

import os
import time
import statistics
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ajet.schema.task import Task
from ajet.task_reader import RouterTaskReader, HuggingFaceTaskReader
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from tutorial.opencode_build_aime.agent_run_v3 import execute_agent
from tutorial.opencode_build_aime import download_data


@dataclass
class AgentConfig:
    """Minimal config for execute_agent (replaces AgentJetJob for client_1)."""
    model: str
    max_response_length: int


def load_eval_tasks(test_dataset: str, label: str = "") -> list:
    eval_tasks = []
    if os.path.exists(test_dataset):
        eval_reader = HuggingFaceTaskReader(
            AjetTaskReader(huggingface_dat_repo=HuggingfaceDatRepo(dataset_path=test_dataset))
        )
        for t in eval_reader.generate_training_tasks():
            eval_tasks.append(t)
        print(f"[INFO] Loaded {len(eval_tasks)} eval tasks from {label or test_dataset}")
    else:
        print(f"[WARN] Eval dataset not found: {test_dataset}. Skipping {label or test_dataset}.")
    return eval_tasks


class AIMESwarmClient:
    """AIME swarm client that follows server started by client_0."""

    def __init__(
        self,
        swarm_url: str,
        result_dir: str,
        max_env_worker: int = 128,
        eval_interval: int = 10,
        eval_k: int = 4,
        grpo_n: int = 4,
    ):
        self.swarm_url = swarm_url or os.getenv("AJET_SWARM_URL", "http://localhost:10086")
        self.result_dir = result_dir
        self.max_env_worker = max_env_worker
        self.eval_interval = eval_interval
        self.eval_k = eval_k
        self.grpo_n = grpo_n

        data_dir = os.path.join(os.path.dirname(__file__), "..", "opencode_build_aime", "data")
        self.train_dataset = os.path.join(data_dir, "dapo-math-17k.parquet")
        self.test_datasets = {
            "AIME-2025": os.path.join(data_dir, "aime-2025.parquet"),
            "AIME-2026": os.path.join(data_dir, "aime-2026.parquet"),
            "DAPO-Math-Tiny-Val": os.path.join(data_dir, "dapo-math-tiny-val.parquet"),
        }

        self.swarm_worker: SwarmClient | None = None
        self.dataset: RouterTaskReader | None = None
        self.eval_tasks_by_set: dict[str, list[Task]] = {}
        self.agent_config: AgentConfig | None = None

        os.makedirs(result_dir, exist_ok=True)

    def setup(self):
        if not os.path.exists(self.train_dataset):
            raise FileNotFoundError(
                f"Training dataset not found: {self.train_dataset}\n"
                "Please run: proxychains python -m tutorial.opencode_build_aime.download_data"
            )

        self.dataset = RouterTaskReader(
            reader_type="huggingface_dat_repo",
            reader_config=AjetTaskReader(
                huggingface_dat_repo=HuggingfaceDatRepo(dataset_path=self.train_dataset)
            )
        )

        self.swarm_worker = SwarmClient(self.swarm_url, verbose=False)
        print("[INFO] Waiting for swarm server to be ready (ENGINE.ROLLING)...")
        self.swarm_worker._wait_until_status_change_to(desired_status="ENGINE.ROLLING")
        print("[INFO] Swarm server is ready.")

        # Config from env vars (must match server config from client_0)
        max_response_length = int(os.getenv("COCKTAIL_MAX_RESPONSE_LENGTH", "20000"))
        self.agent_config = AgentConfig(model="dummy", max_response_length=max_response_length)

        # Load eval datasets
        eval_downloaders = {
            "AIME-2025": download_data.ensure_aime_2025,
            "AIME-2026": download_data.ensure_aime_2026,
        }
        for label, path in self.test_datasets.items():
            if not os.path.exists(path):
                downloader = eval_downloaders.get(label)
                if downloader is None:
                    print(f"[WARN] {label} parquet missing at {path} and no downloader registered. Skipping.")
                    continue
                print(f"[INFO] {label} parquet missing, downloading...")
                try:
                    downloader()
                except Exception as e:
                    print(f"[WARN] Failed to download {label}: {e}")
                    continue
            tasks = load_eval_tasks(path, label=label)
            if tasks:
                self.eval_tasks_by_set[label] = tasks

    def rollout(self, task: Task) -> float:
        assert self.swarm_worker is not None and self.agent_config is not None
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(discard_episode_timeout=120)
        workflow_output = execute_agent(task, api_baseurl_key, self.agent_config)
        self.swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return workflow_output.reward

    def eval_rollout(self, task: Task) -> float:
        assert self.swarm_worker is not None and self.agent_config is not None
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(
            discard_episode_timeout=120, episode_type="eval"
        )
        try:
            workflow_output = execute_agent(task, api_baseurl_key, self.agent_config)
            return workflow_output.reward
        finally:
            self.swarm_worker.abort_episode(episode_uuid)

    def run_eval(self, n_global_step: int):
        if not self.eval_tasks_by_set:
            return
        eval_log_path = os.path.join(self.result_dir, "eval_results.log")

        for label, eval_tasks in self.eval_tasks_by_set.items():
            self._run_eval_one(n_global_step, label, eval_tasks, eval_log_path)

    def _run_eval_one(self, n_global_step: int, label: str, eval_tasks: list, eval_log_path: str):
        k = self.eval_k
        total_rollouts = len(eval_tasks) * k
        print(f"\n[EVAL @ step {n_global_step}] Running {label} eval on {len(eval_tasks)} tasks x {k} (pass@{k})...")
        per_task_rewards = [[] for _ in eval_tasks]
        pbar = tqdm(total=total_rollouts, desc=f"EVAL {label} @ step {n_global_step}")

        with ThreadPoolExecutor(max_workers=self.max_env_worker) as eval_executor:
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
        if flat:
            avg = sum(flat) / len(flat)
            std_reward = statistics.pstdev(flat) if len(flat) > 1 else 0.0
            pass1 = sum(1 for r in flat if r > 0) / len(flat)
            num_all_success_tasks = sum(
                1 for rs in per_task_rewards if rs and all((r is not None and r > 0) for r in rs)
            )
            solved_tasks = [rs for rs in per_task_rewards if any((r is not None and r > 0) for r in rs)]
            num_pass_n_tasks = len(solved_tasks)
            passk = num_pass_n_tasks / len(per_task_rewards)
            summary = (
                f"[EVAL @ step {n_global_step}] {label}  mean_reward={avg:.4f}  std_reward={std_reward:.4f}  "
                f"task_pass_rate@1={pass1*100:.2f}%  task_pass_rate@{k}={passk*100:.2f}%  "
                f"n_tasks={len(per_task_rewards)}  n_rollouts={len(flat)}"
            )
            print(summary)
            with open(eval_log_path, "a") as f:
                f.write(summary + "\n")

            val_result_path = os.path.join(self.result_dir, "val_results.md")
            with open(val_result_path, "a") as f:
                f.write(f"\n## Step {n_global_step}\n")
                f.write(f"- pass_n: {k}\n")
                f.write(f"- total_tasks: {len(per_task_rewards)}\n")
                f.write(f"- num_all_success_tasks: {num_all_success_tasks}\n")
                f.write(f"- num_pass_n_tasks: {num_pass_n_tasks}\n")
                f.write(f"- task_pass_rate@1: {pass1*100:.2f}%\n")
                f.write(f"- task_pass_rate@{k}: {passk*100:.2f}%\n")
                f.write(f"- mean_reward: {avg:.4f}\n")
                f.write(f"- std_reward: {std_reward:.4f}\n")
                f.write(f"- n_rollouts: {len(flat)}\n")
        else:
            print(f"[EVAL @ step {n_global_step}] {label}  no valid rewards")

    def train(self):
        assert self.swarm_worker is not None and self.dataset is not None

        last_eval_step = 0
        # self.run_eval(0)  # skip initial eval for faster iteration

        # Use same executor pattern as client_0 for proper weight sync
        batch_size = 64  # must match server config
        executor = PeriodicDrainThreadPoolExecutor(
            workers=self.grpo_n * batch_size, max_parallel=64, auto_retry=True
        )

        train_log_path = os.path.join(self.result_dir, "train_results.log")

        n_global_step = 0
        num_epochs = 10000
        for epoch in range(num_epochs):
            for _, task in enumerate(self.dataset.generate_training_tasks()):
                for _ in range(self.grpo_n):
                    _, drained_results = executor.submit_with_periodic_drain(
                        fn=self.rollout, task=task
                    )
                    if drained_results:
                        # Log batch rewards before weight sync
                        rewards = [r for r in drained_results if r is not None]
                        if rewards:
                            avg_reward = sum(rewards) / len(rewards)
                            std_reward = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
                            success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
                            step = self.swarm_worker.get_global_step()
                            log_line = (
                                f"[TRAIN @ step {step}] client=aime  "
                                f"batch_size={len(rewards)}  mean_reward={avg_reward:.4f}  "
                                f"std_reward={std_reward:.4f}  success_rate={success_rate*100:.2f}%"
                            )
                            print(log_line)
                            with open(train_log_path, "a") as f:
                                f.write(log_line + "\n")
                        self.swarm_worker.agree_sync_weight()

                n_global_step = self.swarm_worker.get_global_step()

                if n_global_step >= last_eval_step + self.eval_interval:
                    self.run_eval(n_global_step)
                    last_eval_step = n_global_step

        finish_flag = os.path.join(self.result_dir, "finish.flag")
        with open(finish_flag, "w") as f:
            f.write(f"Training completed at {time.time()}\n")

        print("\n[INFO] Training complete!")

    def run(self):
        self.setup()
        self.train()


def main():
    # Hardcoded config (must match client_0 / cocktail_rl_conf.yaml)
    SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
    RESULT_DIR = "./cocktail_training_new/results_aime"
    MAX_ENV_WORKER = 128
    EVAL_INTERVAL = 10
    EVAL_K = 4
    GRPO_N = 4

    client = AIMESwarmClient(
        swarm_url=SWARM_URL,
        result_dir=RESULT_DIR,
        max_env_worker=MAX_ENV_WORKER,
        eval_interval=EVAL_INTERVAL,
        eval_k=EVAL_K,
        grpo_n=GRPO_N,
    )
    client.run()


if __name__ == "__main__":
    main()
