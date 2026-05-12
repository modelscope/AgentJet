# -*- coding: utf-8 -*-
"""
AIME Math Swarm Training - Agent Rollout Script

Usage:
    # First, start the swarm server:
    ajet-swarm start

    # Then run this script:
    python -m tutorial.opencode_build_aime.agent_roll_v3
"""

import os
from ajet.schema.task import Task
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader, HuggingFaceTaskReader
from ajet.utils.thread_executors import TaskCountLimitedThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from tutorial.opencode_build_aime.agent_run_v3 import execute_agent
from tutorial.opencode_build_aime import download_data
from tqdm import tqdm

NUM_EPOCH = 10000
EVAL_INTERVAL = 20  # Evaluate every EVAL_INTERVAL * REMOTE_BATCH_SIZE tasks
EVAL_K = 2  # pass@k: run each eval task K times
REMOTE_MODEL_PATH = os.getenv("REMOTE_MODEL_PATH", "/mnt/data_cpfs/xielipeng.xlp/models/Qwen3-14B")
BATCH_SIZE = 16
PPO_EPOCH = 2
NUM_REPEAT = 8
MINI_BATCH_NUM = 2
ajet_job = AgentJetJob(
    ensure_new_experiment=True,
    algorithm="grpo",
    experiment_name="aime_swarm_14b_v33_ppoepoch4_v3",
    max_env_worker=128,
    n_gpu=8,
    model=REMOTE_MODEL_PATH,
    batch_size=BATCH_SIZE,
    swarm_mode_sample_collection_method="rollout_until_finish_enough_non_dummy_tasks",
    num_repeat=NUM_REPEAT,
    ppo_epochs=PPO_EPOCH,
    mini_batch_num=MINI_BATCH_NUM,
    logging="swanlab",
    max_prompt_length=3000,
    max_response_length=15000,
    max_response_length_in_one_turn=10000,
    max_model_len=18000
)

def load_eval_tasks(test_dataset: str, label: str = "") -> list:
    """Load AIME evaluation tasks from a single parquet file."""
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




class AIMESwarmTrainer:
    """AIME Math Swarm Trainer using GRPO algorithm."""


    def __init__(
        self,
        swarm_url: str = None,
        train_dataset: str = None,
        test_datasets: dict = None,
    ):
        self.swarm_url = swarm_url or os.getenv("AJET_SWARM_URL", "http://localhost:10086")

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.train_dataset = train_dataset or os.path.join(data_dir, "dapo-math-17k.parquet")
        self.test_datasets = test_datasets or {
            "AIME-2024": (os.path.join(data_dir, "aime-2024.parquet"), download_data.ensure_aime_2024),
            "AIME-2025": (os.path.join(data_dir, "aime-2025.parquet"), download_data.ensure_aime_2025),
            "AIME-2026": (os.path.join(data_dir, "aime-2026.parquet"), download_data.ensure_aime_2026),
        }

        self.swarm_worker: SwarmClient = None
        self.dataset: RouterTaskReader = None
        self.eval_tasks_by_set: dict = {}

        self.grpo_n: int = None
        self.remote_batch_size: int = None



    def setup(self):
        """Initialize dataset, job config, and swarm connection."""
        if not os.path.exists(self.train_dataset):
            raise FileNotFoundError(
                f"Training dataset not found: {self.train_dataset}\n"
                "Please run: proxychains python -m tutorial.opencode_build_aime.download_data"
            )

        # Initialize dataset reader
        self.dataset = RouterTaskReader(
            reader_type="huggingface_dat_repo",
            reader_config=AjetTaskReader(
                huggingface_dat_repo=HuggingfaceDatRepo(
                    dataset_path=self.train_dataset
                )
            )
        )

        # Connect to swarm server
        self.swarm_worker = SwarmClient(self.swarm_url, verbose=False)
        self.swarm_worker.auto_sync_train_config_and_start_engine(
            ajet_job,
            force_restart=os.getenv("AJET_SWARM_RESTART", "0") == "1"
        )

        self.grpo_n = ajet_job.num_repeat
        self.remote_batch_size = ajet_job.batch_size
        self.max_env_worker = ajet_job.max_env_worker

        # Load eval tasks for each test set, auto-downloading any missing parquet
        for label, (path, ensure_fn) in self.test_datasets.items():
            if not os.path.exists(path):
                print(f"[INFO] {label} parquet missing, downloading...")
                try:
                    ensure_fn()
                except Exception as e:
                    print(f"[WARN] Failed to download {label}: {e}")
                    continue
            tasks = load_eval_tasks(path, label=label)
            if tasks:
                self.eval_tasks_by_set[label] = tasks



    def rollout(self, task: Task) -> float | None:
        """Execute a single training rollout."""
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(discard_episode_timeout=120)
        workflow_output = execute_agent(task, api_baseurl_key, ajet_job)
        self.swarm_worker.end_episode(task, episode_uuid, workflow_output)
        reward = workflow_output.reward
        return reward[0] if isinstance(reward, list) else reward



    def eval_rollout(self, task: Task) -> float | None:
        """Execute an eval rollout (results do not contribute to training)."""
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(discard_episode_timeout=120, episode_type="eval")
        try:
            workflow_output = execute_agent(task, api_baseurl_key, ajet_job)
            reward = workflow_output.reward
            return reward[0] if isinstance(reward, list) else reward
        finally:
            self.swarm_worker.abort_episode(episode_uuid)



    def run_eval(self, n_global_step: int):
        """Run evaluation on every loaded AIME test set."""
        if not self.eval_tasks_by_set:
            return
        eval_log_path = os.path.join(self.swarm_worker.server_experiment_dir(), "eval_results.log")
        print(eval_log_path)

        for label, eval_tasks in self.eval_tasks_by_set.items():
            self._run_eval_one(n_global_step, label, eval_tasks, eval_log_path)

    def _run_eval_one(self, n_global_step: int, label: str, eval_tasks: list, eval_log_path: str):
        """Run evaluation on a single AIME test set."""
        k = EVAL_K
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
            pass1 = sum(1 for r in flat if r > 0) / len(flat)
            solved_tasks = [rs for rs in per_task_rewards if any((r is not None and r > 0) for r in rs)]
            passk = len(solved_tasks) / len(per_task_rewards)
            summary = (
                f"[EVAL @ step {n_global_step}] {label}  avg_reward={avg:.4f}  "
                f"pass@1={pass1*100:.2f}%  pass@{k}={passk*100:.2f}%  "
                f"n_tasks={len(per_task_rewards)}  n_rollouts={len(flat)}"
            )
            print(summary)
            with open(eval_log_path, "a") as f:
                f.write(summary + "\n")
        else:
            print(f"[EVAL @ step {n_global_step}] {label}  no valid rewards")



    def train(self):
        """Main training loop."""
        # Run eval once before training starts (baseline)
        self.run_eval(0)
        last_eval_step = 0

        max_parallel = 64
        executor = TaskCountLimitedThreadPoolExecutor(
            max_parallel_groups=BATCH_SIZE,
            max_workers=max_parallel,
            auto_retry=True,
        )
        self.swarm_worker.add_entering_weight_sync_callback(executor.on_entering_weight_sync)

        for epoch in range(NUM_EPOCH):
            for _, task in enumerate(self.dataset.generate_training_tasks()):

                args_list = [{"task": task} for _ in range(self.grpo_n)]
                executor.submit_group(task_id=task.task_id, fn=self.rollout, args_list=args_list)

                n_global_step = self.swarm_worker.get_global_step()

                time_to_eval = n_global_step >= last_eval_step + EVAL_INTERVAL
                if time_to_eval:
                    self.run_eval(n_global_step)
                    last_eval_step = n_global_step

        print("\n[INFO] Training complete!")



    def run(self):
        """Setup and start training."""
        self.setup()
        self.train()



def main():
    trainer = AIMESwarmTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
