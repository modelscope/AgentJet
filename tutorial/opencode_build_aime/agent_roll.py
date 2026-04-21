# -*- coding: utf-8 -*-
"""
AIME Math Swarm Training - Agent Rollout Script

Usage:
    # First, start the swarm server:
    ajet-swarm start

    # Then run this script:
    python -m tutorial.opencode_build_aime.agent_roll
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
from tqdm import tqdm

REMOTE_MODEL_PATH = os.getenv("REMOTE_MODEL_PATH", "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct")
BATCH_SIZE = 64
ajet_job = AgentJetJob(
    algorithm="grpo",
    experiment_name="aime_swarm_14b_v3_4",
    max_env_worker=128,
    n_gpu=8,
    model=REMOTE_MODEL_PATH,
    batch_size=BATCH_SIZE,
    swarm_mode_sample_collection_method="rollout_until_finish_enough_non_dummy_tasks",
    num_repeat=8,
    logging="swanlab"
)

def load_eval_tasks(test_dataset: str) -> list:
    """Load AIME-2024 evaluation tasks."""
    eval_tasks = []
    if os.path.exists(test_dataset):
        eval_reader = HuggingFaceTaskReader(
            AjetTaskReader(huggingface_dat_repo=HuggingfaceDatRepo(dataset_path=test_dataset))
        )
        for t in eval_reader.generate_training_tasks():
            eval_tasks.append(t)
        print(f"[INFO] Loaded {len(eval_tasks)} eval tasks from AIME-2024")
    else:
        print(f"[WARN] Eval dataset not found: {test_dataset}. Skipping eval.")
    return eval_tasks




class AIMESwarmTrainer:
    """AIME Math Swarm Trainer using GRPO algorithm."""

    NUM_EPOCH = 10000
    EVAL_INTERVAL = 50  # Evaluate every EVAL_INTERVAL * REMOTE_BATCH_SIZE tasks
    EVAL_K = 2  # pass@k: run each eval task K times

    def __init__(
        self,
        swarm_url: str = None,
        train_dataset: str = None,
        test_dataset: str = None,
    ):
        self.swarm_url = swarm_url or os.getenv("AJET_SWARM_URL", "http://localhost:10086")

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.train_dataset = train_dataset or os.path.join(data_dir, "dapo-math-17k.parquet")
        self.test_dataset = test_dataset or os.path.join(data_dir, "aime-2024.parquet")

        self.swarm_worker: SwarmClient = None
        self.dataset: RouterTaskReader = None
        self.eval_tasks: list = []

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

        # Load eval tasks
        self.eval_tasks = load_eval_tasks(self.test_dataset)



    def rollout(self, task: Task) -> float:
        """Execute a single training rollout."""
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(discard_episode_timeout=120)
        workflow_output = execute_agent(task, api_baseurl_key)
        self.swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return workflow_output.reward



    def eval_rollout(self, task: Task) -> float:
        """Execute an eval rollout (results do not contribute to training)."""
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(discard_episode_timeout=120, episode_type="eval")
        try:
            workflow_output = execute_agent(task, api_baseurl_key)
            return workflow_output.reward
        finally:
            self.swarm_worker.abort_episode(episode_uuid)



    def run_eval(self, n_global_step: int):
        """Run evaluation on AIME-2024 test set."""
        if not self.eval_tasks:
            return

        k = self.EVAL_K
        total_rollouts = len(self.eval_tasks) * k
        print(f"\n[EVAL @ step {n_global_step}] Running AIME-2024 eval on {len(self.eval_tasks)} tasks x {k} (pass@{k})...")
        per_task_rewards = [[] for _ in self.eval_tasks]
        pbar = tqdm(total=total_rollouts, desc=f"EVAL @ step {n_global_step}")

        with ThreadPoolExecutor(max_workers=self.max_env_worker) as eval_executor:
            future_to_idx = {
                eval_executor.submit(self.eval_rollout, t): i
                for i, t in enumerate(self.eval_tasks)
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
                f"[EVAL @ step {n_global_step}] avg_reward={avg:.4f}  "
                f"pass@1={pass1*100:.2f}%  pass@{k}={passk*100:.2f}%  "
                f"n_tasks={len(per_task_rewards)}  n_rollouts={len(flat)}"
            )
            print(summary)
            eval_log_path = os.path.join(os.path.dirname(__file__), "eval_results.log")
            with open(eval_log_path, "a") as f:
                f.write(summary + "\n")
        else:
            print(f"[EVAL @ step {n_global_step}] no valid rewards")



    def train(self):
        """Main training loop."""
        # Run eval once before training starts (baseline)
        # self.run_eval(0)

        task_count = 0
        max_parallel = 512
        executor = TaskCountLimitedThreadPoolExecutor(
            max_parallel_groups=BATCH_SIZE,
            max_workers=max_parallel,
            auto_retry=True,
        )
        self.swarm_worker.add_entering_weight_sync_callback(executor.on_entering_weight_sync)

        for epoch in range(self.NUM_EPOCH):
            for _, task in enumerate(self.dataset.generate_training_tasks()):

                args_list = [{"task": task} for _ in range(self.grpo_n)]
                executor.submit_group(task_id=task.task_id, fn=self.rollout, args_list=args_list)

                task_count += 1

                # Periodic evaluation every EVAL_INTERVAL * REMOTE_BATCH_SIZE tasks
                time_to_eval = task_count % (self.EVAL_INTERVAL * self.remote_batch_size) == 0
                n_global_step = task_count // self.remote_batch_size
                if time_to_eval:
                    self.run_eval(n_global_step)

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
