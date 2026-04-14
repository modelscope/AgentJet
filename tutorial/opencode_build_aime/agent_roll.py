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
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from tqdm import tqdm

REMOTE_MODEL_PATH = os.getenv("REMOTE_MODEL_PATH", "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct")
ajet_job = AgentJetJob(
    algorithm="grpo",
    experiment_name="aime_swarm_14b",
    max_env_worker=128,
    n_gpu=8,
    model=REMOTE_MODEL_PATH,
    batch_size=128,
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


def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
    """Execute the AIME agent."""
    from tutorial.opencode_build_aime.agent_run import execute_agent as _execute_agent
    return _execute_agent(task, api_baseurl_key)

class AIMESwarmTrainer:
    """AIME Math Swarm Trainer using GRPO algorithm."""

    NUM_EPOCH = 10000
    EVAL_INTERVAL = 50  # Evaluate every EVAL_INTERVAL * REMOTE_BATCH_SIZE tasks

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
            # force_restart=True,
        )

        self.grpo_n = ajet_job.num_repeat
        self.remote_batch_size = ajet_job.batch_size
        self.max_env_worker = ajet_job.max_env_worker

        # Load eval tasks
        self.eval_tasks = load_eval_tasks(self.test_dataset)

    def rollout(self, task: Task) -> float:
        """Execute a single training rollout."""
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(discard_episode_timeout=60)
        workflow_output = execute_agent(task, api_baseurl_key)
        self.swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return workflow_output.reward

    def eval_rollout(self, task: Task) -> float:
        """Execute an eval rollout (results do not contribute to training)."""
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(discard_episode_timeout=60, episode_type="eval")
        try:
            workflow_output = execute_agent(task, api_baseurl_key)
            return workflow_output.reward
        finally:
            self.swarm_worker.abort_episode(episode_uuid)

    def run_eval(self, task_count: int):
        """Run evaluation on AIME-2024 test set."""
        if not self.eval_tasks:
            return

        print(f"\n[EVAL @ task {task_count}] Running AIME-2024 eval on {len(self.eval_tasks)} tasks...")
        drained = []
        pbar = tqdm(total=len(self.eval_tasks), desc=f"EVAL @ {task_count}")

        with ThreadPoolExecutor(max_workers=self.max_env_worker) as eval_executor:
            futures = [eval_executor.submit(self.eval_rollout, t) for t in self.eval_tasks]
            for fut in as_completed(futures):
                try:
                    drained.append(fut.result())
                except Exception as e:
                    print(f"[EVAL] future error: {e}")
                pbar.update(1)
        pbar.close()

        rewards = [r for r in drained if r is not None]
        if rewards:
            avg = sum(rewards) / len(rewards)
            acc = sum(1 for r in rewards if r > 0) / len(rewards)
            print(f"[EVAL @ task {task_count}] avg_reward={avg:.4f}  pass@1={acc*100:.2f}%  n={len(rewards)}")
        else:
            print(f"[EVAL @ task {task_count}] no valid rewards")

    def train(self):
        """Main training loop."""
        # Run eval once before training starts (baseline)
        self.run_eval(0)

        task_count = 0
        executor = PeriodicDrainThreadPoolExecutor(
            workers=self.grpo_n * self.remote_batch_size,
            max_parallel=64,
            auto_retry=True
        )

        for epoch in range(self.NUM_EPOCH):
            for _, task in enumerate(self.dataset.generate_training_tasks()):
                for _ in range(self.grpo_n):
                    executor.submit_with_periodic_drain(fn=self.rollout, task=task)

                task_count += 1

                # Periodic evaluation every EVAL_INTERVAL * REMOTE_BATCH_SIZE tasks
                if task_count % (self.EVAL_INTERVAL * self.remote_batch_size) == 0:
                    self.run_eval(task_count)

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
