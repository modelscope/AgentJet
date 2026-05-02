# -*- coding: utf-8 -*-
"""
AIME Math Swarm Training - Auto Research Client
Configurable for batch_size and max_response_length_in_one_turn experiments
"""

import os
import sys
import argparse
import time
import statistics
from urllib.parse import urlparse
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


DEFAULT_PROJECT_NAME = "subject12_aime_kl_reward_study"


def extract_swarm_port(swarm_url: str) -> int:
    parsed = urlparse(swarm_url)
    if parsed.port is None:
        raise ValueError(f"Swarm URL must include an explicit port: {swarm_url}")
    return parsed.port


def validate_length_config(
    max_prompt_length: int,
    max_response_length: int,
    max_response_length_in_one_turn: int,
    max_model_len: int,
):
    if max_prompt_length + max_response_length > max_model_len:
        raise ValueError(
            "Invalid length config: max_prompt_length + max_response_length must be <= max_model_len"
        )
    if max_response_length_in_one_turn > max_response_length:
        raise ValueError(
            "Invalid length config: max_response_length_in_one_turn must be <= max_response_length"
        )


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


class AIMEAutoResearchTrainer:
    def __init__(
        self,
        batch_size: int,
        max_response_length_in_one_turn: int,
        experiment_name: str,
        result_dir: str,
        swarm_url: str = None,
        project_name: str = DEFAULT_PROJECT_NAME,
        resolved_yaml_path: str | None = None,
        prepare_only: bool = False,
        max_prompt_length: int = 3000,
        max_response_length: int = 15000,
        max_model_len: int = 18000,
        total_training_steps: int = 60,
        n_gpu: int = 8,
        max_env_worker: int = 128,
        eval_interval: int = 10,
        eval_k: int = 4,
        grpo_repeat: int = 8,
        ppo_epochs: int = 1,
        mini_batch_num: int = 1,
    ):
        self.swarm_url = swarm_url or os.getenv("AJET_SWARM_URL", "http://localhost:10086")
        self.batch_size = batch_size
        self.max_response_length_in_one_turn = max_response_length_in_one_turn
        self.experiment_name = experiment_name
        self.result_dir = result_dir
        self.project_name = project_name
        self.resolved_yaml_path = resolved_yaml_path or os.path.join(result_dir, "resolved_swarm_config.yaml")
        self.prepare_only = prepare_only
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_len = max_model_len
        self.total_training_steps = total_training_steps
        self.n_gpu = n_gpu

        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.train_dataset = os.path.join(data_dir, "dapo-math-17k.parquet")
        self.test_datasets = {
            # "AIME-2024": os.path.join(data_dir, "aime-2024.parquet"),
            "AIME-2025": os.path.join(data_dir, "aime-2025.parquet"),
            "AIME-2026": os.path.join(data_dir, "aime-2026.parquet"),
            "DAPO-Math-Tiny-Val": os.path.join(data_dir, "dapo-math-tiny-val.parquet"),
        }

        self.swarm_worker: SwarmClient | None = None
        self.dataset: RouterTaskReader | None = None
        self.eval_tasks_by_set: dict[str, list[Task]] = {}
        self.eval_interval = eval_interval
        self.eval_k = eval_k
        self.grpo_n = grpo_repeat
        self.ppo_epochs = ppo_epochs
        self.mini_batch_num = mini_batch_num
        self.max_env_worker = max_env_worker

        os.makedirs(result_dir, exist_ok=True)
        model_path = os.getenv("REMOTE_MODEL_PATH", "/mnt/data_cpfs/xielipeng.xlp/models/Qwen3-8B")
        validate_length_config(
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            max_response_length_in_one_turn=max_response_length_in_one_turn,
            max_model_len=max_model_len,
        )
        self.ajet_job = AgentJetJob(
            ensure_new_experiment=True,
            experiment_dir=result_dir,
            project_name=project_name,
            algorithm="grpo",
            experiment_name=experiment_name,
            max_env_worker=max_env_worker,
            n_gpu=n_gpu,
            model=model_path,
            batch_size=batch_size,
            swarm_mode=True,
            swarm_mode_sample_collection_method="rollout_until_finish_enough_non_dummy_tasks",
            num_repeat=grpo_repeat,
            ppo_epochs=ppo_epochs,
            mini_batch_num=mini_batch_num,
            logging="swanlab",
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            max_response_length_in_one_turn=max_response_length_in_one_turn,
            max_model_len=max_model_len,
            val_print_to_markdown_file_path=os.path.join(result_dir, "val_results.md"),
            train_print_to_markdown_file_path=os.path.join(result_dir, "train_results.md"),
            total_training_steps=total_training_steps,
        )
        self.ajet_job.config.ajet.execute_test = False
        self.ajet_job.config.ajet.interchange_server.interchange_server_port = extract_swarm_port(self.swarm_url)
        self.ajet_job.config.ajet.trainer_common.test_freq = eval_interval
        self.ajet_job.config.ajet.trainer_common.save_freq = 10**9
        self.ajet_job.config.ajet.trainer_common.total_epochs = 10000
        self.ajet_job.config.ajet.trainer_common.val_pass_n = eval_k
        # Swarm mode cannot enable val_before_train, so the script runs an explicit step-0 eval instead.
        self.ajet_job.config.ajet.trainer_common.val_before_train = False

    def setup(self):
        if not os.path.exists(self.train_dataset):
            raise FileNotFoundError(
                f"Training dataset not found: {self.train_dataset}\n"
                "Please run: proxychains python -m tutorial.opencode_build_aime.download_data"
            )

        self.ajet_job.dump_job_as_yaml(self.resolved_yaml_path)

        if self.prepare_only:
            return

        self.dataset = RouterTaskReader(
            reader_type="huggingface_dat_repo",
            reader_config=AjetTaskReader(
                huggingface_dat_repo=HuggingfaceDatRepo(dataset_path=self.train_dataset)
            )
        )

        self.swarm_worker = SwarmClient(self.swarm_url, verbose=False)
        self.swarm_worker.auto_sync_train_config_and_start_engine(
            self.ajet_job,
            force_restart=os.getenv("AJET_SWARM_RESTART", "0") == "1"
        )

        eval_downloaders = {
            "AIME-2024": download_data.ensure_aime_2024,
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
        assert self.swarm_worker is not None, "setup() must be called before rollout()"
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(discard_episode_timeout=120)
        workflow_output = execute_agent(task, api_baseurl_key, self.ajet_job)
        self.swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return workflow_output.reward

    def eval_rollout(self, task: Task) -> float:
        assert self.swarm_worker is not None, "setup() must be called before eval_rollout()"
        episode_uuid, api_baseurl_key = self.swarm_worker.begin_episode(
            discard_episode_timeout=120, episode_type="eval"
        )
        try:
            workflow_output = execute_agent(task, api_baseurl_key, self.ajet_job)
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
        assert self.swarm_worker is not None and self.dataset is not None, "setup() must be called before train()"
        self.run_eval(0)

        max_parallel = 64
        executor = TaskCountLimitedThreadPoolExecutor(
            max_parallel_groups=self.batch_size,
            max_workers=max_parallel,
            auto_retry=True,
        )
        self.swarm_worker.add_entering_weight_sync_callback(executor.on_entering_weight_sync)

        num_epochs = 10000
        last_eval_step = 0
        for epoch in range(num_epochs):
            for _, task in enumerate(self.dataset.generate_training_tasks()):
                args_list = [{"task": task} for _ in range(self.grpo_n)]
                executor.submit_group(task_id=task.task_id, fn=self.rollout, args_list=args_list)

                n_global_step = self.swarm_worker.get_global_step()

                time_to_eval = n_global_step >= last_eval_step + self.eval_interval
                if time_to_eval:
                    self.run_eval(n_global_step)
                    last_eval_step = n_global_step

                if n_global_step >= self.total_training_steps:
                    break

            if n_global_step >= self.total_training_steps:
                break

        finish_flag = os.path.join(self.result_dir, "finish.flag")
        with open(finish_flag, "w") as f:
            f.write(f"Training completed at {time.time()}\n")

        print("\n[INFO] Training complete!")

    def run(self):
        self.setup()
        if self.prepare_only:
            print(f"[INFO] Prepared run artifacts only. Resolved config written to {self.resolved_yaml_path}")
            return
        self.train()


def main():
    parser = argparse.ArgumentParser(description="AIME Auto Research Swarm Training")
    parser.add_argument("--batch-size", default=32, type=int, required=True, help="Training batch size")
    parser.add_argument("--experiment-name", type=str, required=True,
                        help="Experiment name for this run")
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Directory to store results")
    parser.add_argument("--swarm-url", type=str, default="http://localhost:10086",
                        help="Swarm server URL")
    parser.add_argument("--project-name", type=str, default=DEFAULT_PROJECT_NAME,
                        help="Shared project name used for this research line")
    parser.add_argument("--resolved-yaml-path", type=str, default=None,
                        help="Optional output path for the fully resolved swarm config yaml")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Build the config, dump the resolved yaml, and exit without training")
    parser.add_argument("--max-response-length-in-one-turn", type=int, default=12000,
                        help="Max response length in one turn")
    parser.add_argument("--max-prompt-length", type=int, default=3000,
                        help="Maximum prompt length")
    parser.add_argument("--max-response-length", type=int, default=20000,
                        help="Maximum total response length")
    parser.add_argument("--max-model-len", type=int, default=23000,
                        help="Maximum total model context length")
    parser.add_argument("--total-training-steps", type=int, default=120,
                        help="Hard cap on total training steps")
    parser.add_argument("--n-gpu", type=int, default=8,
                        help="Number of GPUs reserved for the swarm server")
    parser.add_argument("--max-env-worker", type=int, default=128,
                        help="Estimated number of parallel environment workers")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate every N global steps")
    parser.add_argument("--eval-k", type=int, default=4,
                        help="Number of rollouts per eval task (pass@k)")
    parser.add_argument("--grpo-repeat", type=int, default=8,
                        help="GRPO num_repeat per training task")
    parser.add_argument("--ppo-epochs", type=int, default=1,
                        help="Number of PPO epochs per update")
    parser.add_argument("--mini-batch-num", type=int, default=1,
                        help="Number of mini-batches per PPO update")
    args = parser.parse_args()

    trainer = AIMEAutoResearchTrainer(
        batch_size=args.batch_size,
        max_response_length_in_one_turn=args.max_response_length_in_one_turn,
        experiment_name=args.experiment_name,
        result_dir=args.result_dir,
        swarm_url=args.swarm_url,
        project_name=args.project_name,
        resolved_yaml_path=args.resolved_yaml_path,
        prepare_only=args.prepare_only,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_model_len=args.max_model_len,
        total_training_steps=args.total_training_steps,
        n_gpu=args.n_gpu,
        max_env_worker=args.max_env_worker,
        eval_interval=args.eval_interval,
        eval_k=args.eval_k,
        grpo_repeat=args.grpo_repeat,
        ppo_epochs=args.ppo_epochs,
        mini_batch_num=args.mini_batch_num,
    )
    trainer.run()


if __name__ == "__main__":
    main()
