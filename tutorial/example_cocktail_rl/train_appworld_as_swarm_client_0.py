# -*- coding: utf-8 -*-

# python -m tutorial.example_cocktail_rl.train_appworld_as_swarm_client_0

import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, List

from tqdm import tqdm

from ajet.copilot.job import AgentJetJob
from ajet.schema.task import Task
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from ajet.utils.env_service_client.env_client_ng import EnvClient
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor

NUM_EPOCH = 10000
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")

ENV_URL = os.getenv("APPWORLD_ENV_URL", "http://127.0.0.1:8080")
ENV_TYPE = os.getenv("APPWORLD_ENV_TYPE", "appworld")
TRAINING_SPLIT = os.getenv("APPWORLD_TRAINING_SPLIT", "train")
VALIDATION_SPLIT = os.getenv("APPWORLD_VALIDATION_SPLIT", "dev")
MAX_STEPS = int(os.getenv("APPWORLD_MAX_STEPS", "25"))

EVAL_INTERVAL = int(os.getenv("APPWORLD_EVAL_INTERVAL", "10"))
EVAL_K = int(os.getenv("APPWORLD_EVAL_K", "1"))
TOTAL_TRAINING_STEPS = int(os.getenv("APPWORLD_TOTAL_TRAINING_STEPS", "200"))
RESULT_DIR = os.getenv("APPWORLD_RESULT_DIR", "./cocktail_training_new/results_appworld")
MAX_ENV_WORKER = int(os.getenv("APPWORLD_MAX_ENV_WORKER", "64"))


def get_appworld_tasks(split: str) -> List[Task]:
    """Enumerate appworld task ids from env_service for the given split.

    The swarm client owns task generation, so we hit env_service directly
    (rather than going through `EnvServiceTaskReader`) to keep the config
    surface flat.
    """
    env_client = EnvClient(base_url=ENV_URL)
    task_id_array = env_client.get_env_profile(ENV_TYPE, split=split)
    if len(task_id_array) == 0:
        raise ValueError(
            f"No task_id found for env_type={ENV_TYPE}, split={split}, "
            f"check connection to {ENV_URL}"
        )
    return [
        Task(
            main_query="[not defined]",
            init_messages=[],
            task_id=str(task_id),
            env_type=ENV_TYPE,
            metadata={},
        )
        for task_id in task_id_array
    ]


def generate_training_tasks() -> Generator[Task, None, None]:
    for task in get_appworld_tasks(TRAINING_SPLIT):
        yield task


def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
    import asyncio
    from tutorial.example_appworld_swarm.appworld_swarm import ExampleAgentScopeWorkflow
    workflow = ExampleAgentScopeWorkflow(
        env_url=ENV_URL,
        env_type=ENV_TYPE,
        max_steps=MAX_STEPS,
    )
    return asyncio.run(workflow.execute(task, api_baseurl_key))


def main():

    ajet_job = AgentJetJob(
        base_yaml_config="tutorial/example_cocktail_rl/cocktail_rl_conf.yaml",
        algorithm="grpo",
        experiment_name="cocktail_rl",
        max_env_worker=MAX_ENV_WORKER,
    )

    # Hand shake with remote swarm server
    swarm_worker = SwarmClient(AJET_SWARM_URL)
    swarm_worker.auto_sync_train_config_and_start_engine(
        ajet_job,
        # force_restart=True,
    )

    GRPO_N = ajet_job.num_repeat
    REMOTE_BATCH_SIZE = ajet_job.batch_size

    os.makedirs(RESULT_DIR, exist_ok=True)
    eval_log_path = os.path.join(RESULT_DIR, "eval_results.log")
    val_result_path = os.path.join(RESULT_DIR, "val_results.md")

    eval_tasks = get_appworld_tasks(VALIDATION_SPLIT)
    print(f"[INFO] Loaded {len(eval_tasks)} eval tasks (split={VALIDATION_SPLIT})")

    def rollout(task: Task) -> float:
        # begin episode
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=600)
        # execute agent ( base_url = api_baseurl_key.base_url, api_key = api_baseurl_key.api_key )
        workflow_output = execute_agent(task, api_baseurl_key)
        # report output back to swarm remote
        swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return workflow_output.reward

    def eval_rollout(task: Task) -> float:
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(
            discard_episode_timeout=600, episode_type="eval"
        )
        try:
            workflow_output = execute_agent(task, api_baseurl_key)
            return workflow_output.reward
        finally:
            # eval samples must NOT be fed back into the training pool
            swarm_worker.abort_episode(episode_uuid)

    def run_eval(n_global_step: int):
        if not eval_tasks:
            return
        k = EVAL_K
        total_rollouts = len(eval_tasks) * k
        print(f"\n[EVAL @ step {n_global_step}] {len(eval_tasks)} tasks x {k} (pass@{k})...")
        per_task_rewards: List[List[float]] = [[] for _ in eval_tasks]
        pbar = tqdm(total=total_rollouts, desc=f"EVAL @ step {n_global_step}")

        with ThreadPoolExecutor(max_workers=MAX_ENV_WORKER) as eval_executor:
            future_to_idx = {
                eval_executor.submit(eval_rollout, t): i
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
            print(f"[EVAL @ step {n_global_step}] no valid rewards")
            return

        avg = sum(flat) / len(flat)
        std_reward = statistics.pstdev(flat) if len(flat) > 1 else 0.0
        # Full success requires raw_reward >= 1 (final_reward >= 1.5).
        # Partial-credit rollouts have 0 < final_reward <= 0.5, so they must NOT
        # count as passes; see EnvServiceJudge.compute_reward.
        SUCCESS_THRESHOLD = 1.0
        pass1 = sum(1 for r in flat if r >= SUCCESS_THRESHOLD) / len(flat)
        num_all_success_tasks = sum(
            1
            for rs in per_task_rewards
            if rs and all((r is not None and r >= SUCCESS_THRESHOLD) for r in rs)
        )
        num_pass_n_tasks = sum(
            1
            for rs in per_task_rewards
            if any((r is not None and r >= SUCCESS_THRESHOLD) for r in rs)
        )
        passk = num_pass_n_tasks / len(per_task_rewards)
        summary = (
            f"[EVAL @ step {n_global_step}] mean_reward={avg:.4f} std_reward={std_reward:.4f} "
            f"task_pass_rate@1={pass1*100:.2f}% task_pass_rate@{k}={passk*100:.2f}% "
            f"n_tasks={len(per_task_rewards)} n_rollouts={len(flat)}"
        )
        print(summary)
        with open(eval_log_path, "a") as f:
            f.write(summary + "\n")
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

    # step-0 eval disabled for faster iteration
    last_eval_step = 0
    # run_eval(0)  # skip initial eval

    executor = PeriodicDrainThreadPoolExecutor(
        workers=GRPO_N * REMOTE_BATCH_SIZE, max_parallel=64, auto_retry=True
    )

    train_log_path = os.path.join(RESULT_DIR, "train_results.log")

    n_global_step = 0
    for _ in range(NUM_EPOCH):
        for task in generate_training_tasks():
            for _ in range(GRPO_N):
                # `submit_with_periodic_drain` returns drained results only when the
                # in-flight pool was actually drained on this submission. Each drain
                # boundary corresponds to a fully-collected local batch -- exactly
                # when this client should agree to a weight sync under
                # `rollout_until_all_clients_agree_sync_weight`.
                _, drained_results = executor.submit_with_periodic_drain(
                    fn=rollout, task=task
                )
                if drained_results:
                    # Log batch rewards before weight sync
                    rewards = [r for r in drained_results if r is not None]
                    if rewards:
                        avg_reward = sum(rewards) / len(rewards)
                        std_reward = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
                        success_rate = sum(1 for r in rewards if r >= 1.0) / len(rewards)
                        step = swarm_worker.get_global_step()
                        log_line = (
                            f"[TRAIN @ step {step}] client=appworld  "
                            f"batch_size={len(rewards)}  mean_reward={avg_reward:.4f}  "
                            f"std_reward={std_reward:.4f}  success_rate={success_rate*100:.2f}%"
                        )
                        print(log_line)
                        with open(train_log_path, "a") as f:
                            f.write(log_line + "\n")
                    swarm_worker.agree_sync_weight()

            n_global_step = swarm_worker.get_global_step()
            if n_global_step >= last_eval_step + EVAL_INTERVAL:
                run_eval(n_global_step)
                last_eval_step = n_global_step

            if n_global_step >= TOTAL_TRAINING_STEPS:
                break

        if n_global_step >= TOTAL_TRAINING_STEPS:
            break

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
