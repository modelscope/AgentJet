# -*- coding: utf-8 -*-
"""
Timeline-merging study driver.

Forks agent_roll.py with three differences:
  1. timeline_compare_level configurable via env TIMELINE_COMPARE_LEVEL (token|text)
  2. model path / total_training_steps overridable via env
  3. per-step training metrics dumped to a markdown file we can later parse
     for update_actor timing comparison.

python -m tutorial.example_appworld_swarm.agent_roll_timeline_study
"""

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
MAX_STEPS = int(os.getenv("APPWORLD_MAX_STEPS", "50"))

EVAL_INTERVAL = int(os.getenv("APPWORLD_EVAL_INTERVAL", "9999"))  # disable mid-run eval for timing study
EVAL_K = int(os.getenv("APPWORLD_EVAL_K", "1"))
TOTAL_TRAINING_STEPS = int(os.getenv("APPWORLD_TOTAL_TRAINING_STEPS", "50"))
RESULT_DIR = os.getenv("APPWORLD_RESULT_DIR", "./appworld_swarm_results")
MAX_ENV_WORKER = int(os.getenv("APPWORLD_MAX_ENV_WORKER", "64"))

# --- study-specific knobs ------------------------------------------------
TIMELINE_COMPARE_LEVEL = os.getenv("TIMELINE_COMPARE_LEVEL", "token")
MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/data_cpfs/xielipeng.xlp/models/Qwen3-8B")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", f"appworld_swarm_tlstudy2_qwen3orignal_{TIMELINE_COMPARE_LEVEL}")
N_GPU = int(os.getenv("N_GPU", "8"))
TRAIN_METRICS_MD = os.getenv("TRAIN_METRICS_MD", os.path.join(RESULT_DIR, "train_metrics.md"))


def get_appworld_tasks(split: str) -> List[Task]:
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
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TRAIN_METRICS_MD) or ".", exist_ok=True)

    ajet_job = AgentJetJob(
        base_yaml_config="tutorial/example_appworld_swarm/appworld.yaml",
        algorithm="grpo",
        experiment_name=EXPERIMENT_NAME,
        max_env_worker=MAX_ENV_WORKER,
        model=MODEL_PATH,
        n_gpu=N_GPU,
        total_training_steps=TOTAL_TRAINING_STEPS,
        timeline_compare_level=TIMELINE_COMPARE_LEVEL,
        train_print_to_markdown_file_path=TRAIN_METRICS_MD,
    )

    swarm_worker = SwarmClient(AJET_SWARM_URL, agentjet_job=ajet_job)
    swarm_worker.auto_sync_train_config_and_start_engine(ajet_job)

    GRPO_N = ajet_job.num_repeat
    REMOTE_BATCH_SIZE = ajet_job.batch_size

    print(f"[STUDY] timeline_compare_level={TIMELINE_COMPARE_LEVEL}")
    print(f"[STUDY] model={MODEL_PATH}")
    print(f"[STUDY] total_training_steps={TOTAL_TRAINING_STEPS}")
    print(f"[STUDY] train_metrics_md={TRAIN_METRICS_MD}")

    def rollout(task: Task) -> float:
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=600)
        workflow_output = execute_agent(task, api_baseurl_key)
        swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return workflow_output.reward

    executor = PeriodicDrainThreadPoolExecutor(
        workers=GRPO_N * REMOTE_BATCH_SIZE, max_parallel=64, auto_retry=True
    )

    n_global_step = 0
    for _ in range(NUM_EPOCH):
        for task in generate_training_tasks():
            for _ in range(GRPO_N):
                _, drained_results = executor.submit_with_periodic_drain(
                    fn=rollout, task=task
                )
                if drained_results:
                    swarm_worker.agree_sync_weight()

            n_global_step = swarm_worker.get_global_step()
            if n_global_step >= TOTAL_TRAINING_STEPS:
                break

        if n_global_step >= TOTAL_TRAINING_STEPS:
            break

    print(f"[STUDY] Training complete at global_step={n_global_step}.")


if __name__ == "__main__":
    main()
