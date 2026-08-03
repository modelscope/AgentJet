# -*- coding: utf-8 -*-
"""
Geo3K multimodal swarm client, migrated from rllm's examples/geo3k into
AgentJet's swarm architecture.

Run:
    python -m tutorial.claudecode_geo3k_swarm.geo3k_swarm_client

Dataset:
    hiyouga/geometry3k (expected to be pre-downloaded or loadable via
    huggingface_hub; see readme.md for how to download with proxychains).

Model:
    A vision-language model such as Qwen/Qwen2.5-VL-7B-Instruct. Set
    REMOTE_MODEL_PATH to a local checkpoint.

The per-episode agent logic lives in ``geo3k_agent._execute_agent`` and
can be tested standalone against DashScope or a vLLM endpoint; see
``geo3k_agent.py``.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ajet.schema.task import Task
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader, HuggingFaceTaskReader
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.experimental.swarm_client import SwarmClient

from tutorial.claudecode_geo3k_swarm.geo3k_agent import _execute_agent


GRPO_N = 4  # grpo group size

NUM_EPOCH = 10000
EVAL_INTERVAL = 100  # Evaluate every EVAL_INTERVAL * REMOTE_BATCH_SIZE training tasks
EVAL_K = 2  # pass@k: run each eval task K times
MAX_ENV_WORKER = 64  # also used as eval threadpool size
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
REMOTE_MODEL_PATH = os.getenv(
    "REMOTE_MODEL_PATH",
    "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen3___6-35B-A3B",
)
REMOTE_BATCH_SIZE = 32
REMOTE_ALLOCATE_GPU_PER_NODE = 8
REMOTE_NNODES = 2
# Qwen3.6-35B-A3B has 2 KV heads, so TP must divide 2 (1 or 2). TP=2 for the
# 35B MoE VL model on 97GB H20 GPUs.
REMOTE_TENSOR_PARALLEL_SIZE = 8
# Geo3k dataset — either local parquet/dir or a HF repo id
GEO3K_DATASET_PATH = os.getenv(
    "GEO3K_DATASET_PATH",
    "/mnt/data_cpfs/model_cache/modelscope/dataset/hiyouga/geometry3k/data/train-00000-of-00001.parquet",
)
# Geo3k test/validation parquet — used for periodic eval, mirrors rllm's test split.
GEO3K_TEST_DATASET_PATH = os.getenv(
    "GEO3K_TEST_DATASET_PATH",
    "/mnt/data_cpfs/model_cache/modelscope/dataset/hiyouga/geometry3k/data/test-00000-of-00001.parquet",
)

# vLLM sleep-mode resume (wake_up) re-maps the KV cache + weights on top of the
# colocated FSDP training state. At 0.85 the wake_up allocation OOMs
# (cumem_allocator.cpp:163) because the actor params/optimizer are still
# resident during weight sync. 0.5 leaves headroom for the 35B MoE actor.
REMOTE_GPU_MEMORY_UTILIZATION = float(os.getenv("REMOTE_GPU_MEMORY_UTILIZATION", "0.8"))

ajet_job = AgentJetJob(
    ensure_new_experiment=True,
    experiment_name="geo3k_grpo_z1",
    algorithm="grpo",
    logging="swanlab",
    n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
    nnodes=REMOTE_NNODES,
    model=REMOTE_MODEL_PATH,
    batch_size=REMOTE_BATCH_SIZE,
    num_repeat=GRPO_N,
    max_env_worker=MAX_ENV_WORKER,
    tensor_model_parallel_size=REMOTE_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=REMOTE_GPU_MEMORY_UTILIZATION,
    # Length params must all be set together (job.py enforces all-None or
    # all-non-None). Only max_response_length_in_one_turn changes here:
    # 4096 -> 1024 to test whether shorter per-turn decode stays under the
    # interchange server's 300s ZMQ receive budget (~12 tok/s -> ~85s worst case).
    max_prompt_length=3000,
    max_response_length=15000,
    max_model_len=18000,
    max_response_length_in_one_turn=3000,
)


def _load_eval_tasks(test_dataset_path: str, label: str = "") -> list:
    """Load a single eval parquet for periodic evaluation."""
    if not os.path.exists(test_dataset_path):
        print(f"[WARN] Eval dataset not found: {test_dataset_path}. Skipping {label or test_dataset_path}.")
        return []
    eval_reader = HuggingFaceTaskReader(
        AjetTaskReader(huggingface_dat_repo=HuggingfaceDatRepo(dataset_path=test_dataset_path))
    )
    eval_tasks = list(eval_reader.generate_training_tasks())
    print(f"[INFO] Loaded {len(eval_tasks)} eval tasks from {label or test_dataset_path}")
    return eval_tasks


def _run_eval(swarm_worker: SwarmClient, eval_tasks_by_set: dict, n_global_step: int):
    """Run pass@k evaluation on every loaded eval set; rewards do not feed training."""
    if not eval_tasks_by_set:
        return
    eval_log_path = os.path.join(swarm_worker.server_experiment_dir(), "eval_results.log")
    print(eval_log_path)
    for label, eval_tasks in eval_tasks_by_set.items():
        _run_eval_one(swarm_worker, label, eval_tasks, n_global_step, eval_log_path)


def _run_eval_one(
    swarm_worker: SwarmClient,
    label: str,
    eval_tasks: list,
    n_global_step: int,
    eval_log_path: str,
):
    """Run pass@k evaluation on a single eval set."""
    if not eval_tasks:
        return

    def eval_rollout(task: Task) -> float:
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(
            discard_episode_timeout=240, episode_type="eval"
        )
        try:
            workflow_output = _execute_agent(task, api_baseurl_key)
            return workflow_output.reward
        finally:
            swarm_worker.abort_episode(episode_uuid)

    k = EVAL_K
    total = len(eval_tasks) * k
    print(f"\n[EVAL @ step {n_global_step}] Running {label} eval on {len(eval_tasks)} tasks x {k} (pass@{k})...")
    per_task_rewards = [[] for _ in eval_tasks]
    pbar = tqdm(total=total, desc=f"EVAL {label} @ step {n_global_step}")

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
    if flat:
        avg = sum(flat) / len(flat)
        pass1 = sum(1 for r in flat if r > 0) / len(flat)
        solved = [rs for rs in per_task_rewards if any((r is not None and r > 0) for r in rs)]
        passk = len(solved) / len(per_task_rewards)
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


def main():
    assert AJET_SWARM_URL != "http://swarm-server-ip:10086", (
        "Please set AJET_SWARM_URL to your swarm server's URL, "
        "e.g., http://localhost:10086"
    )

    dataset = RouterTaskReader(
        reader_type="huggingface_dat_repo",
        reader_config=AjetTaskReader(
            huggingface_dat_repo=HuggingfaceDatRepo(
                dataset_path=GEO3K_DATASET_PATH,
            )
        ),
    )

    swarm_worker = SwarmClient(AJET_SWARM_URL)

    swarm_worker.auto_sync_train_config_and_start_engine(
        ajet_job,
        force_restart=os.environ.get("FORCE_RESTART_SWARM_ENGINE", "0") == "1"
    )

    test_datasets = {
        "Geo3K-test": GEO3K_TEST_DATASET_PATH,
    }
    eval_tasks_by_set = {}
    for label, path in test_datasets.items():
        tasks = _load_eval_tasks(path, label=label)
        if tasks:
            eval_tasks_by_set[label] = tasks

    # Baseline eval before any training updates. Gated behind RUN_BASELINE_EVAL
    # (default off): the step-0 baseline over the full 1202-task test set takes
    # very long under the current per-episode routing throughput and delays the
    # start of training. Set RUN_BASELINE_EVAL=1 to re-enable it.
    if os.environ.get("RUN_BASELINE_EVAL", "0") == "1":
        _run_eval(swarm_worker, eval_tasks_by_set, 0)

    def rollout(task):
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=240)
        workflow_output = _execute_agent(task, api_baseurl_key)
        swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return

    executor = PeriodicDrainThreadPoolExecutor(
        workers=GRPO_N * REMOTE_BATCH_SIZE, auto_retry=True
    )
    task_count = 0
    for _ in range(NUM_EPOCH):
        for _, task in enumerate(dataset.generate_training_tasks()):
            for _ in range(GRPO_N):
                executor.submit_with_periodic_drain(fn=rollout, task=task)

            task_count += 1
            if task_count % (EVAL_INTERVAL * REMOTE_BATCH_SIZE) == 0:
                n_global_step = task_count // REMOTE_BATCH_SIZE
                _run_eval(swarm_worker, eval_tasks_by_set, n_global_step)

    return None


if __name__ == "__main__":
    main()
