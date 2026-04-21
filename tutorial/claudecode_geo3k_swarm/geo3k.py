# -*- coding: utf-8 -*-
"""
Geo3K multimodal agent, migrated from rllm's examples/geo3k into
AgentJet's swarm architecture.

Run:
    python -m tutorial.claudecode_geo3k_swarm.geo3k

Dataset:
    hiyouga/geometry3k (expected to be pre-downloaded or loadable via
    huggingface_hub; see readme.md for how to download with proxychains).

Model:
    A vision-language model such as Qwen/Qwen2.5-VL-7B-Instruct. Set
    REMOTE_MODEL_PATH to a local checkpoint.
"""

import os
import re
import requests
from textwrap import dedent

from ajet.schema.task import Task, WorkflowOutput
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from ajet.utils.multimodal import build_multimodal_messages, extract_image


GRPO_N = 4  # grpo group size
NUM_EPOCH = 10000
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
REMOTE_MODEL_PATH = os.getenv(
    "REMOTE_MODEL_PATH",
    "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-VL-7B-Instruct",
)
REMOTE_BATCH_SIZE = 32
REMOTE_ALLOCATE_GPU_PER_NODE = 8
# Geo3k dataset — either local parquet/dir or a HF repo id
GEO3K_DATASET_PATH = os.getenv(
    "GEO3K_DATASET_PATH",
    "/mnt/data_cpfs/model_cache/modelscope/dataset/hiyouga/geometry3k/data/train-00000-of-00001.parquet",
)

SYSTEM_PROMPT = dedent(
    """You are an agent specialized in solving geometry problems.
    Look carefully at the provided figure and reason step by step.
    Return your final answer enclosed in \\boxed{}."""
)

ANSWER_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower()


def _compute_reward(final_answer: str, reference_answer: str) -> float:
    """Extract \\boxed{} answer and compare to ground truth string."""
    m = ANSWER_RE.search(final_answer or "")
    if not m:
        return 0.0
    predicted = _normalize(m.group(1))
    target = _normalize(reference_answer)
    return 1.0 if predicted == target else 0.0


def _execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey) -> WorkflowOutput:
    base_url, api_key = api_baseurl_key.base_url, api_baseurl_key.api_key

    # Geo3k row layout (from rllm preprocess):
    #   question: str, image/images: list[PIL.Image] or dict with bytes,
    #   answer / ground_truth: str
    meta = task.metadata
    question = meta.get("question") or meta.get("problem") or task.main_query
    if "\\boxed" not in question:
        question = question + "\nLet's think step by step and output your final answer in \\boxed{}."

    reference_answer = meta.get("ground_truth") or meta.get("answer") or ""

    messages = build_multimodal_messages(
        system_prompt=SYSTEM_PROMPT,
        user_text=question,
        image=extract_image(meta),
    )

    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": "fill_whatever_model",
            "messages": messages,
            "stream": False,
            "max_tokens": 2048,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Connection": "close",
        },
        timeout=600,
    )
    response.raise_for_status()
    final_answer = response.json()["choices"][0]["message"]["content"]

    raw_reward = _compute_reward(final_answer, reference_answer)
    return WorkflowOutput(
        reward=raw_reward,
        metadata={"final_answer": final_answer, "reference": reference_answer},
    )


def run_agent_and_compute_reward(task: Task, base_url: str, api_key: str) -> WorkflowOutput:
    return _execute_agent(task, OpenaiBaseUrlAndApiKey(base_url=base_url, api_key=api_key))


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
    ajet_job = AgentJetJob(
        experiment_name="geo3k_grpo",
        algorithm="grpo",
        n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
        model=REMOTE_MODEL_PATH,
        batch_size=REMOTE_BATCH_SIZE,
        num_repeat=GRPO_N,
    )
    print(ajet_job.config.to_dict())
    swarm_worker.auto_sync_train_config_and_start_engine(
        ajet_job,
        force_restart=True,
    )

    def rollout(task):
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=240)
        workflow_output = _execute_agent(task, api_baseurl_key)
        swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return

    executor = PeriodicDrainThreadPoolExecutor(
        workers=GRPO_N * REMOTE_BATCH_SIZE, auto_retry=True
    )
    for _ in range(NUM_EPOCH):
        for _, task in enumerate(dataset.generate_training_tasks()):
            for _ in range(GRPO_N):
                executor.submit_with_periodic_drain(fn=rollout, task=task)

    return None


if __name__ == "__main__":
    main()
