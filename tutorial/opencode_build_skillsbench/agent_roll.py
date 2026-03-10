#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgentJet training script for SkillsBench with OpenCode agent.
"""

import os
import sys
from pathlib import Path
from ajet.schema.task import Task, WorkflowOutput
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.default_config.ajet_default import AjetTaskReader, JsonlDatasetFile, JsonlTrainingFp
from ajet.tuner_lib.experimental.as_swarm_client import SwarmClient
from tutorial.opencode_build_skillsbench.get_training_dataset_item_list import get_training_dataset_item_list
from tutorial.opencode_build_skillsbench.run_episode import run_episode
# tutorial/opencode_build_skillsbench


# Training configuration
NUM_EPOCH = 10000
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
REMOTE_MODEL_PATH = os.getenv("REMOTE_MODEL_PATH", "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2__5-14B-Instruct")


class SkillsBenchTaskReader:
    """Custom task reader for SkillsBench dataset."""

    def __init__(self):
        self.tasks = get_training_dataset_item_list()
        print(f"Loaded {len(self.tasks)} SkillsBench tasks")

    def generate_training_tasks(self):
        """Generate training tasks in AgentJet format."""
        for task_data in self.tasks:
            # Create a Task object for each SkillsBench task
            task = Task(
                task_id=task_data["task_id"],
                metadata={
                    "task_id": task_data["task_id"],
                    "task_path": task_data["task_path"],
                }
            )
            yield task


def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey) -> WorkflowOutput:
    """
    Execute the OpenCode agent on a SkillsBench task.

    Args:
        task: AgentJet Task object containing task_id and task_path
        api_baseurl_key: API credentials from swarm server

    Returns:
        WorkflowOutput with reward and metadata
    """

    task_id = task.metadata["task_id"]
    task_path = task.metadata["task_path"]

    # Use run_episode from run_episode.py
    # The model parameter is ignored in run_episode (hardcoded model is used)
    model = "placeholder-model"

    try:
        # Call the imported run_episode function
        reward, metadata = run_episode(
            task_id=task_id,
            task_path=task_path,
            api_key=api_baseurl_key.api_key,
            base_url=api_baseurl_key.base_url,
            model=model,
        )

        return WorkflowOutput(reward=float(reward), metadata=metadata)

    except Exception as e:
        print(f"ERROR: Exception during task execution: {e}\n")
        metadata = {
            "task_id": task_id,
            "task_path": task_path,
            "success": False,
            "error": str(e),
        }
        return WorkflowOutput(reward=0.0, metadata=metadata)


def main():

    # Create custom task reader
    dataset = SkillsBenchTaskReader()

    ajet_job = AgentJetJob(
        base_yaml_config="tutorial/opencode_build_skillsbench/skillbench.yaml",
        algorithm="grpo",
        experiment_name="skillbench_swarm",
        max_env_worker=128,
    )

    # Hand shake with remote swarm server
    swarm_worker = SwarmClient(AJET_SWARM_URL)
    swarm_worker.auto_sync_train_config_and_start_engine(
        ajet_job,
        # force_restart=True,
    )

    GRPO_N = ajet_job.num_repeat
    REMOTE_BATCH_SIZE = ajet_job.batch_size

    def rollout(task):
        # begin episode
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=240)
        # execute agent ( base_url = api_baseurl_key.base_url, api_key = api_baseurl_key.api_key )
        workflow_output = execute_agent(task, api_baseurl_key)  # reward is in `workflow_output`
        # report output back to swarm remote
        swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return


    executor = PeriodicDrainThreadPoolExecutor(workers=GRPO_N*REMOTE_BATCH_SIZE, max_parallel=4, auto_retry=True, block_first_run=False)
    for _ in range(NUM_EPOCH):
        for _, task in enumerate(dataset.generate_training_tasks()):
            for _ in range(GRPO_N):
                executor.submit_with_periodic_drain(fn=rollout, task=task)

    return



if __name__ == "__main__":
    main()
