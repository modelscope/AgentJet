import re
import threading
import requests
import time
from loguru import logger
from textwrap import dedent
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.weight_tuner.experimental.as_swarm_client import SwarmClient
from ajet.default_config.ajet_default import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.weight_tuner.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.utils.thread_executors import BoundedThreadPoolExecutor
from ajet.schema.task import Task
from ajet.task_reader import RouterTaskReader
from ajet.utils.retry import retry_with_backoff
from concurrent.futures import ThreadPoolExecutor
from tutorial.example_academic_trans.trans import execute_agent

# python -m tutorial.example_academic_trans.trans_roll


# --------- configurations that take effect locally -------------
LOCAL_GRPO_N = 4  # grpo group size
LOCAL_NUM_EPOCH = 10000
LOCAL_NUM_EPOCH = 1
LOCAL_MAX_PARALLEL = 32
LOCAL_DATASET_PATH = "/mnt/data_cpfs/qingxu.fu/agentjet/agentjet/tmp/arxiv_papers/train.parquet"
REMOTE_SWARM_URL = "http://localhost:10086" # Change to your swarm remote url

# --------- configurations that take effect remotely -------------
REMOTE_BATCH_SIZE = 32
REMOTE_ALLOCATE_GPU_PER_NODE = 8
REMOTE_TRAIN_MODEL_01 = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'

class WeightUpdatedHalfway(Exception):
    """Raised when the remote side starts updating model weights halfway through an episode."""


def main():

    # Handshake with swarm remote, then send training param to swarm remote (such as model to be trained, algorithm, etc)
    dataset = RouterTaskReader(
        reader_type = "huggingface_dat_repo",
        reader_config = AjetTaskReader(
            huggingface_dat_repo = HuggingfaceDatRepo(
                dataset_path = LOCAL_DATASET_PATH
            )
        )
    )

    # Hand shake with remote swarm server
    swarm_remote = SwarmClient(REMOTE_SWARM_URL)
    swarm_remote.auto_sync_train_config_and_start_engine(
        AgentJetJob(
            algorithm="grpo",
            n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
            model=REMOTE_TRAIN_MODEL_01,
            batch_size=REMOTE_BATCH_SIZE,
            grpo_n=LOCAL_GRPO_N,
        ),
    )

    def rollout(task):
        group_reward = []
        try:
            for _ in range(LOCAL_GRPO_N):
                try:
                    # begin episode
                    episode_uuid, api_baseurl_key = swarm_remote.begin_episode()
                    # execute agent
                    workflow_output = execute_agent(task, api_baseurl_key)
                    # report output back to swarm remote
                    swarm_remote.end_episode(task, episode_uuid, workflow_output)
                    # collect reward
                    group_reward.append(workflow_output.reward)
                except Exception as e:
                    logger.exception("Exception during rollout:", e)

            print(f"Group reward mean & std: {sum(group_reward)/len(group_reward)} +/- { (max(group_reward)-min(group_reward))/2 }")
        except Exception as e:
            logger.exception("Exception during rollout group", e)

    task_batch = []
    executor = BoundedThreadPoolExecutor(max_workers=LOCAL_MAX_PARALLEL, max_queue_size=LOCAL_MAX_PARALLEL*2)
    for i, task in enumerate(dataset.generate_training_tasks()):
        task_batch += [task]

        if len(task_batch) == REMOTE_BATCH_SIZE:
            print('*********** beginning a new batch of tasks... ***********')
            for task in task_batch:
                executor.submit(rollout, task)
            task_batch = []

    executor.shutdown(wait=True)
    return None




if __name__ == "__main__":
    main()
