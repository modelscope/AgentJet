from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient, run_episodes_until_all_complete
from ajet.default_config.ajet_config_schema import AjetTaskReader
from ajet.task_reader import RouterTaskReader
from .frozenlake import FrozenLake

import asyncio
import threading

# step 1: ajet-swarm start --swarm-port=10086
# step 2: ajet-swarm start --swarm-port=10087
# step 3: python -m tutorial.example_frozenlake_swarm.frozen_lake_roll

# --------- configurations that take effect locally -------------
LOCAL_GRPO_N = 4  # grpo group size
LOCAL_NUM_EPOCH = 10000

# --------- configurations that take effect remotely -------------
REMOTE_BATCH_SIZE = 32
REMOTE_1_SWARM_URL = "http://localhost:10086" # Change to your swarm remote url
REMOTE_1_ALLOCATE_GPU_PER_NODE = 4
REMOTE_1_TRAIN_MODEL = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'


class WeightUpdatedHalfway(Exception):
    """Raised when the remote side starts updating model weights halfway through an episode."""


def main():

    dataset = RouterTaskReader(reader_type = "random_dummy", reader_config = AjetTaskReader())

    # Hand shake with remote swarm server
    swarm_worker_7B = SwarmClient(REMOTE_1_SWARM_URL)
    swarm_worker_7B.auto_sync_train_config_and_start_engine(
        AgentJetJob(
            algorithm="grpo",
            project_name="ajet-swarm",
            experiment_name="test",
            n_gpu=REMOTE_1_ALLOCATE_GPU_PER_NODE,
            model=REMOTE_1_TRAIN_MODEL,
            batch_size=REMOTE_BATCH_SIZE,
            num_repeat=LOCAL_GRPO_N,
        ),
    )

    def play_different_swarm_server(task, swarm_worker:SwarmClient) -> float | None:
        # begin episode
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=120)
        # execute agent ( base_url = api_baseurl_key.base_url, api_key = api_baseurl_key.api_key )
        env = FrozenLake(
            env_max_steps=20,
            agent_max_steps=20,
            seed=task.metadata["random_number"],
        )
        workflow_output = asyncio.run(env.execute(task, api_baseurl_key.api_key, api_baseurl_key.base_url))
        # report output back to swarm remote
        swarm_worker.end_episode(task, episode_uuid, workflow_output)
        # print global rollout status across the swarm
        swarm_worker.print_rollout_stat()
        return workflow_output.reward

    def rollout(task):
        f1 = threading.Thread(target=play_different_swarm_server, args=(task, swarm_worker_7B), daemon=True)
        f1.start()
        f1.join()
        return


    next_batch = []
    for epoch in range(LOCAL_NUM_EPOCH):
        for _, task in enumerate(dataset.generate_training_tasks()):
            for _ in range(LOCAL_GRPO_N):
                next_batch.append(task)
                if len(next_batch) >= (REMOTE_BATCH_SIZE * LOCAL_GRPO_N):
                    # wait until getting `local_batching_size` next_batch, then execute them with with retry logic
                    run_episodes_until_all_complete(next_batch, func=rollout, auto_retry=True)
                    next_batch.clear()
    return None


if __name__ == "__main__":
    main()
