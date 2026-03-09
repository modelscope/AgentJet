# -*- coding: utf-8 -*-

import os
from ajet.schema.task import Task, WorkflowTask
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.default_config.ajet_default import AjetTaskReader
from ajet.tuner_lib.experimental.swarm_client import SwarmClient

NUM_EPOCH = 10000
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")

def main():

    # Handshake with swarm remote, then send training param to swarm remote (such as model to be trained, algorithm, etc)
    dataset = RouterTaskReader(
        reader_type = "random_dummy",
        reader_config = AjetTaskReader()
    )

    ajet_job = AgentJetJob(
        base_yaml_config="tutorial/example_werewolves_swarm/werewolves.yaml",
        algorithm="grpo",
        experiment_name="werewolves_swarm",
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


    executor = PeriodicDrainThreadPoolExecutor(workers=GRPO_N * REMOTE_BATCH_SIZE, max_parallel=64, auto_retry=True)
    for _ in range(NUM_EPOCH):
        for _, task in enumerate(dataset.generate_training_tasks()):
            for _ in range(GRPO_N):
                executor.submit_with_periodic_drain(fn=rollout, task=task)

    return


def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
    import asyncio
    from tutorial.example_werewolves.start import ExampleWerewolves
    game = ExampleWerewolves(
        trainable_targets=["werewolf"],
        big_external_opponent_llm_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
        big_external_opponent_llm_url="http://22.14.116.243:2888/v1",
    )
    res = asyncio.run(game.execute(WorkflowTask(task=task), api_baseurl_key))
    return res


if __name__ == "__main__":
    main()
