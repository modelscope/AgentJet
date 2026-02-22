# -*- coding: utf-8 -*-

import os
import re
import requests
from textwrap import dedent
from ajet.schema.task import Task, WorkflowOutput
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader
from ajet.utils.retry import retry_with_backoff
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.tuner_lib.experimental.interchange_utils import SwarmThrottlePolicy
from ajet.default_config.ajet_default import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.experimental.as_swarm_client import SwarmClient, SwarmThrottlePolicy

# python -m tutorial.example_math_swarm.math

GRPO_N = 4  # grpo group size
NUM_EPOCH = 10000
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
REMOTE_MODEL_PATH = os.getenv("REMOTE_MODEL_PATH", "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-3B-Instruct")
REMOTE_BATCH_SIZE = 32
REMOTE_ALLOCATE_GPU_PER_NODE = 8

def main():

    # Handshake with swarm remote, then send training param to swarm remote (such as model to be trained, algorithm, etc)
    dataset = RouterTaskReader(
        reader_type = "huggingface_dat_repo",
        reader_config = AjetTaskReader(
            huggingface_dat_repo = HuggingfaceDatRepo(
                dataset_path = "C:/Users/fuqingxu-hub/Downloads/dataset/gsm8k/socratic",
                # dataset_path = "openai/gsm8k",
                # dataset_name = "main",
            )
        )
    )

    # # Hand shake with remote swarm server
    swarm_worker = SwarmClient(AJET_SWARM_URL)
    swarm_worker.auto_sync_train_config_and_start_engine(
        AgentJetJob(
            experiment_name="math_gsm8k_grpo",
            algorithm="grpo",
            n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
            model=REMOTE_MODEL_PATH,
            batch_size=REMOTE_BATCH_SIZE,
            num_repeat=GRPO_N,
        ),
        # force_restart=True,
    )

    def rollout(task):
        try:
            # begin episode
            episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=60)
            # execute agent ( base_url = api_baseurl_key.base_url, api_key = api_baseurl_key.api_key )
            workflow_output = execute_agent(task, api_baseurl_key)  # reward is in `workflow_output`
            # report output back to swarm remote
            swarm_worker.end_episode(task, episode_uuid, workflow_output)
            return
        except:
            pass

    executor = PeriodicDrainThreadPoolExecutor(workers=GRPO_N * REMOTE_BATCH_SIZE, auto_retry=True)
    for _ in range(NUM_EPOCH):
        for _, task in enumerate(dataset.generate_training_tasks()):
            for _ in range(GRPO_N):
                executor.submit_with_periodic_drain(fn=rollout, task=task)

    return None




def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
    # Prepare base_url, api_key
    base_url, api_key = (api_baseurl_key.base_url, api_baseurl_key.api_key)
    # Read dataset item
    query, reference_answer = (task.main_query, task.metadata["answer"])
    # Prepare messages
    messages = [
        { "role": "system", "content": dedent("""You are an agent specialized in solving math problems. Please solve the math problem given to you.
           You can write and execute Python code to perform calculation or verify your answer. You should return your final answer within \\boxed{{}}.""") },
        { "role": "user", "content": query }
    ]
    # Use raw http requests (non-streaming) to get response
    # "Connection: close" prevents keep-alive pool reuse, which can cause BadStatusLine
    # errors under high concurrency when stale pooled connections return residual bytes.
    response = requests.post(
        f"{base_url}/chat/completions",
        json    = { "model": "fill_whatever_model", "messages": messages, "stream": False },
        headers = { "Authorization": f"Bearer {api_key}", "Connection": "close" },
        timeout = 300,
    )
    response.raise_for_status()
    final_answer = response.json()['choices'][0]['message']['content']

    reference_answer = reference_answer.split("####")[-1].strip()
    pattern = r"\\boxed\{([^}]*)\}"
    match = re.search(pattern, final_answer)
    if match: is_success = match.group(1) == reference_answer
    else: is_success = False
    raw_reward = 1.0 if is_success else 0.0
    # Return
    return WorkflowOutput(reward=raw_reward, metadata={"final_answer": final_answer})




if __name__ == "__main__":
    main()
