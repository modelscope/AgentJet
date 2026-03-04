---
name: train-complex-blackbox
description: Create a trainable agent loop or agent workflow with AgentJet
license: Complete terms in LICENSE.txt
---


## 0. Ask user for API key + model (or API key + base url + model) for debugging

This is not 100% necessary, but it can help a lot in debugging in step 1.
If user has not given a API, ask user to give your one.


By default, the code you write should be located at ./tutorial/opencode_build_xxxxxx/*.py

## 1. Initial Programming

### Writing dataset collector (`get_training_dataset_item_list.py`)
- `get_training_dataset_item_list.py`: Returns a list of training data items. Maybe a list of training tasks, each item is a string identifier of a training task, or a dict containing necessary information for the training task.

### Episode Runner (`run_episode_once.py`)
- `run_episode_once.py`:

  - Argument Parser: takes (training data item identifier + api-key + base-url) as input, model-name is not required, you can make up a model name because we ignore it.

  - Execute the agent: read the document of the agent user asked you to train, figure out how to execute the agent. In most cases you can use subprocess to start a commandline process to execute the agent, your biggest issue is to figure out how to pass the training data item identifier, api-key and base-url to that commandline process. You can also use python code to execute the agent if you think it's more convenient.

  - Reward: extract / compute the reward/score for the agent's output. Some agents have clear reward sigal, but others don't.
    - clear reward signal: take that down as the reward, no need to do extra reward engineering.
    - no clear reward signal: you need to design a reward function to compute the reward/score for the agent's output. You can use another LLM to help you design the reward function, or you can design it by yourself if you have domain knowledge.


### Test

Remember to test these two parts before moving to step 2, make sure they work as expected.



## 2. Writing training code

This part is easy, simply follow this template and change the necessary part such as dataset path, model name, etc.

`agent_roll.py`

```python
# -*- coding: utf-8 -*-

import os
import re
import requests
from textwrap import dedent
from ajet.schema.task import Task, WorkflowOutput
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.default_config.ajet_default import AjetTaskReader, HuggingfaceDatRepo
from ajet.tuner_lib.experimental.as_swarm_client import SwarmClient

# python -m tutorial.example_math_swarm.math

GRPO_N = 4  # grpo group size
NUM_EPOCH = 10000
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
REMOTE_MODEL_PATH = os.getenv("REMOTE_MODEL_PATH", "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct")
REMOTE_BATCH_SIZE = 32
REMOTE_ALLOCATE_GPU_PER_NODE = 8

def main():

    # Handshake with swarm remote, then send training param to swarm remote (such as model to be trained, algorithm, etc)
    dataset = RouterTaskReader(
        reader_type = "huggingface_dat_repo",
        reader_config = AjetTaskReader(
            huggingface_dat_repo = HuggingfaceDatRepo(
                dataset_path = '/mnt/data_cpfs/model_cache/modelscope/dataset/openai/gsm8k/main',
                # dataset_path = "/root/agentjet/benchmark_datasets/dataset/gsm8k/socratic",
                # dataset_path = "openai/gsm8k",
                # dataset_name = "main",
            )
        )
    )
    # Load the CountDown dataset
    # print(f"Loading dataset from: {LOCAL_DATASET_PATH}")
    # dataset = RouterTaskReader(
    #     reader_type="jsonl_dataset_file",
    #     reader_config=AjetTaskReader(
    #         jsonl_dataset_file=JsonlDatasetFile(
    #             training=JsonlTrainingFp(file_path=LOCAL_DATASET_PATH)
    #         )
    #     ),
    # )

    # Hand shake with remote swarm server
    swarm_worker = SwarmClient(AJET_SWARM_URL)
    ajet_job = AgentJetJob(
        experiment_name="math_gsm8k_grpo",
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
        # begin episode
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=60)
        # execute agent ( base_url = api_baseurl_key.base_url, api_key = api_baseurl_key.api_key )
        workflow_output = execute_agent(task, api_baseurl_key)  # reward is in `workflow_output`
        # report output back to swarm remote
        swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return

    executor = PeriodicDrainThreadPoolExecutor(workers=GRPO_N * REMOTE_BATCH_SIZE, auto_retry=True)
    for _ in range(NUM_EPOCH):
        for _, task in enumerate(dataset.generate_training_tasks()):
            for _ in range(GRPO_N):
                executor.submit_with_periodic_drain(fn=rollout, task=task)

    return None


def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
    ....
    raw_reward: float = ...  # compute the reward for the agent's output
    return WorkflowOutput(reward=raw_reward, metadata={"important_metadata": important_metadata})


if __name__ == "__main__":
    main()


```


It is very clear now, your job in step 2 is to:

- use `get_training_dataset_item_list.py` to generate `List[Task]` (`from ajet.schema.task import Task`)
- use `run_episode_once.py` to execute a single episode and place it in `execute_agent` function


## 3. Simplify your code and fix bugs

before moving to step 4, you can simplify your code and fix bugs to make sure it can run smoothly.


## 4. Training

Finally, you can start training.

Run `ajet-swarm start` to start training server (if the user has already installed agentjet swarm environment),
if the user has docker environment, you can also refer to `docs/en/ajet-swarm-docker.md` to start a AgentSwarm docker container.

Create a duplication of `agent_roll.py` named `agent_roll_one_episode_debug.py`, and modify it to only run one episode, this can help you debug whether the episode runner and reward function work as expected.

After the server side is ready, run
```bash
python /path/to/agent_roll_one_episode_debug.py
```
watch console log to see if the episode can be executed successfully and reward can be computed correctly.

If anything goes wrong, keep server running, rewrite and fix `agent_roll_one_episode_debug.py`, and run it again until it can run one episode successfully.

Next, patch `agent_roll.py` if there are any bugs discorvered via the debugging of `agent_roll_one_episode_debug.py`, and then run
```bash
python /path/to/agent_roll.py
```

to start the training!
