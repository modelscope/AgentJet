# AgentJet Swarm Training

<div align="center">
<img width="640" alt="image" src="https://serve.gptacademic.cn/publish/shared/Image/classic+swarm+revise.jpg"/>
</div>

In previous training modes, the training base supported by VERL could be likened to an "aircraft carrier". From this "mother ship",
only lightweight "Jets" could take off as carriers for Agent operations, and all Jets were strongly tied to the "mother ship".
This means it was impossible to use models from other "mother ships" for multi-agent training with non-shared parameters, nor could keys and reward parameters fixed in environment variables and code be switched conveniently,
and they couldn't flow freely between multiple hardware platforms. Once any issue arose, the entire process had to be terminated and reverted to the previous checkpoint.

However, the AgentJet Swarm mode has pioneered a brand-new training approach. Continuing with the previous metaphor, in swarm mode,
you can freely launch multiple "mother ships" (corresponding to multiple LLM models to be trained) on one or more servers.
Then, from an "airport" (e.g., your workstation, server, or even your Mac), you can "take off" any number of "Jets" to act as "worker bees" running the Agent workflow awaiting training,
forming a many-to-many training system:

- "Jets" are responsible for reading datasets, running the Agent workflow, and finally sending reward signals back to each "mother ship".
- "Mother ships" are responsible for providing vllm/sglang API interfaces (with AgentJet’s automatic context tracking & timeline merging capabilities that significantly accelerate training), coordinating and computing samples.


# Using AgentJet Swarm to Train Your Agents

AgentJet Swarm opens infinite possibilities for both LLM Agent engineers and LLM researchers. It is very easy to use and understand. In fact, there is no need for verbose explaination, code explains itself:

## (1/2) Launching a Swarm Server ("aircraft carrier")

Simply run `ajet-swarm start` on a GPU server (or GPU cluster master), and we are done ✅. (You may ask: what about training config? Well, config will come from swarm client.)

![alt text](https://img.alicdn.com/imgextra/i4/O1CN01bm585R20h63S9NSSy_!!6000000006880-2-tps-1649-765.png)

Notes:
1. launch server together with a swarm monitor:
    ```bash
    (ajet-swarm start &> ajet-swarm-server.log)  & (ajet-swarm overwatch)
    ```

2. overwatch swarm status with url:
    ```bash
    ajet-swarm overwatch --swarm-url=http://localhost:10086
    ```

3. changing customized port (default port is 10086):
    ```bash
    ajet-swarm start --swarm-port=10086
    ```

4. if you are using a multi-node cluster to train huge models, make sure you have already set up the ray cluster before you hit `ajet-swarm start`.

## (2/2) Launching Swarm Clients ("jets")

You can run any amount of swarm client:

- on any devices (macbook, workstation, the same machine you run swarm-server, **wherever you want**).
- at any time (before or in the middle of a training, **whenever you want**)

But just remember: **ALL** swarm clients are equally authorized to order swarm server(s) **start or terminate** the training process. There is **no such role like Queen** in AgentJet Swarm.

### 2-1. Connecting to a swarm server and make it rock&roll

The primary objective of swarm client is to make sure network connection is good.
Now, create a python script and start coding:

```python
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
REMOTE_SWARM_URL = "http://localhost:10086" # Change to your swarm remote url
swarm_worker = SwarmClient(REMOTE_SWARM_URL)
```

Secondly, generate a configuration (basically VERL yaml, but slightly different), **connect** to swarm server and then tell the swarm server **which model to train**, etc. When configuration is ready, tell engine to read yaml and begin VERL training cycles with `auto_sync_train_config_and_start_engine`.

```python
LOCAL_GRPO_N = 32
yaml_job = AgentJetJob(
    experiment_name="math_gsm8k_grpo",
    algorithm="grpo",
    n_gpu=4,
    model='/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-3B-Instruct',
    batch_size=LOCAL_GRPO_N,
    num_repeat=4,
)
# hint: you can `yaml_job.dump_job_as_yaml('./config.yaml')` to take a look at the full configuration
# hint: you can `yaml_job.build_job_from_yaml('./config.yaml')` to load yaml configuration as override. (there are some configurations that must be edited from yaml)
swarm_worker.auto_sync_train_config_and_start_engine(yaml_job)
```

The swarm server can be in the following states and transition between them as follows:

- **OFFLINE**: The swarm server is started but does not load any models or perform any training. It enters this state directly after startup. Additionally, it transitions to this state upon receiving a `stop_engine` command from (any) client while in any other state.
- **BOOTING**: The swarm server enters this state upon receiving a configuration followed by an explicit `begin_engine` command. In this state, it loads model parameters, initializes FSDP, and initializes vLLM.
- **ROLLING**: The swarm server enters this state automatically after completing **BOOTING** or after finishing the **WEIGHT_SYNCING** state. This represents the sampling phase.
- **ROLLING_POST**: When the swarm server determines that the sample pool is sufficient for proceeding to the next policy gradient step, it automatically transitions to this state. While in this state, ongoing episodes can still complete normally, but no new episodes can begin.
- **WEIGHT_SYNCING**: After being in the **ROLLING_POST** state, once all computational resources and threads related to ongoing episodes are reclaimed and cleaned up, the swarm server transitions to this state. During this stage, VERL completes the current policy gradient strategy update and then returns to the **ROLLING** state, repeating the cycle.


![alt text](https://img.alicdn.com/imgextra/i1/O1CN010Bropn1TbFgJ58c3d_!!6000000002400-0-tps-2752-1536.jpg)

### 2-2. Read your dataset and create training epoch loop

```python
LOCAL_DATASET_PATH = "/mnt/data_cpfs/dataset/openai/gsm8k/main"
dataset = RouterTaskReader(
    reader_type = "huggingface_dat_repo",
    reader_config = AjetTaskReader(
        huggingface_dat_repo = HuggingfaceDatRepo(
            dataset_path = LOCAL_DATASET_PATH
        )
    )
)
for epoch in range(LOCAL_NUM_EPOCH):
    for _, task in enumerate(dataset.generate_training_tasks()):
        for _ in range(LOCAL_GRPO_N):
            print(task)
            # executor.submit(rollout, task) # this is the place to begin a agent workflow / agent loop, we come back here later
```

You may ask: when does the policy gradient & llm weight update take place?
The answer is simple: **the swarm server takes care of everything related to training, while the swarm clients do not need to worry about this process at all**.

By default, when enough amount (>=batch size) of samples (with reward) reach the swarm server, that when a llm weight update step begins.
Please run `ajet-swarm overwatch` during training, this panel displays everything about the weight update timing, transparently.
When opening this panel, you can see 3 modes which you can select from: "rollout_until_finish_enough_episodes"(only count episodes), "rollout_until_finish_enough_tasks" (+consider task group), "rollout_until_finish_enough_non_dummy_tasks" (+consider group reward)


### 2-3. Intergrate with your agent loop.

Before intergrating your agent loop, we need to explain two concepts: **Episode** and **Task Group**.

The following process is called an "**Episode**":
*<br/>
(input task) -> (agent init and run task) -> (agent complete) -> (compute reward)
<br/>*

And in comparison, we call a group of same-input episodes a **Task Group**:

*(input task 001) -> (agent init and run task) -> (agent complete) -> (compute reward 001-1)
<br/>
(input task 001) -> (agent init and run task) -> (agent complete) -> (compute reward 001-2)
<br/>
..............................
<br/>
(input task 001) -> (agent init and run task) -> (agent complete) -> (compute reward 001-N)*

![alt text](https://img.alicdn.com/imgextra/i2/O1CN0177QGjT28M2m6kPvXF_!!6000000007917-2-tps-2752-1536.png)

With these two concepts in mind, we can write our training program (yes, this is training, NOT just inference):
```python
def rollout(task):
    try:
        # begin episode
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode()
        # execute agent ( base_url = api_baseurl_key.base_url, api_key = api_baseurl_key.api_key )
        workflow_output = execute_you_agent_here(task, api_baseurl_key)  # reward is in `workflow_output`
        # report output back to swarm remote
        swarm_worker.end_episode(task, episode_uuid, workflow_output)
        return
    except:
        pass

executor = BoundedThreadPoolExecutor(max_workers=LOCAL_MAX_PARALLEL)
for epoch in range(1024):
    # loop dataset epoch
    for _, task in enumerate(dataset.generate_training_tasks()):
        # loop dataset tasks
        for _ in range(LOCAL_GRPO_N):
            # loop episode
            executor.submit(rollout, task)
swarm_worker.stop_engine()
```

Explaination:
1. each agent workflow (each episode) must be started with `swarm_worker.begin_episode`,
which take vLLM/SGLang computational resource from swarm server, and returns `api_baseurl_key`.
2. take `base_url = api_baseurl_key.base_url` and `api_key = api_baseurl_key.api_key`, run your agent, compute reward, and return `workflow_output` to wrap the computed reward.
3. call `end_episode` to report reward to swarm server. (or alternatively call `abort_episode` to give up this episode, hoping to rollout something else in next try)
4. the whole `swarm_worker` thing is thread safe, go threading any way as you wish.
5. you can kill this script in the middle from the training, move it to another computer(s), and resume training without losing any progress.
6. you can **debug during training**, this is a crazy but very useful feature. Just change `end_episode` to `abort_episode` and remove `stop_engine`, and then you can copy the modified script to wherevery you want, and debug however you wish.
