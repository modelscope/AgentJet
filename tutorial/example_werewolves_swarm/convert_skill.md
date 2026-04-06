训练复杂智能体的时候，推荐先从yaml配置出发

首先，复制一份基础配置 ajet/default_config/ajet_swarm_default.yaml

cp  ajet/default_config/ajet_swarm_default.yaml  tutorial/example_werewolves_swarm/werewolves.yaml

然后对配置中的参数进行修改：

---- opencode命令：这里补充一个参数配置说明表格，参考tutorial/example_werewolves_swarm/werewolves.yaml ----


# 编写训练循环 (Swarm Client)

Swarm Client 流程如下：

- 连接蜂群
- 蜂群server初始化
- 开始EPOCH循环


```python
# -*- coding: utf-8 -*-

NUM_EPOCH = 10000
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")

def main():

    ajet_job = AgentJetJob(
        base_yaml_config="tutorial/example_werewolves_swarm/werewolves.yaml",
        algorithm="grpo",
        experiment_name="werewolves_swarm",
    )

    # Hand shake with remote swarm server
    swarm_worker = SwarmClient(AJET_SWARM_URL)
    swarm_worker.auto_sync_train_config_and_start_engine( ajet_job, force_restart=True )

    GRPO_N = ajet_job.num_repeat
    REMOTE_BATCH_SIZE = ajet_job.batch_size

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

    # Handshake with swarm remote, then send training param to swarm remote (such as model to be trained, algorithm, etc)
    dataset = RouterTaskReader(
        reader_type = "random_dummy",
        reader_config = AjetTaskReader()
    )
    executor = PeriodicDrainThreadPoolExecutor(workers=GRPO_N * REMOTE_BATCH_SIZE, auto_retry=True)
    for _ in range(NUM_EPOCH):
        for _, task in enumerate(dataset.generate_training_tasks()):
            for _ in range(GRPO_N):
                executor.submit_with_periodic_drain(fn=rollout, task=task)

    return None


def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
    raise NotImplementedError("see below.")


if __name__ == "__main__":
    main()

```

# 编写Agent (Swarm Client)
