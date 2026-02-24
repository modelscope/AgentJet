

## 解耦训练器和执行器



## 推理即训练

再也不需要给推理和训练写两套程序了，
在下面的蜂群worker代码中，你可以随意编排怎么读取和使用数据集，
想如何并行执行Agent都悉听尊便。
唯一要注意的是：需要在Agent开始前后，调用 `swarm_remote.begin_episode` 和 `swarm_remote.end_episode` 和蜂群server进行联络。


```python
# 读取数据集
dataset = RouterTaskReader(
    reader_type = "huggingface_dat_repo",
    reader_config = AjetTaskReader(
        huggingface_dat_repo = HuggingfaceDatRepo(
            dataset_path = LOCAL_DATASET_PATH
        )
    )
)

# 连接到蜂群
swarm_worker = SwarmClient("http://localhost:10086")
# 注意：如果蜂群中有多个worker，其中一个蜂群worker需要调用 swarm_remote.auto_sync_train_config_and_start_engine
# 告知蜂群server需要训练哪个模型（以及各种训练参数）
swarm_worker.auto_sync_train_config_and_start_engine(
    AgentJetJob(
        algorithm="grpo",
        n_gpu=REMOTE_ALLOCATE_GPU_PER_NODE,
        model=REMOTE_TRAIN_MODEL_01,
        batch_size=REMOTE_BATCH_SIZE,
        num_repeat=LOCAL_GRPO_N,
    ),
)

# 单个任务的主逻辑
def rollout(task) -> float | None:
    # 向蜂群索要运行一个episode的许可
    episode_uuid, api_baseurl_key = swarm_worker.begin_episode()
    # 使用获得的api key和base url运行智能体 ( base_url = api_baseurl_key.base_url, api_key = api_baseurl_key.api_key )
    workflow_output = execute_agent(task, api_baseurl_key)  # reward is in `workflow_output`
    # 将episode的运行结果（奖励）汇报给蜂群
    swarm_worker.end_episode(task, episode_uuid, workflow_output)
    # 打印蜂群、蜂群其他worker的任务汇总（例如距离下次LLM梯度更新还差多少样本）
    swarm_worker.print_rollout_stat()
    return workflow_output.reward

# 并行执行 Agent 完成多个 task，逐步地完成几个epoch
episodes = []
for epoch in range(10):
    for i_task, task in enumerate(dataset.generate_training_tasks()):
        for j_repeat in range(LOCAL_GRPO_N):
            episodes += [ task ]
            # wait until getting `local_batching_size` episodes, then execute them with retry logic
            if len(episodes) == (REMOTE_BATCH_SIZE * LOCAL_GRPO_N):
                episode_results = run_episodes_until_all_complete(episodes, func=rollout, auto_retry=True)
                for episode, reward in zip(episodes, episode_results):
                    logger.info(f"Episode for task {episode.task_id} completed with reward: {reward}")
                episodes = []

# ... 等待训练结束 ...
```
