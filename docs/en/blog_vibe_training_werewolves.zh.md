# Vibe Training：用 AgentJet 蜂群模式训练狼人杀 Agent，Prompt 一下就行

> English version: [Vibe Training Werewolves (EN)](https://modelscope.github.io/AgentJet/en/blog_vibe_training_werewolves/)

## 摘要

AgentJet 蜂群模式太爽了，想训练啥直接 Prompt AI 就行了，我称之为 **Vibe Training**。本文展示如何用 AgentJet Swarm 搭建一套完整的狼人杀训练流水线——整个训练代码只有 **70 行**。我们配置了一个异构多智能体系统：可训练的狼人和预言家共享一个 7B 模型，而猎人、女巫和平民由静态的 Qwen3-235B-A22B 驱动。最终结果？平稳上升的训练曲线——即使训练途中 CPFS 硬盘被挤满炸了两次。

## 游戏配置：异构多智能体系统

狼人杀是经典的 POMDP（部分可观测马尔可夫决策过程）问题。在本次实验中，我们配置了一个 9 人游戏，分为 **两个层次的智能**：

| 角色 | 数量 | 模型 | 模式 |
|------|------|------|------|
| 狼人 | 3 | Qwen2.5-7B | **可训练**（共享参数） |
| 预言家 | 1 | Qwen2.5-7B | **可训练**（共享参数） |
| 猎人 | 1 | Qwen3-235B-A22B | 静态 |
| 女巫 | 1 | Qwen3-235B-A22B | 静态 |
| 平民 | 3 | Qwen3-235B-A22B | 静态 |

这个配置很有意思：可训练的智能体（狼人 + 预言家）共享一个轻量级的 7B 模型，与 Qwen3-235B-A22B 驱动的强大对手对弈。共享参数的设计意味着 7B 模型必须从同一组权重中学习**角色特定的行为**——狼人需要学会欺骗和协调配合，而预言家需要学会调查和有效传达信息。

## 70 行训练代码

最精彩的部分来了：蜂群模式下的整个训练编排代码只有大约 70 行 Python。没有模板代码，没有框架体操——干净、可读的逻辑：

```python
import os
from ajet.schema.task import Task, WorkflowTask
from ajet.copilot.job import AgentJetJob
from ajet.task_reader import RouterTaskReader
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.default_config.ajet_config_schema import AjetTaskReader
from ajet.tuner_lib.experimental.swarm_client import SwarmClient

NUM_EPOCH = 10000
AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")

def main():
    dataset = RouterTaskReader(
        reader_type="random_dummy",
        reader_config=AjetTaskReader()
    )

    ajet_job = AgentJetJob(
        base_yaml_config="tutorial/example_werewolves_swarm/werewolves.yaml",
        algorithm="grpo",
        experiment_name="werewolves_swarm",
        max_env_worker=128,
    )

    # 与远程蜂群服务器握手
    swarm_worker = SwarmClient(AJET_SWARM_URL)
    swarm_worker.auto_sync_train_config_and_start_engine(ajet_job)

    GRPO_N = ajet_job.num_repeat
    REMOTE_BATCH_SIZE = ajet_job.batch_size

    def rollout(task):
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(
            discard_episode_timeout=240
        )
        workflow_output = execute_agent(task, api_baseurl_key)
        swarm_worker.end_episode(task, episode_uuid, workflow_output)

    executor = PeriodicDrainThreadPoolExecutor(
        workers=GRPO_N * REMOTE_BATCH_SIZE, max_parallel=64, auto_retry=True
    )
    for _ in range(NUM_EPOCH):
        for _, task in enumerate(dataset.generate_training_tasks()):
            for _ in range(GRPO_N):
                executor.submit_with_periodic_drain(fn=rollout, task=task)


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
```

[GitHub 完整源码](https://github.com/modelscope/AgentJet/blob/main/tutorial/example_werewolves_swarm/agent_roll.py)

代码遵循一个简洁的三步模式，这也是所有 Swarm Client 程序的基本结构：

1. **握手**：`SwarmClient` 连接 Swarm Server，同步训练配置（模型路径、算法、超参数），并启动训练引擎。
2. **Rollout 循环**：对每个任务，客户端调用 `begin_episode()` 获取一个由训练模型支撑的临时 API 端点，运行完整的游戏模拟，然后调用 `end_episode()` 将奖励报告回去。
3. **并行执行**：`PeriodicDrainThreadPoolExecutor` 管理并发的游戏模拟，最多 64 个并行 episode 同时向训练样本池输送数据。

## 架构：工作原理

```
Swarm Server（GPU 集群，8 块 GPU）
    ├── Qwen2.5-7B-Instruct（可训练）
    ├── vLLM 推理引擎
    ├── 策略梯度计算（GRPO）
    └── Checkpoint 管理
         │
         │  ◄── OpenAI 兼容 API ──►
         │
Swarm Client（任意设备）
    ├── 游戏模拟引擎
    ├── 9 人狼人杀游戏
    │    ├── 4 个可训练智能体 → Swarm Server API（7B）
    │    └── 5 个静态智能体 → 外部 Qwen3-235B API
    ├── 奖励计算（胜/负）
    └── Episode 提交到 Swarm Server
```

蜂群架构的核心洞察是**训练与采样的完全解耦**。Swarm Server 处理所有 GPU 密集型工作（推理 + 梯度更新），而 Swarm Client 可以在任何地方运行——甚至是你的笔记本电脑。客户端只需使用 API 端点进行游戏并报告奖励。当样本池达到配置的 batch size 后，服务器自动触发权重更新。

## 训练配置

训练通过 `werewolves.yaml` 配置：

```yaml
ajet:
  model:
    path: /path/to/Qwen2.5-7B-Instruct
  rollout:
    temperature: 0.7
    num_repeat: 6           # 每个任务的 GRPO rollout 次数
    max_env_worker: 64
    max_response_length_in_one_turn: 1024
    max_model_len: 22000
  data:
    train_batch_size: 32
    max_prompt_length: 4000
    max_response_length: 18000
  trainer_common:
    save_freq: 5            # 每 5 步保存一次 Checkpoint
    total_training_steps: 25
    n_gpus_per_node: 8
  enable_swarm_mode: True
```

关键设计决策：
- **`save_freq: 5`** — 频繁保存 Checkpoint 被证明至关重要。训练过程中 CPFS 硬盘被其他同学挤满了两次，幸好保存间隔足够小，每次崩溃损失的进度都很少。
- **`num_repeat: 6`** — GRPO 对每个任务生成 6 个候选回复，用于相对奖励对比。
- **`train_batch_size: 32`** — 服务器等待积累 32 个已完成的任务后，才触发一次梯度更新。

## 训练曲线

训练过程中，CPFS 共享硬盘被其他同学挤满——两次——导致了两次崩溃。还好保存间隔足够小（`save_freq: 5`），每次都能快速恢复，没有损失太多进度。

尽管有这些中断，奖励曲线仍然显示出明显的上升趋势，说明智能体在持续学习中：

- **狼人**学会了协调击杀、在讨论中保持伪装、被投票出局时不暴露身份
- **预言家**学会了何时公布查验结果、如何与村民建立信任

## 快速上手

### 第一步：启动 Swarm Server

```bash
# 使用 Docker（推荐）
docker run --rm -it \
  -v ./swarmlog:/workspace/log \
  -v ./swarmexp:/workspace/saved_experiments \
  -p 10086:10086 --gpus=all --shm-size=32GB \
  ghcr.io/modelscope/agentjet:main bash -c \
  "(ajet-swarm overwatch) & (NO_COLOR=1 LOGURU_COLORIZE=NO ajet-swarm start &>/workspace/log/swarm_server.log)"
```

### 第二步：运行训练客户端

```bash
git clone https://github.com/modelscope/agentjet.git && cd agentjet
pip install -e .
python tutorial/example_werewolves_swarm/agent_roll.py
```

### 第三步：监控训练

```bash
ajet-swarm overwatch
```

## 关于 AgentJet

欢迎关注 [AgentJet](https://modelscope.github.io/AgentJet/)，独创的**蜂群训练模式**，可以帮你在任何环境中舒适地 RL。

AgentJet 基于 [VERL](https://github.com/volcengine/verl)，对于熟悉 VERL 的研究者来说，几乎所有 VERL 实现的算法，都可无损地应用到 AgentJet 中。AgentJet 在此基础上增加了蜂群通信层和时间线合并优化，但核心的训练逻辑保持一致。迁移成本低，性能表现有保障。

更多关于蜂群架构的深入介绍，请参阅[蜂群训练模式介绍](https://modelscope.github.io/AgentJet/en/swarm_intro_blog_zh/)。
