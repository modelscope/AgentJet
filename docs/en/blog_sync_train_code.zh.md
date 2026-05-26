# 从笔记本到 GPU 集群：用 `sync_train_code` 遥控 Swarm 训练

> `sync_train_code` 允许 AgentJet Swarm Client 在训练启动前，把本地 AgentJet 源码快照上传到远程 Swarm Server。实际效果是：一台笔记本，或者运行在笔记本上的自动化科研 agent，可以修改训练代码、同步到 GPU 集群、重启训练引擎，并继续下一轮实验，而不需要人工部署。

AgentJet Swarm 已经把 rollout 分布式化：训练引擎运行在 GPU 集群上，笔记本或工作站上的 client 运行 agent workflow，并把 reward 回传给服务器。

`sync_train_code` 把这种能力从 rollout 控制推进到**训练代码控制**。Client 不仅能决定发送什么样本，也能决定远程训练引擎运行哪一份 AgentJet 源码。

## 为什么重要

Agent RL 研究中的代码迭代，本身就是实验的一部分：

- reward processing 会改；
- trajectory recording 会改；
- config conversion 会改；
- launcher 行为会改；
- backend 默认参数会改；
- 有些 bug 只有远程训练引擎真正启动后才会暴露。

如果没有代码同步，每一次服务端代码改动都会变成运维任务：SSH 到集群改文件、重新构建镜像、手动同步仓库，或者靠人为纪律保持笔记本和服务器代码一致。

有了 `sync_train_code`，流程变成：

```text
本地修改 -> 同步源码快照 -> 重启远程引擎 -> 观察结果 -> 继续迭代
```

这对自动化科研尤其关键。一个 research agent 可以在没有人工部署步骤的情况下完成闭环：分析结果、修改 AgentJet 代码、调用 `sync_train_code_from_dir()`、重启 Swarm engine、等待指标、决定下一组实验。GPU 集群从“需要人手动操作的机器”，变成“可被程序控制的执行后端”。

## 最小用法

先在 GPU 集群上启动 Swarm Server：

```bash
ajet-swarm start --swarm-port=10086
```

然后从本地 AgentJet checkout 运行 client：

```python
import os
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient

swarm_worker = SwarmClient("http://gpu-cluster-host:10086")

if os.getenv("SYNC_CODE", "0") == "1":
    swarm_worker.sync_train_code_from_dir(os.getcwd(), force_restart=True)

swarm_worker.auto_sync_train_config_and_start_engine(
    AgentJetJob(
        experiment_name="remote_controlled_grpo",
        algorithm="grpo",
        n_gpu=8,
        model="/mnt/models/Qwen2.5-7B-Instruct",
        batch_size=32,
        num_repeat=4,
    ),
    force_restart=True,
)
```

启用代码同步运行：

```bash
SYNC_CODE=1 python your_swarm_client.py
```

本机开发时，也可以配合自动启动本地 Swarm Server：

```python
swarm_worker = SwarmClient(
    "http://localhost:10086",
    auto_start_swarm_server=True,
)
swarm_worker.sync_train_code_from_dir(os.getcwd(), force_restart=True)
```

如果是远程 GPU 集群，需要先在服务器上运行 `ajet-swarm start`，再从 client 连接过去。

## 遥控了什么

`sync_train_code` 控制的是训练引擎进程使用的 AgentJet 源码。它和已有 Swarm Client API 组合后，可以覆盖远程训练的核心生命周期：

| 操作 | Client API | 远程效果 |
|---|---|---|
| 同步代码 | `sync_train_code_from_dir()` | 上传带时间戳的 `ajet/` 源码快照。 |
| 同步配置 | `auto_sync_train_config_and_start_engine()` | 发送模型路径、算法、GPU 数量、batch size 等训练参数。 |
| 启动引擎 | `start_engine()` | 用同步后的代码和配置启动训练。 |
| 停止引擎 | `stop_engine()` | 停止当前引擎并回到 `ENGINE.OFFLINE`。 |
| 运行 rollout | `begin_episode()` / `end_episode()` | 向远程 trainer 发送样本和 reward。 |

这些能力组合起来，Swarm Client 就成为远程训练集群的控制平面。

## 它如何工作

实现机制很直接，也足够安全：

```text
本地 checkout
  -> git ls-files -- ajet
  -> 创建 /tmp/ajet_train_code_*.zip
  -> POST /sync_train_code
  -> 服务端解压到 ./ajet_temp/<timestamp>/ajet
  -> start_engine 设置 ISOLATED_AGENTJET_BASE_DIR
  -> 训练 subprocess 优先从同步快照 import ajet
```

服务器上的 checkout 不会被覆盖。每次同步都会创建一份隔离的、带时间戳的源码副本。训练成功启动时，服务端日志会出现类似内容：

```text
[start_engine] Using synced training code from ./ajet_temp/20260526_120000_123456/ajet
```

## 重要规则

只会打包 `ajet/` 下被 Git 跟踪的文件。

- 已修改的 tracked 文件会被包含。
- 新建但未被 Git 跟踪的文件不会被包含。
- `ajet/` 之外的文件不会被包含。
- 数据集、checkpoint、虚拟环境、ignored 文件不会被包含。

如果新增了模块，同步前先 stage：

```bash
git add ajet/path/to/new_module.py
```

代码同步只允许在服务端 `ENGINE.OFFLINE` 时进行。开发时如果你明确希望停止当前训练并使用新代码重启，可以使用 `force_restart=True`。

`sync_train_code` 不会同步 Python 依赖。如果新代码依赖新的包，需要先在 GPU 服务器环境中安装。

## 对自动化科研的意义

自动化科研需要闭环：

```text
规划 -> 修改代码/配置 -> 启动训练 -> 等待结果 -> 分析 -> 调整 -> 再启动
```

最容易卡住的环节通常是部署。如果每次代码改动都需要人登录 GPU 服务器处理，那么这个闭环就不是真正自动化的。

AgentJet Swarm 加上 `sync_train_code` 后，自动化科研系统可以在实验层面运转：

- 生成新的假设；
- 修改 AgentJet 训练逻辑或配置映射；
- 把修改后的代码同步到集群；
- 重启训练；
- 监控结果；
- 决定下一次 patch 或下一组实验。

这使 GPU 集群变成一个可编程的科研仪器。笔记本或自动化 agent 负责研究闭环，集群负责稳定计算。

`sync_train_code` 的核心价值就在这里：**远程 Swarm 训练不再只是分布式 rollout，而是可以被远程编程和自动化调度的训练系统。**
