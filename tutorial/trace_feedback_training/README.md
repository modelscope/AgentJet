# 从 Tracing Log 回流数据训练一个新的 Agent

在该文档中，我们将展示如何从 Tracing Log 回流数据训练一个新的 Agent。

## 1. 准备数据

假设你已经有一个基于 agentscope 构建的 Agent，且连接在 studio 上运行了一段时间。

在 `agent_deployed.py` 中展示了一个能够解决数学问题的最小化 agent。现在，我们将先使用它来模拟数据收集过程。

1. 安装 agentscope-studio。
2. 依照默认设置启动 agentscope-studio。
3. 运行 `agent_deployed.py`，并与 agent 交互。

在完成几轮交互后，studio 在 `~/AgentScope-Studio/database.sqlite` 中保存了 tracing log。

## 2. 启动数据回流训练

在获得 tracing log（`database.sqlite`）后，接下来就能使用本项目的回流训练功能来训练一个 Agent。

1. 配置参数中 `task_reader` 为 `tracing` 来使用回流模式。
2. 依照需求，配置 `tracing` 中的数据库地址以及数据筛选设置。
3. 仿照正常的训练流程，配置其他参数与 Reward。

在 `trace_feedback_training` 中已准备了一份示例 database 及相应的训练配置文件。

在一切准备妥当后，使用 launcher 启动训练。

```bash
python launcher.py --conf tutorial/trace_feedback_training/trace_feedback_training.yaml --backbone='trinity' --with-ray
```