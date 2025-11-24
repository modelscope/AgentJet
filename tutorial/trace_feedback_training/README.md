# 从 Tracing Log 回流数据训练一个新的 Agent

ASTune 支持回收和利用 Agent 生产过程产生的聊天日志，不断训练、提升 Agent 表现。在该文档中，我们将展示如何从 Tracing Log 回流数据训练一个 Agent。

## 1. 准备数据

为使用回流数据训练，应当已经有一个基于 agentscope 构建的 Agent，且连接在 studio 上运行了一段时间。

在本示例中，我们在 `agent_deployed.py` 实现了一个能用于解决数学问题的 agent。为方便演示，我们首先使用它模拟数据收集的过程。

1. 安装 [agentscope-studio](https://github.com/agentscope-ai/agentscope-studio)。
2. 依照默认端口设置启动 agentscope-studio。
3. 运行 `agent_deployed.py`，并模拟用户与 agent 聊天交互。

在完成几轮交互后，studio 在 `~/AgentScope-Studio/database.sqlite` 中保存了 tracing log，其中包含了用户与 agent 的对话记录。

> **AgentScope 与 Studio 版本**
>
> 建议使用 AgentScope 及与之匹配的 Studio 版本：
> 
> - AgentScope (v1.0.7)
> - Studio (23eb7c0b1185486d1baca36aea0ce8b85ea9de48)

## 2. 启动数据回流训练

在获得 tracing log（`database.sqlite`）后，接下来就能使用本项目的回流训练功能来训练一个 Agent。

1. 修改配置文件中参数 `task_reader` 为 `tracing`，启用回流模式。
2. 依照需求，配置 `tracing` 中的数据库地址以及数据筛选设置。
3. 仿照正常的训练流程，配置其他参数与 Reward。

在文件夹 `trace_feedback_training` 中已准备了一份示例 database 及相应的训练配置。

在一切准备妥当后，使用 launcher 启动训练。

```bash
python launcher.py --conf tutorial/trace_feedback_training/trace_feedback_training.yaml --backbone='trinity' --with-ray
```

## 3. 部署新的 Agent

现在，可以将全新的 Agent 部署到生产环境，最终实现迭代式的回流数据训练增强。