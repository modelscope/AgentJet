# App交互模拟

本教程介绍如何训练一个智能体与 AppWorld 交互并解决复杂的任务。

## 1. 概述

AppWorld 是一个模拟现实 APP 操作的沙盒环境，包含 9 个日常应用，可通过 457 个 API 操作，并预置了 106 个在模拟世界中生活的数字用户行为数据。我们的目标是调优一个智能体，使其能够有效地在这些应用中执行并完成复杂任务。

本文结构如下：

- 快速开始
- 理解实现：Workflow 核心流程、配置、代码位置、奖励机制
- 结果：训练曲线与案例对比

## 2. 快速开始

### 2.1 准备工作

首先，需要准备 AppWorld 所需的环境服务：

- 下载并部署 `env_service`
- 下载并部署 `appworld`

详细的安装与启动步骤，请参考 [EnvService 文档](https://modelscope.github.io/AgentEvolver/tutorial/install/#step-2-setup-env-service-appworld-as-example)。

### 2.2 开始训练

运行训练脚本：

```bash
astuner --conf tutorial/example_appworld/appworld.yaml --backbone='trinity' --with-ray
```

<details>
<summary>快速调试（可选）</summary>

不启用 Ray 在本地运行，便于更快迭代：

```bash
astuner --conf tutorial/example_appworld/learn2ask.yaml --backbone='debug' --with-logview
```

如果结果不对，最快的排查点包括：数据路径是否存在、如果 judge 需要 API key 则是否已设置、以及 `agentscope_workflow` 中的 workflow 类路径是否与你的代码位置一致。

</details>

## 3. 理解实现

本节将对如何搭建 AppWorld workflow 进行更详细的说明，包括核心流程、配置与关键代码位置。

### 3.1 核心流程

AppWorld 示例所使用的 AgentScope Workflow 代码位于：`tutorial/example_appworld/appworld.py`。

代码首先定义了 AgentScope Workflow（将智能体的 `model` 设置为 `model_tuner`）：

```python
agent = ReActAgent(
    name="Qwen",
    sys_prompt=first_msg["content"],
    model=model_tuner,
    formatter=DashScopeChatFormatter(),
    memory=InMemoryMemory(),
    toolkit=None,
    print_hint_msg=False,
)

env = workflow_task.gym_env
for step in range(model_tuner.config.astuner.rollout.multi_turn.max_steps):
    # agentscope 处理交互消息
    reply_message = await agent(interaction_message)
    # env_service 协议
    obs, _, terminate, _ = env.step(
        action={"content": reply_message.content, "role": "assistant"}
    )
    # 使用环境的输出构造新的交互消息
    interaction_message = Msg(name="env", content=obs, role="user")
    # 是否终止？
    if terminate:
        break
    if model_tuner.get_context_tracker().context_overflow:
        break
```

在上述代码中：

- `env.step`：模拟 gym 接口。输入一个 action，返回四元组 `(observation, reward, terminate_flag, info)`。
- `model_tuner.get_context_tracker().context_overflow`：检查当前上下文窗口是否已经超过 token 限制。


### 3.2 奖励

在 `astuner/task_judge/env_service_as_judge.py` 中，我们通过 `env.evaluate(...)` 从环境中读取奖励信号。

你也可以参考该文件，为自己的任务实现专用的 Judge 模块。


### 3.3 配置说明

`tutorial/example_appworld/appworld.yaml` 中的关键配置参数用 ✨✨✨✨ 标出：

1. **读取任务**（对应字段：`astuner.task_reader`）
2. **定义 Workflow**（对应字段：`astuner.rollout.agentscope_workflow`）
   - 示例：如果 AgentScope Workflow 定义在 `tutorial/example_appworld/appworld.py` 中的 `ExampleAgentScopeWorkflow` 类里
   - 则配置：
`astuner.rollout.agentscope_workflow = "tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow"`
3. **定义评分函数**（对应字段：`astuner.task_judge.judge_protocol`）
   - 示例：
`astuner.task_judge.judge_protocol = "astuner.task_judge.env_service_as_judge->EnvServiceJudge"`
4. **指定模型**（对应字段：`astuner.model.path`）

```yaml
astuner:
  project_name: example_appworld
  experiment_name: "read_yaml_name"
  task_judge:
    # ✨✨✨✨ 编写并选择评估函数
    judge_protocol: agentscope_tuner.task_judge.env_service_as_judge->EnvServiceJudge
  model:
    # ✨✨✨✨ 设置需要训练的模型
    path: /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-14B-Instruct
  rollout:
    # ✨✨✨✨ 编写并选择智能体
    agentscope_workflow: tutorial.example_appworld.appworld->ExampleAgentScopeWorkflow
    agentscope_disable_toolcalls: True
  debug:
    debug_max_parallel: 1
    debug_first_n_tasks: 1
```

## 4. 结果

### 4.1 训练曲线

![Training curve (small batch)](https://img.alicdn.com/imgextra/i2/O1CN01toRt2c1Nj8nKDqoTd_!!6000000001605-2-tps-1410-506.png)

> **可视化说明：** 训练曲线由 SwanLab 生成。详见 [训练可视化](./visualization.md).

随着训练的进展，奖励也会增加。这通常意味着智能体在**两个方面**变得更加稳定：

* **遵循正确的 API 协议**：它学会在调用前查阅 API 文档，并使用有效的 API 端点，而不是虚构不存在的 API。
* **完成多步工作流**：它能够正确获取 access token，并串联多个 API 调用以完成复杂任务。

### 4.2 案例展示

#### 调优前：

1. 频繁调用不存在的 API

![Before tuning](https://img.alicdn.com/imgextra/i1/O1CN015FgjqI20Ip3AJybr0_!!6000000006827-2-tps-1259-683.png)

智能体在不检查 API 是否存在的情况下产生幻觉，导致重复失败。

2. 没有学会按照说明去获取 access token

![Before tuning](https://img.alicdn.com/imgextra/i1/O1CN01bGZ1s01VyjCSrTJte_!!6000000002722-2-tps-1181-954.png)

智能体在未先获取所需的访问令牌（access token）的情况下尝试调用受保护的 API，导致认证错误。

#### 调优后：

1. 会先查阅 API 文档，并学会使用有效的 API

![After tuning](https://img.alicdn.com/imgextra/i4/O1CN01VRIDy922PoKD1bETl_!!6000000007113-2-tps-1180-944.png)

智能体现在会先检查可用的 API 再发起调用，从而避免臆造不存在的接口端点。

2. 学会正确获取 access token

![After tuning](https://img.alicdn.com/imgextra/i2/O1CN01xiF9UU20h62dyrZ4x_!!6000000006880-2-tps-1182-793.png)

智能体在访问受保护的 API 之前，会先正确完成认证步骤。

> **Token级可视化：** 这些详细日志由 Beast-Logger 生成。详见 [Beast-Logger 使用说明](./beast_logger.md).
