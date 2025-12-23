# Frozen Lake

## 1. 介绍

**Frozen Lake** 是一个经典的强化学习任务，来自 [Gymnasium](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)。

在该环境中，智能体会被安排在一个随机生成的冰湖上，冰湖由安全的冰面（_），危险的冰洞（O）以及终点（G）组成，智能体的位置用 P 表示。智能体的目标是从 P 位置出发到达终点 G，并避开途中的冰洞。智能体每次行动可以选择向上、下、左、右四个方向移动，但由于冰面的光滑特性，智能体有一定概率会滑向非预期的方向。

本实例展示了如何创建一个可训练的 Agent 工作流来解决这一导航挑战。

## 2. 快速开始

### 2.1 准备环境

安装 Frozen Lake 任务所需的依赖：

```bash
pip install gymnasium[toy_text]
```

### 2.2 启动训练

使用提供的配置文件快速开始训练：

```bash
astuner --conf tutorial/example_frozenlake/frozenlake.yaml --backbone=trinity
```

<details>
<summary>快速调试（可选）</summary>

不启用 Ray 在本地运行，便于更快迭代：

```bash
astuner --conf tutorial/example_learn2ask/learn2ask.yaml --backbone='debug' --with-logview
```

如果结果不对，最快的排查点包括：数据路径是否存在、如果 judge 需要 API key 则是否已设置、以及 `agentscope_workflow` 中的 workflow 类路径是否与你的代码位置一致。

</details>

## 3. 实现细节介绍

### 3.1 实现 Frozen Lake 环境

在 `tutorial/example_frozenlake/frozenlake.py` 中的 `FrozenLakeEnv` 类实现了对 Gymnasium Frozen Lake 环境的封装，主要对外提供 `step` 和 `reset` 两个方法。

- `step` 方法会根据智能体的动作（action）返回环境的下一个状态（observation）、奖励（reward）、是否结束（done）以及其他辅助信息。
    - observation: 智能体移动后冰湖的状态信息，使用字符串表示，例如:
        ```
        _  _  G
        _  _  _
        P  O  O
        ```
    - reward: 智能体每次移动获得的奖励，达到终点 G 时奖励为 1，否则为 0。
    - done: 布尔值，如果智能体是否到达终点或掉入冰洞则为 True，否则为 False。
    - info: 其他辅助信息。

- `reset` 方法可根据用户传入的参数重新生成冰湖环境。

### 3.2 实现 Agent

`tutorial/example_frozenlake/frozenlake.py` 中的 `FrozenLakeAgent` 类实现了智能体的决策逻辑，主要通过 `step` 方法接收当前环境状态（observation）作为输入，返回智能体选择的动作（action），其中的核心是一个 ReActAgent。

```python
class FrozenLakeAgent:

    def __init__(self, model: ModelTuner, max_steps: int = 20):
        self.agent = ReActAgent(
            name="frozenlake_agent",
            sys_prompt=SYSTEM_PROMPT,
            model=model,
            formatter=DashScopeChatFormatter(),
            max_iters=2,
        )
        # other initialization code

    async def step(self, current_observation: str) -> str:
        # Step 1: 基于 current_observation 构建用户提示信息
        # Step 2: 调用 ReActAgent 获取原始响应
        # Step 3: 解析响应并返回动作
```

### 3.3 将环境和智能体集成为 Workflow

`tutorial/example_frozenlake/frozenlake.py` 中的 `FrozenLakeWorkflow` 类实现了环境和智能体的集成，主要通过 `step` 和 `reset` 方法与外部交互。

其中的核心流程如下：

```python
class FrozenLakeWorkflow(Workflow):

    async def execute(self, workflow_task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        # init agent and env
        # self.agent = FrozenLakeAgent(...)
        # self.env = FrozenLakeEnv(...)
        # reset environment and get initial `observation_str`
        rewards = []
        for _ in range(self.max_steps):
            action = await self.agent.step(observation_str)
            observation_str, reward, done, info = self.env.step(action)
            rewards.append(reward)
            if done:
                break
        return WorkflowOutput(
            reward=sum(rewards),
        )
```

## 4. 参考效果

### 4.1 训练曲线

*(留空)*

### 4.2 案例分析

*(留空)*
