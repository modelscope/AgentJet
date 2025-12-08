# 可训练工作流

本教程介绍如何在 AgentScope 中定义一个可训练的工作流（Workflow）🚀。

ASTuner 为 AgentScope Workflow 提供了两种方便且**互相兼容**的封装方式：

- 第一种更强调 **简单、易用、容易理解**；
- 第二种更强调 **灵活、可控、易扩展**。

下面分别说明。

## 简单 Agent 场景

### 1. 在 ASTuner 中转换你的 AgentScope Workflow

**只需要在初始化 ReActAgent 时，把 `model` 参数替换为 `model_tuner` 即可。**

```python

# 修改前 >>>
agent_instance = ReActAgent(
   name=f"Friday",
   sys_prompt="You are a helpful assistant",
   model=DashScopeChatModel(model_name="qwen-max", stream=False),
   formatter=DashScopeChatFormatter(),
)

# 修改后 <<<
agent_instance = ReActAgent(
   name=f"Friday",
   sys_prompt="You are a helpful assistant",
   model=model_tuner,  # change here
   formatter=DashScopeChatFormatter(),
)
```

然后，将你的 Workflow 封装到一个继承自 `Workflow` 的类中（`from astnue import Workflow`），这样这个 Workflow 就可以被 ASTuner 训练了。

```python
class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"

    async def agentscope_execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        ... your ReActAgent workflow here ...
        return WorkflowOutput(reward=workflow_reward)
```

### 2. 什么时候使用这种「简单实践」

这种写法适合大多数用户，如果你满足下面的情况，可以优先采用：

- 🌟 很清楚**哪些 Agent 需要被训练**，或者 Agent 的数量本身就不多；
- ✨ 已经完成了 Workflow 的基础调试，确认在使用非微调模型（例如 `qwen-max`）时工作流是正常可用的；
- 🎇 不需要在运行过程中**动态改变**要训练的 Agent 集合。

### 3. 代码示例

- 假设你已经实现了一个 ReAct Agent，大致如下：

```python
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, execute_python_code

self.toolkit = Toolkit()
self.toolkit.register_tool_function(execute_python_code)
self.agent = ReActAgent(
    name="math_react_agent",
    sys_prompt=system_prompt,
    model=DashScopeChatModel(model='qwen-max'),
    formatter=DashScopeChatFormatter(),
    toolkit=self.toolkit,
    memory=InMemoryMemory(),
)
msg = Msg("user", query, role="user")
result = await self.agent.reply(msg, structured_model=FinalResult)
final_answer = extract_final_answer(result)
```

- 那么，你只需要把它包裹进一个 Workflow 类即可：

```python
class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"

    async def agentscope_execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        from agentscope.agent import ReActAgent
        from agentscope.formatter import DashScopeChatFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit, execute_python_code

        self.toolkit = Toolkit()
        self.toolkit.register_tool_function(execute_python_code)
        self.agent = ReActAgent(
            name="math_react_agent",
            sys_prompt=system_prompt,
            model=model_tuner,  # 这里改为使用 model_tuner
            formatter=DashScopeChatFormatter(),
            toolkit=self.toolkit,
            memory=InMemoryMemory(),
        )

        query = task.task.main_query
        msg = Msg("user", query, role="user")
        result = await self.agent.reply(msg, structured_model=FinalResult)
        final_answer = extract_final_answer(result)
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})
```


## 进阶 Agent 场景

当你设计的是一个**多 Agent 协作**的复杂 Workflow，并且每个 Agent 扮演不同角色时，如果 ASTuner 能够「知道」每个 Agent 的身份，那么在训练和调试时就能提供更好的能力和更高的可控性。

### 1. 升级为进阶 ASTuner Workflow

核心思路是：使用 `ModelTuner.register_model` 注册不同的「可训练目标」（agent targets）。

- 先从一个基础的 AgentScope `ReActAgent` 开始：

```python
agent_instance = ReActAgent(
   name=f"Player-X",
   sys_prompt="You are a helpful assistant",
   model=DashScopeChatModel(model_name="qwen-max", stream=False),
   formatter=DashScopeChatFormatter(),
)
```

- 为 Agent 声明一个角色标记（tag），并通过 `model_tuner.register_model` 指定该 Agent **在未被训练时**应当使用的默认模型：

```python
agent_role = "TYPE-ZERO"
default_model_when_not_training = DashScopeChatModel(model_name="qwen-max", stream=False)
model_tuner.register_model(agent_role, default_model=default_model_when_not_training)
```

- 再使用 `model_tuner.get_model` 创建与 `agent_role` 绑定的 `ReActAgent`：

```python
agent_instance = ReActAgent(
   name=f"Player-X",
   sys_prompt="You are a helpful assistant",
   model=model_tuner.get_model(agent_role),  # replace there
   formatter=DashScopeChatFormatter(),
)
```

- 最后，将 Workflow 封装到类中，并定义 `trainable_tragets`：

```python
class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"
    trainable_targets: list = ["TYPE-ZERO", ...]

    async def agentscope_execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        ... your agents and workflow here ...
```

### 2. 何时使用进阶 Workflow，而不是简单 Workflow

推荐在以下场景下采用这种进阶写法：

- 🌟 需要**精细地控制**哪些 Agent 会被微调；
- ✨ 希望为「当前未被训练」的 Agent 明确定义其使用的默认模型；
- ⚡ 希望在**不修改 Workflow 源码**的前提下，动态切换不同的可训练目标（trainable agent targets）。


### 3. 一个多 Agent 示例

下面是一个多 Agent 场景的示例代码片段：

```python
roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]
players = []
for i, role in enumerate(roles):
    default_model_for_good_guys = OpenAIChatModel(model_name="qwen-max", stream=False)
    default_model_for_bad_guys = OpenAIChatModel(model_name="qwen-plus", stream=False)
    chosen_model = default_model_for_good_guys if role != "werewolf" else default_model_for_bad_guys  # 🌟
    players += [ReActAgent(
        name=f"Player{i + 1}",
        sys_prompt=get_official_agent_prompt(f"Player{i + 1}"),
        model=model_tuner.register_model(role, default_model=chosen_model),
        formatter=OpenAIMultiAgentFormatter(),
    )]
```

在这里：

- `role` 既描述了 Agent 在游戏中的身份（例如狼人、村民等），
- 又作为 `model_tuner.register_model` 的 key，标识一个**可训练目标**；
- `chosen_model` 定义了该角色在「当前未训练」时所使用的默认底座模型；
- 通过这种方式，可以在多 Agent 场景下灵活地指定和切换各角色的训练与推理行为。

