# 可训练工作流

本教程介绍如何在 AgentScope 中定义一个可训练的工作流（Workflow）🚀。

ASTuner 为 AgentScope Workflow 提供了两种方便且**互相兼容**的封装方式：

- 第一种更强调 **简单、易用、容易理解**；
- 第二种更强调 **灵活、可控、易扩展**。

下面分别说明。

## 简单智能体场景

### 1. 在 ASTuner 中转换你的 AgentScope Workflow

**只需要在初始化 ReActAgent 时，把 `model` 参数替换为 `model_tuner` 即可。**

<table style="width: 100%;table-layout: fixed;border: solid 1px;border-radius: 5px;padding: 1em; font-size: 0.5rem;">
  <thead>
    <tr>
      <th>修改前</th>
      <th>修改后</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <pre style="margin: 0; white-space: pre; overflow-x: auto;"><code class="language-python">agent_instance = ReActAgent(
   name=f"Friday",
   sys_prompt="You are a helpful assistant",
   model=DashScopeChatModel(model_name="qwen-max", stream=False),
   formatter=DashScopeChatFormatter(),
)</code></pre>
      </td>
      <td>
        <pre style="margin: 0; white-space: pre; overflow-x: auto;"><code class="language-python">agent_instance = ReActAgent(
   name=f"Friday",
   sys_prompt="You are a helpful assistant",
   <span style="
    color: red;
    font-weight: bold;
">model=model_tuner,  # change here</span>
   formatter=DashScopeChatFormatter(),
)</code></pre>
      </td>
    </tr>
  </tbody>
</table>

然后，将你的 Workflow 封装到一个继承自 `Workflow` 的类中（`from astnuer import Workflow`），这样这个 Workflow 就可以被 ASTuner 训练了。

```python
class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"

    async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        ... your ReActAgent workflow here ...
        return WorkflowOutput(reward=workflow_reward)
```

### 2. 什么时候使用这种「简单实践」

这种写法适合大多数用户，如果你满足下面的情况，可以优先采用：

- 🌟 很清楚**哪些智能体需要被训练**，或者智能体的数量本身就不多；
- ✨ 已经完成了 Workflow 的基础调试，确认在使用非微调模型（例如 `qwen-max`）时工作流是正常可用的；
- 🎇 不需要在运行过程中**动态改变**要训练的智能体集合。

### 3. 代码示例

- 假设你已经实现了一个 ReAct智能体，大致如下：

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

    async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
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


## 进阶智能体场景

当设计的是一个**多智能体协作**的复杂 Workflow，并且每个智能体扮演不同**角色**时，如果 ASTuner 能够「知道」每个智能体的身份，那么在训练和调试时就能提供更好的能力和更高的可控性。

通过多智能体协作，你可以
- 🌟 **精细地控制**哪些智能体会被微调；
- ✨ 为「当前未被训练」的智能体明确定义其使用的默认模型；
- ⚡ 在**不修改 Workflow 源码**的前提下，动态切换不同的可训练目标（trainable agent targets）。

### 1. 可训练开关与模型生命周期

#### 模型多角色注册与使用

在多智能体协作中，每个智能体拥有自己的「角色」。

在 Workflow 中，我们需要显式的注册待训练的智能体角色，并在创建智能体的时候显式的指明角色：

- **注册（register）**：`model_tuner.register_model(agent_role, default_model=...)`
  - 定义：向 Tuner 注册一个待训练的智能体角色，并指定该角色在未训练/不训练时的默认模型。
- **使用（get/bind）**：`model_tuner.get_model(agent_role)`
  - 定义：在构建智能体或执行 Workflow 时，根据 `agent_role` 返回该智能体的模型对象。

#### 可训练模型 vs 不可训练模型

在 Workflow 中能够自由地控制每个智能体的训练状态。一个智能体是否参与训练由 Workflow 的 **`trainable_targets`** 声明决定：

```python
class ExampleMathLearn(Workflow):
    name: str = "a_workflow"
    trainable_targets: list = ["TYPE-ZERO", ...]

    # ...
```

- **可训练（trainable）**：如果智能体（角色）在 `trainable_targets` 列表中，则设置可训练模型。
- **不可训练（non-trainable）**：智能体（角色）不在 `trainable_targets` 列表中，则智能体将使用默认模型。

无论角色异同，所有智能体（角色）共享一个模型实例。也就是具有相同参数的模型将分别扮演不同的角色。

### 2. 升级为进阶 ASTuner Workflow

本节通过一个简单的例子展示使用 `ModelTuner.register_model` 为不同角色注册「可训练模型」，并在构建智能体时以角色维度进行模型绑定。

- 先从一个基础的 AgentScope `ReActAgent` 开始：

```python
agent_instance = ReActAgent(
   name=f"Player-X",
   sys_prompt="You are a helpful assistant",
   model=DashScopeChatModel(model_name="qwen-max", stream=False),
   formatter=DashScopeChatFormatter(),
)
```

- 为智能体声明一个角色标记（tag），并通过 `model_tuner.register_model` 指定该智能体**在未被训练时**应当使用的默认模型：

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

- 最后，将 Workflow 封装到类中，并定义 `trainable_targets`：

```python
class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"
    trainable_targets: list = ["TYPE-ZERO", ...]

    async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        ... your agents and workflow here ...
```

### 3. 一个多智能体示例

下面是一个多智能体场景的示例代码片段：

```python
roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]
players = []
for i, role in enumerate(roles):
    default_model_for_good_guys = OpenAIChatModel(model_name="qwen-max", stream=False)
    default_model_for_bad_guys = OpenAIChatModel(model_name="qwen-plus", stream=False)
    chosen_model = default_model_for_good_guys if role != "werewolf" else default_model_for_bad_guys  # 🌟
    model_tuner.register_model(role, default_model=chosen_model)
    players += [ReActAgent(
        name=f"Player{i + 1}",
        sys_prompt=get_official_agent_prompt(f"Player{i + 1}"),
        model=model_tuner.get_model(role),
        formatter=OpenAIMultiAgentFormatter(),
    )]
```

在这里：

- `role` 既描述了智能体在游戏中的身份（例如狼人、村民等），
- 又作为 `model_tuner.register_model` 的 key，标识一个**可训练目标**；
- `chosen_model` 定义了该角色在「当前未训练」时所使用的默认底座模型；
- 通过这种方式，可以在多智能体场景下灵活地指定和切换各角色的训练与推理行为。