# 可训练工作流

本教程介绍如何在 AgentScope 中定义一个可训练的工作流（Workflow）。

!!! info "两种封装方式"
    ASTuner 为 AgentScope Workflow 提供了两种方便且**互相兼容**的封装方式：
    
    - **简单模式**：强调简单、易用、容易理解
    - **进阶模式**：强调灵活、可控、易扩展

---

## 简单智能体场景

### 1. 在 ASTuner 中转换你的 AgentScope Workflow

**只需要在初始化 ReActAgent 时，把 `model` 参数替换为 `model_tuner` 即可。**

=== "修改前"

    ```python
    agent_instance = ReActAgent(
        name=f"Friday",
        sys_prompt="You are a helpful assistant",
        model=DashScopeChatModel(model_name="qwen-max", stream=False),
        formatter=DashScopeChatFormatter(),
    )
    ```

=== "修改后"

    ```python
    agent_instance = ReActAgent(
        name=f"Friday",
        sys_prompt="You are a helpful assistant",
        model=model_tuner,  # 关键修改点
        formatter=DashScopeChatFormatter(),
    )
    ```

然后，将您的 Workflow 封装到一个继承自 `Workflow` 的类中：

```python
from agentscope_tuner import Workflow, WorkflowTask, WorkflowOutput, ModelTuner

class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"

    async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        # ... your ReActAgent workflow here ...
        return WorkflowOutput(reward=workflow_reward)
```

### 2. 什么时候使用「简单模式」

!!! tip "适用场景"
    这种写法适合大多数用户，如果您满足下面的情况，可以优先采用：
    
    - <img src="https://api.iconify.design/lucide:star.svg" class="inline-icon" /> 很清楚**哪些智能体需要被训练**，或者智能体的数量本身就不多
    - <img src="https://api.iconify.design/lucide:sparkles.svg" class="inline-icon" /> 已经完成了 Workflow 的基础调试，确认在使用非微调模型时工作流正常可用
    - <img src="https://api.iconify.design/lucide:sparkle.svg" class="inline-icon" /> 不需要在运行过程中**动态改变**要训练的智能体集合

### 3. 代码示例

假设您已经实现了一个 ReAct 智能体：

```python title="原始智能体代码"
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

只需将它包裹进一个 Workflow 类即可：

```python title="封装为 Workflow"
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
            model=model_tuner,  # 关键：改为使用 model_tuner
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

---

## 进阶智能体场景

当设计的是一个**多智能体协作**的复杂 Workflow，并且每个智能体扮演不同**角色**时，如果 ASTuner 能够「知道」每个智能体的身份，那么在训练和调试时就能提供更好的能力和更高的可控性。

!!! success "进阶模式的优势"
    通过多智能体协作，您可以：
    
    - <img src="https://api.iconify.design/lucide:star.svg" class="inline-icon" /> **精细地控制**哪些智能体会被微调
    - <img src="https://api.iconify.design/lucide:sparkles.svg" class="inline-icon" /> 为「当前未被训练」的智能体明确定义其使用的默认模型
    - <img src="https://api.iconify.design/lucide:zap.svg" class="inline-icon" /> 在**不修改 Workflow 源码**的前提下，动态切换不同的可训练目标

### 1. 可训练开关与模型生命周期

#### 模型多角色注册与使用

在多智能体协作中，每个智能体拥有自己的「角色」。在 Workflow 中，我们需要显式地注册待训练的智能体角色：

| 操作 | 方法 | 说明 |
|------|------|------|
| **注册** | `model_tuner.register_model(agent_role, default_model=...)` | 向 Tuner 注册一个待训练的智能体角色，并指定该角色在未训练时的默认模型 |
| **使用** | `model_tuner.get_model(agent_role)` | 根据 `agent_role` 返回该智能体的模型对象 |

#### 可训练模型 vs 不可训练模型

在 Workflow 中可以自由控制每个智能体的训练状态。一个智能体是否参与训练由 Workflow 的 **`trainable_targets`** 声明决定：

```python
class ExampleMathLearn(Workflow):
    name: str = "a_workflow"
    trainable_targets: list = ["TYPE-ZERO", ...]

    # ...
```

| 状态 | 条件 | 行为 |
|------|------|------|
| **可训练** | 角色在 `trainable_targets` 列表中 | 使用可训练模型 |
| **不可训练** | 角色不在 `trainable_targets` 列表中 | 使用默认模型 |

!!! info "模型共享"
    无论角色异同，所有智能体（角色）共享一个模型实例。也就是说，具有相同参数的模型将分别扮演不同的角色。

### 2. 升级为进阶 ASTuner Workflow

本节通过一个简单的例子展示使用 `ModelTuner.register_model` 为不同角色注册「可训练模型」。

=== "步骤 1：基础智能体"

    ```python
    agent_instance = ReActAgent(
        name=f"Player-X",
        sys_prompt="You are a helpful assistant",
        model=DashScopeChatModel(model_name="qwen-max", stream=False),
        formatter=DashScopeChatFormatter(),
    )
    ```

=== "步骤 2：注册角色"

    ```python
    agent_role = "TYPE-ZERO"
    default_model = DashScopeChatModel(model_name="qwen-max", stream=False)
    model_tuner.register_model(agent_role, default_model=default_model)
    ```

=== "步骤 3：绑定模型"

    ```python
    agent_instance = ReActAgent(
        name=f"Player-X",
        sys_prompt="You are a helpful assistant",
        model=model_tuner.get_model(agent_role),  # 使用角色绑定的模型
        formatter=DashScopeChatFormatter(),
    )
    ```

=== "步骤 4：定义 Workflow"

    ```python
    class ExampleMathLearn(Workflow):
        name: str = "math_agent_workflow"
        trainable_targets: list = ["TYPE-ZERO", ...]

        async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
            # ... your agents and workflow here ...
    ```

### 3. 多智能体示例

下面是一个狼人杀游戏的多智能体场景示例：

```python title="werewolves_workflow.py"
roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]
players = []

for i, role in enumerate(roles):
    # 好人使用 qwen-max，狼人使用 qwen-plus
    default_model_for_good_guys = OpenAIChatModel(model_name="qwen-max", stream=False)
    default_model_for_bad_guys = OpenAIChatModel(model_name="qwen-plus", stream=False)
    chosen_model = default_model_for_good_guys if role != "werewolf" else default_model_for_bad_guys
    
    # 注册角色
    model_tuner.register_model(role, default_model=chosen_model)
    
    # 创建智能体
    players += [ReActAgent(
        name=f"Player{i + 1}",
        sys_prompt=get_official_agent_prompt(f"Player{i + 1}"),
        model=model_tuner.get_model(role),
        formatter=OpenAIMultiAgentFormatter(),
    )]
```

!!! note "代码说明"
    - `role` 既描述了智能体在游戏中的身份（例如狼人、村民等）
    - 又作为 `model_tuner.register_model` 的 key，标识一个**可训练目标**
    - `chosen_model` 定义了该角色在「当前未训练」时所使用的默认底座模型
    - 通过这种方式，可以在多智能体场景下灵活地指定和切换各角色的训练与推理行为

---

## 下一步

<div class="card-grid">
<a href="./data_pipeline/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>数据管道</h3></div><p class="card-desc">了解如何加载和处理训练数据。</p></a>
<a href="./task_judger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>任务评判器</h3></div><p class="card-desc">学习如何定义奖励函数。</p></a>
</div>
