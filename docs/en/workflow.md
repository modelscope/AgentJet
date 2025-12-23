# Trainable Workflow

This tutorial introduces how to define a trainable workflow 🚀 with AgentScope.

ASTuner provides two convenient and **mutually compatible** ways to wrap an AgentScope Workflow:

- The first emphasizes **simplicity, ease of use, and readability**;
- The second emphasizes **flexibility, controllability, and extensibility**.

## Simple Agent Scenario

### 1. Convert your AgentScope Workflow in ASTuner

**Simply set ReActAgent's `model` argument to `model_tuner` when initializing your agent.**

<table style="width: 100%;table-layout: fixed;border: solid 1px;border-radius: 5px;padding: 1em;">
  <thead>
    <tr>
      <th>Before</th>
      <th>After</th>
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
   model=model_tuner,  # change here
   formatter=DashScopeChatFormatter(),
)</code></pre>
      </td>
    </tr>
  </tbody>
</table>

Then, wrap your workflow in a class that inherits `Workflow` (`from astnuer import Workflow`), and the workflow is ready to be tuned.

```python
class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"

    async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        ... your ReActAgent workflow here ...
        return WorkflowOutput(reward=workflow_reward)

```


### 2. When to use this simple practice

This practice suits most users. You can choose it if you:

- 🌟 Know exactly which agents should be trained, or the number of agents are small;
- ✨ Already finished basic debugging of your workflow, confirming that your workflow works well when implemented with a non-tuned model such as `qwen-max`;
- 🎇 Do not need to change which agents are trained on the fly.


### 3. Code Example

Suppose you have built a ReAct agent that looks like this:

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

Then, wrap it in a workflow class:

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
            model=model_tuner,
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



## Advanced Agent Scenario

When designing a **multi-agent collaborative** workflow in which each agent plays a different **role**, ASTuner can provide better training and debugging capabilities.

With a multi-agent setup, you can:

- 🌟 **Precisely control** which agents are fine-tuned;
- ✨ Explicitly define the default model used by agents that are **not being trained**;
- ⚡ Switch trainable targets on the fly **without modifying** the workflow source code.

### 1. Trainability switch and model lifecycle

#### Multi-role model registration and usage

In a multi-agent workflow, each agent is associated with a role.

Within the workflow, we register roles to be tuned, and specify the role explicitly when creating agents:

- **Register**: `model_tuner.register_model(agent_role, default_model=...)`
  - Definition: register an agent role in the tuner and provide the default model to be used when the role is not trained / not being trained.
- **Use (bind)**: `model_tuner.get_model(agent_role)`
  - Definition: when constructing agents or executing the workflow, return the model object bound to the given `agent_role`.

#### Trainable vs. non-trainable models

In a workflow, trainability can be controlled at the role level. Whether a role participates in training is determined by the workflow's **`trainable_targets`**:

```python
class ExampleMathLearn(Workflow):
    name: str = "a_workflow"
    trainable_targets: list = ["TYPE-ZERO", ...]

    # ...
```

- **Trainable**: if an agent appears in `trainable_targets`, the agent uses the trainable model.
- **Non-trainable**: if an agent does not appear in `trainable_targets`, the agent uses the registered default model.

Regardless of role differences, all agents share a single model instance; i.e., one set of parameters is used to play different roles.

### 2. Promote to an advanced ASTuner Workflow

This section demonstrates how to register role-specific trainable targets via `model_tuner.register_model`, and bind models to agents by role during construction.

- Let's begin from a basic AgentScope `ReActAgent`:

```python
agent_instance = ReActAgent(
   name=f"Player-X",
   sys_prompt="You are a helpful assistant",
   model=DashScopeChatModel(model_name="qwen-max", stream=False),
   formatter=DashScopeChatFormatter(),
)
```

- Declare the tag of an agent, and specify what model should be used when an agent is not being trained using `model_tuner.register_model`:
```python
agent_role = "TYPE-ZERO"
default_model_when_not_training = DashScopeChatModel(model_name="qwen-max", stream=False)
model_tuner.register_model(agent_role, default_model=default_model_when_not_training)
```

- Create `ReActAgent` to link with the `agent_role` using `model_tuner.get_model`:
```python
agent_instance = ReActAgent(
   name=f"Player-X",
   sys_prompt="You are a helpful assistant",
   model=model_tuner.get_model(agent_role), # replace there
   formatter=DashScopeChatFormatter(),
)
```

- Wrap the workflow in a class and define `trainable_targets`:

```python
class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"

    trainable_targets: list = ["TYPE-ZERO", ...]

    async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        ... your agents and workflow here ...
```

### 3. A multi-agent example

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

In this example:

- `role` describes an agent's in-game identity (e.g., werewolf, villager) and also serves as the key for `model_tuner.register_model`.
- `chosen_model` defines the default base model to use when the role is not being trained.
- With this setup, you can flexibly specify and switch training and inference behaviors by role in multi-agent workflows.
