# Trainable Workflow

This tutorial introduces how to define a trainable workflow with AgentScope.

!!! info "Two Approaches"
    ASTuner provides two convenient and **mutually compatible** ways to wrap an AgentScope Workflow:
    
    - **Simple**: Emphasizes simplicity, ease of use, and readability
    - **Advanced**: Emphasizes flexibility, controllability, and extensibility

---

## Simple Agent Scenario

### 1. Convert Your AgentScope Workflow in ASTuner

Simply set ReActAgent's `model` argument to `model_tuner` when initializing your agent.

=== "Before"

    ```python
    agent_instance = ReActAgent(
       name=f"Friday",
       sys_prompt="You are a helpful assistant",
       model=DashScopeChatModel(model_name="qwen-max", stream=False),
       formatter=DashScopeChatFormatter(),
    )
    ```

=== "After"

    ```python
    agent_instance = ReActAgent(
       name=f"Friday",
       sys_prompt="You are a helpful assistant",
       model=model_tuner,  # ← change here
       formatter=DashScopeChatFormatter(),
    )
    ```

Then, wrap your workflow in a class that inherits `Workflow`:

```python
from agentscope_tuner import Workflow, WorkflowTask, WorkflowOutput, ModelTuner

class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"

    async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        # ... your ReActAgent workflow here ...
        return WorkflowOutput(reward=workflow_reward)
```

### 2. When to Use This Simple Practice

!!! tip "Choose Simple Practice If You..."
    - <img src="https://api.iconify.design/lucide:star.svg" class="inline-icon" /> Know exactly which agents should be trained, or the number of agents is small
    - <img src="https://api.iconify.design/lucide:sparkles.svg" class="inline-icon" /> Already finished basic debugging of your workflow
    - <img src="https://api.iconify.design/lucide:sparkle.svg" class="inline-icon" /> Do not need to change which agents are trained on the fly

### 3. Code Example

Suppose you have built a ReAct agent:

```python title="Original AgentScope Code"
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

Wrap it in a workflow class:

```python title="Trainable Workflow"
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
            model=model_tuner,  # ← Key change!
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

## Advanced Agent Scenario

When designing a **multi-agent collaborative** workflow where each agent plays a different **role**, ASTuner provides enhanced training and debugging capabilities.

!!! success "Multi-Agent Benefits"
    With a multi-agent setup, you can:
    
    - <img src="https://api.iconify.design/lucide:star.svg" class="inline-icon" /> **Precisely control** which agents are fine-tuned
    - <img src="https://api.iconify.design/lucide:sparkles.svg" class="inline-icon" /> Explicitly define the default model for agents **not being trained**
    - <img src="https://api.iconify.design/lucide:zap.svg" class="inline-icon" /> Switch trainable targets on the fly **without modifying** source code

### 1. Trainability Switch and Model Lifecycle

#### Multi-role Model Registration

In a multi-agent workflow, each agent is associated with a role.

| Method | Description |
|--------|-------------|
| `model_tuner.register_model(role, default_model=...)` | Register an agent role and provide the default model for non-training scenarios |
| `model_tuner.get_model(role)` | Return the model object bound to the given role |

#### Trainable vs. Non-trainable Models

Trainability is controlled at the role level via **`trainable_targets`**:

```python
class ExampleMathLearn(Workflow):
    name: str = "a_workflow"
    trainable_targets: list = ["TYPE-ZERO", ...]  # ← Roles to train

    # ...
```

| Scenario | Model Used |
|----------|------------|
| Role in `trainable_targets` | Trainable model |
| Role NOT in `trainable_targets` | Registered default model |

!!! note "Shared Parameters"
    Regardless of role differences, all agents share a single model instance (one set of parameters playing different roles).

### 2. Promote to An Advanced ASTuner Workflow

<div class="workflow-single">
<div class="workflow-header">Conversion Steps</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>Start with a basic AgentScope ReActAgent</strong>

```python
agent_instance = ReActAgent(
   name=f"Player-X",
   sys_prompt="You are a helpful assistant",
   model=DashScopeChatModel(model_name="qwen-max", stream=False),
   formatter=DashScopeChatFormatter(),
)
```
</li>
<li><strong>Register the agent role with model_tuner</strong>

```python
agent_role = "TYPE-ZERO"
default_model = DashScopeChatModel(model_name="qwen-max", stream=False)
model_tuner.register_model(agent_role, default_model=default_model)
```
</li>
<li><strong>Create ReActAgent linked to the role</strong>

```python
agent_instance = ReActAgent(
   name=f"Player-X",
   sys_prompt="You are a helpful assistant",
   model=model_tuner.get_model(agent_role),  # ← Bind to role
   formatter=DashScopeChatFormatter(),
)
```
</li>
<li><strong>Wrap in workflow class with trainable_targets</strong>

```python
class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"
    trainable_targets: list = ["TYPE-ZERO", ...]

    async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        # ... agents and workflow here ...
```
</li>
</ol>
</div>
</div>

### 3. Multi-Agent Example

Here's a complete example with multiple agent roles (Werewolves game):

```python title="Multi-Agent Workflow"
roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]
players = []

for i, role in enumerate(roles):
    # Define different default models for different roles
    default_model_for_good_guys = OpenAIChatModel(model_name="qwen-max", stream=False)
    default_model_for_bad_guys = OpenAIChatModel(model_name="qwen-plus", stream=False)
    
    chosen_model = (
        default_model_for_good_guys 
        if role != "werewolf" 
        else default_model_for_bad_guys
    )
    
    # Register role with its default model
    model_tuner.register_model(role, default_model=chosen_model)
    
    # Create agent bound to the role
    players += [ReActAgent(
        name=f"Player{i + 1}",
        sys_prompt=get_official_agent_prompt(f"Player{i + 1}"),
        model=model_tuner.get_model(role),
        formatter=OpenAIMultiAgentFormatter(),
    )]
```

!!! tip "Configuration Flexibility"
    In this example:
    
    - `role` describes an agent's in-game identity (werewolf, villager, etc.)
    - `chosen_model` defines the default model when the role is not being trained
    - You can flexibly switch training targets by modifying `trainable_targets`

---

## Next Steps

<div class="card-grid">
<a href="../data_pipeline/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>Data Pipeline</h3></div><p class="card-desc">Configure data loading from files, HuggingFace, or environments.</p></a>
<a href="../task_judger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>Task Judger</h3></div><p class="card-desc">Set up reward functions to evaluate agent performance.</p></a>
</div>
