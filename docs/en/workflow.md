# Trainable Workflow

This tutorial introduces how to define a trainable workflow.

!!! info ""
    AgentJet provides two **convenient** and **mutually compatible** ways to wrap your Workflow:

    - **Simple**: Emphasizes simplicity, ease of use, and readability
    - **Advanced**: Emphasizes flexibility, controllability, and extensibility

In this article we use **AgentScope** framework for demonstration. For other frameworks (OpenAI SDK, Langchain, HTTP Requests), please follow the same pattern.



<div align="center">
<img width="800" alt="AgentJet" src="https://img.alicdn.com/imgextra/i2/O1CN01eygO6k1CzCvHnALzs_!!6000000000151-2-tps-1408-768.png"/>
</div>

## Simple Practice

!!! Example "Simple Practice Abstract"
    - Simply set `model` argument in AgentScope ReActAgent argument to `tuner.as_agentscope_model()` when initializing your agent.
    - Wrap your code with `class MyWorkflow(Workflow)` and your agent is ready to be tuned.

### 1. When to Use This Simple Practice

!!! warning "Choose Simple Practice If You..."
    - <img src="https://api.iconify.design/lucide:star.svg" class="inline-icon" /> Know exactly which agents should be trained, or the number of agents is small
    - <img src="https://api.iconify.design/lucide:sparkles.svg" class="inline-icon" /> Already finished basic debugging of your workflow
    - <img src="https://api.iconify.design/lucide:sparkle.svg" class="inline-icon" /> Do not need to change which agents are trained on the fly


### 2. Convert Your Workflow to AgentJet Trainable Workflow

The very first step is to create a class as a container to wrap your code:

=== "`converted_workflow.py` - AgentJet Workflow"

    ```python
    from ajet import AjetTuner, Workflow, WorkflowOutput, WorkflowTask
    class MyWorkflow(Workflow):
        async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:
            # ... your ReActAgent workflow here ✈️ ...
            return WorkflowOutput(reward=..., metadata={...})

    ```


Next, use the `tuner` argument, call its `tuner.as_agentscope_model()` method:

=== "Before"

    ```python
    model = DashScopeChatModel(model_name="qwen-max", stream=False)  # ✈️ change here
    agent_instance = ReActAgent(
       name=f"Friday",
       sys_prompt="You are a helpful assistant",
       model=model,
       formatter=DashScopeChatFormatter(),
    )
    ```

=== "After"

    ```python
    model = tuner.as_agentscope_model() # ✈️ change here
    agent_instance = ReActAgent(
       name=f"Friday",
       sys_prompt="You are a helpful assistant",
       model=model,
       formatter=DashScopeChatFormatter(),
    )
    ```

!!! warning "AjetTuner"
    `AjetTuner` also has `.as_raw_openai_sdk_client()` and `.as_oai_baseurl_apikey()` method. But `.as_agentscope_model()` is more convenient for AgentScope agent workflow.



### 3. Code Example

<div class="card-grid">
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent</h3></div><p class="card-desc">Training a math agent that can write Python code to solve mathematical problems.</p></a>
<a href="../example_learning_to_ask/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:comment-question.svg" class="card-icon card-icon-general" alt=""><h3>Learning to Ask</h3></div><p class="card-desc">Learning to ask questions like a doctor for medical consultation scenarios.</p></a>
<a href="../example_countdown/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:timer-sand.svg" class="card-icon card-icon-tool" alt=""><h3>Countdown Game</h3></div><p class="card-desc">Writing a countdown game using AgentScope and solving it with RL.</p></a>
<a href="../example_frozenlake/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:snowflake.svg" class="card-icon card-icon-data" alt=""><h3>Frozen Lake</h3></div><p class="card-desc">Solving a frozen lake walking puzzle using AgentJet's reinforcement learning.</p></a>
</div>





## Advanced Practice

!!! Example "Advanced Practice Abstract"
    - The `tuner.as_agentscope_model()` function has hidden parameters, please further complete them to tell AgentJet the identity of agents.
    - The `ajet.Workflow` class has hidden attribute `trainable_targets`, please assign it manually to narrow down agents to be tuned.

### 1. When to Use Advanced Practice

When designing a **multi-agent collaborative** workflow where each agent plays a different **target_tag**, AgentJet provides enhanced training and debugging capabilities.

!!! warning "Multi-Agent Benefits"
    With a multi-agent setup, you can:

    - <img src="https://api.iconify.design/lucide:star.svg" class="inline-icon" /> **Precisely control** which agents are fine-tuned
    - <img src="https://api.iconify.design/lucide:sparkles.svg" class="inline-icon" /> Explicitly define the default model for agents **not being trained**
    - <img src="https://api.iconify.design/lucide:zap.svg" class="inline-icon" /> Switch trainable targets on the fly **without modifying** source code

### 1. How to promote to advanced agent scenario:

Simple, there are only two more issues that should be take care of in addition:

i. **`.as_agentscope_model` has three hidden (optional) parameters, complete them for each agent.**

| parameter | explanation |
|----------|------------|
| `agent_name` | The name of this agent |
| `target_tag` | A tag that mark the agent category |
| `debug_model` | The model used when this agent is not being tuned |

=== "`as_agentscope_model()` parameters"

    ```python
    model_for_an_agent = tuner.as_agentscope_model(
        agent_name="AgentFriday",    # the name of this agent
        target_tag="Agent_Type_1",                # `target_tag in self.trainable_targets` means we train this agent, otherwise we do not train this agent.
        debug_model=OpenAIChatModel(
            model_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            stream=False,
            api_key="api_key",
        ),      # the model used when this agent is not in `self.trainable_targets`
    )
    ```

ii. **`Workflow` has a hidden (optional) attribute called `trainable_targets`, config it.**

| `trainable_targets` value | explanation |
|----------|------------|
| `trainable_targets = None` | All agents using `as_agentscope_model` will be trained |
| `trainable_targets = ["Agent_Type_1", "Agent_Type_2"]` | Agents with `target_tag=Agent_Type_1`, `target_tag=Agent_Type_2`, ... will be trained |
| `trainable_targets = []` | Illegal, no agents are trained |


| Scenario | Model Used |
|----------|------------|
| `target_tag` in `trainable_targets` | Trainable model |
| `target_tag` NOT in `trainable_targets` | Registered `debug_model` |



!!! warning
    Regardless of `target_tag` differences, all agents share a single model instance (one model weight to play different roles, the model receives different perceptions when playing different roles).


### 2. Multi-Agent Example

Here's a complete example with multiple agent roles (Werewolves game):

=== "`tutorial/example_werewolves/start.py`"
    ```python
    class ExampleWerewolves(Workflow):
        trainable_targets: List[str] | None = Field(default=["werewolf"], description="List of agents to be fine-tuned.")

        async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:

            # ensure trainable targets is legal
            assert self.trainable_targets is not None, "trainable_targets cannot be None in ExampleWerewolves (because we want to demonstrate a explicit multi-agent case)."

            # bad guys and good guys cannot be trained simultaneously
            # (because mix-cooperation-competition MARL needs too many advanced techniques to be displayed here)
            if "werewolf" in self.trainable_targets:
                assert len(self.trainable_targets) == 1, "Cannot train hostile roles simultaneously."
            else:
                assert len(self.trainable_targets) != 0, "No trainable targets specified."

            # make and shuffle roles (fix random seed for reproducibility)
            roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]
            task_id = workflow_task.task.metadata["random_number"]
            np.random.seed(int(task_id))
            np.random.shuffle(roles)

            # initialize agents
            players = []
            for i, role in enumerate(roles):
                default_model = OpenAIChatModel(
                    model_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
                    stream=False,
                    api_key="no_api_key",
                )
                model_for_this_agent = tuner.as_agentscope_model(
                    agent_name=f"Player{i + 1}",    # the name of this agent
                    target_tag=role,                # `target_tag in self.trainable_targets` means we train this agent, otherwise we do not train this agent.
                    debug_model=default_model,      # the model used when this agent is not in `self.trainable_targets`
                )
                agent = ReActAgent(
                    name=f"Player{i + 1}",
                    sys_prompt=get_official_agent_prompt(f"Player{i + 1}"),
                    model=model_for_this_agent,
                    formatter=DashScopeMultiAgentFormatter()
                        if role in self.trainable_targets
                        else OpenAIMultiAgentFormatter(),
                    max_iters=3 if role in self.trainable_targets else 5,
                )
                # agent.set_console_output_enabled(False)
                players += [agent]

            # reward condition
            try:
                good_guy_win = await werewolves_game(players, roles)
                raw_reward = 0
                is_success = False
                if (good_guy_win and self.trainable_targets[0] != "werewolf") or (
                    not good_guy_win and self.trainable_targets[0] == "werewolf"
                ):
                    raw_reward = 1
                    is_success = True
                logger.warning(f"Raw reward: {raw_reward}")
                logger.warning(f"Is success: {is_success}")
            except BadGuyException as e:
                logger.bind(exception=True).exception(
                    f"Error during game execution. Game cannot continue, whatever the cause, let's punish trainable agents  (Although they maybe innocent)."
                )
                raw_reward = -0.1
                is_success = False
            except Exception as e:
                logger.bind(exception=True).exception(
                    f"Error during game execution. Game cannot continue, whatever the cause, let's punish trainable agents  (Although they maybe innocent)."
                )
                raw_reward = -0.1
                is_success = False

            return WorkflowOutput(reward=raw_reward, is_success=is_success)
    ```

!!! tip "Configuration Flexibility"
    In this example:

    - `role` describes an agent's in-game identity (werewolf, villager, etc.)
    - `chosen_model` defines the default model when the role is not being trained
    - You can flexibly switch training targets by modifying `trainable_targets`


## TinkerScript

Wrapping and training your agent on a machine without GPU.

Working in progress and coming soon.


## Next Steps

<div class="card-grid">
<a href="../data_pipeline/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>Data Pipeline</h3></div><p class="card-desc">Configure data loading from files, HuggingFace, or environments.</p></a>
<a href="../task_judger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>Task Judger</h3></div><p class="card-desc">Set up reward functions to evaluate agent performance.</p></a>
</div>
