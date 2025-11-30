
This tutorial introduces how to define a trainable workflow 🚀 with AgentScope.

ASTuner offers two convenient and mutually compatible encapsulation methods for AgentScope Workflow. The first emphasizes **simplicity, convenience, and ease of understanding**, while the second focuses on **flexibility, controllability, and extensibility**.

## Simple agent scenario ✨:


1. Converting your AgentScope Workflow in ASTuner.

**Simply set ReActAgent's `model` argument to `model_tuner` when initializing your agent.**

```python

# From >>>
agent_instance = ReActAgent(
   name=f"Friday",
   sys_prompt="You are a helpful assistant",
   model=DashScopeChatModel(model_name="qwen-max", stream=False),
   formatter=DashScopeChatFormatter(),
)

# To <<<
agent_instance = ReActAgent(
   name=f"Friday",
   sys_prompt="You are a helpful assistant",
   model=model_tuner,	# change here
   formatter=DashScopeChatFormatter(),
)
```

Then, wrap your workflow in a class that inherit `Workflow` (`from astnue import Workflow`), and your workflow is ready to be tuned.

```
class ExampleMathLearn(Workflow):
	name: str = "math_agent_workflow"
	async def agentscope_execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
		... your ReActAgent workflow here ...
		return WorkflowOutput(reward=workflow_reward)

```


2. When to use the simple agent practice.

This practice suits well for most users, you can choose to follow this practice if you:

- 🌟 Know exactly which agents should be trained, or the number of agents are small;
- ✨ Already finished basic debugging of your workflow, confirming that your workflow works well when implemented with a non-tuned model such as `qwen-max`;
- 🎇 Do not requires changing which agents to be trained on the fly.


3. Code Example

    - Suppose you have built a react agent that looks like this:

		```
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

    - Then all you have to do is to wrap it in a workflow class:

		```
        [+] class ExampleMathLearn(Workflow):
        [+]    name: str = "math_agent_workflow"
        [+]    async def agentscope_execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        [ ]       from agentscope.agent import ReActAgent
        [ ]       from agentscope.formatter import DashScopeChatFormatter
        [ ]       from agentscope.memory import InMemoryMemory
        [ ]       from agentscope.tool import Toolkit, execute_python_code
        [ ]       self.toolkit = Toolkit()
        [ ]       self.toolkit.register_tool_function(execute_python_code)
        [ ]       self.agent = ReActAgent(
        [ ]           name="math_react_agent",
        [ ]           sys_prompt=system_prompt,
        [-]           model=DashScopeChatModel(model='qwen-max'),
        [+]           model=model_tuner,
        [ ]           formatter=DashScopeChatFormatter(),
        [ ]           toolkit=self.toolkit,
        [ ]           memory=InMemoryMemory(),
        [ ]       )
        [+]        query = task.task.main_query
        [ ]       msg = Msg("user", query, role="user")
        [ ]       result = await self.agent.reply(msg, structured_model=FinalResult)
        [ ]       final_answer = extract_final_answer(result)
        [+]       return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})
		```



## Advanced agent scenario 🤝:

When designing an advanced multi-agent workflow composed by agents with different roles,
ASTuner can work better if it knows the identity of each agent,
thus providing training and debugging convinience.

1. Promoting to advanced ASTuner workflow.

The basic idea is to use `ModelTuner.register_model` to register different agent targets.

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

- Wrap workflow in class and define `trainable_tragets`

```
class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"
	trainable_targets: list = ["TYPE-ZERO", ...]
	async def agentscope_execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
		... your agents and workflow here ...
```

2. When to apply this advanced ASTuner workflow rather than the simple ASTuner workflow.

It is recommanded if you need to:
- 🌟 Achieve fine-grained control over which agents to be fine-tuned;
- ✨ Define what model agents should use when they are NOT being tuned;
- ⚡ Change which trainable agent targets on the fly without modifying the workflow code.


2. An multi-agent example:

	```
    [ ]   roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]
    [ ]   players = []
    [ ]   for i, role in enumerate(roles):
    [ ]       default_model_for_good_guys = OpenAIChatModel(model_name="qwen-max", stream=False)
    [ ]       default_model_for_bad_guys = OpenAIChatModel(model_name="qwen-plus", stream=False)
    [ ]       chosen_model = default_model_for_good_guys if role != "werewolf" else default_model_for_bad_guys  # 🌟
    [ ]       players += [ReActAgent(
    [ ]           name=f"Player{i + 1}",
    [ ]           sys_prompt=get_official_agent_prompt(f"Player{i + 1}"),
    [-]           model=chosen_model,
    [+]           model=model_tuner.register_model(role, default_model=chosen_model),
    [ ]           formatter=OpenAIMultiAgentFormatter(),
    [ ]       )]
	```
