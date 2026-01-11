


# Type 1-1: AgentScope Agents

```python
class ExampleMathLearn(Workflow):

    async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:
        from agentscope.agent import ReActAgent
        from agentscope.formatter import DashScopeChatFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit, execute_python_code

        query = workflow_task.task.main_query
        self.toolkit = Toolkit()
        self.toolkit.register_tool_function(execute_python_code)
        self.agent = ReActAgent(
            name="math_react_agent",
            sys_prompt=system_prompt,
            model=tuner.as_agentscope_model(),  # 🌟 this will do the trick
            formatter=DashScopeChatFormatter(),
            toolkit=self.toolkit,
            memory=InMemoryMemory(),
            max_iters=2,
        )

        self.agent.set_console_output_enabled(False)
        msg = Msg("user", query, role="user")
        result = await self.agent.reply(msg)
        final_answer = extract_final_answer(result)
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})

```



# Type 1-2: AgentScope Agents: Triple-M (Multi-Role, Multi-Agent, Multi-Turn) Case

```python
roles = ["werewolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]
players = []
for i, agent_role in enumerate(roles):
    if agent_role != "werewolf":
        chosen_model_for_current_agent = OpenAIChatModel(model_name="qwen-max", stream=False)
    else:
        chosen_model_for_current_agent = OpenAIChatModel(model_name="qwen-plus", stream=False)
    players += [ReActAgent(
        name=f"Player{i + 1}",
        sys_prompt=get_official_agent_prompt(f"Player{i + 1}"),
        model=agentscope_model,
        model=tuner.as_agentscope_model(
            agent_name=f"Player{i + 1}",
            target_tag=agent_role,                          # 🌟 tag agents with their role
            debug_model=chosen_model_for_current_agent      # 🌟 assign a debug model, ONLY used when we are NOT training this agent
        )
        formatter=OpenAIMultiAgentFormatter(),
    )]
```



# Type 2: Raw OpenAI SDK Agents

```python

import openai
client = openai.OpenAI(api_key='dummy-api-key')

response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # You can replace this with "gpt-4" if available
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Tell me a joke about programming."}
    ],
    max_tokens=100,  # Limit the response length
    temperature=0.7  # Control the randomness of the output
)


```