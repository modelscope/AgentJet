from typing import List

from pydantic import BaseModel, Field

from agentscope_tuner import ModelTuner
from agentscope_tuner.schema.task import WorkflowOutput, WorkflowTask


class Workflow(BaseModel):
    model_config = {"extra": "allow"}
    name: str = Field(default="default_workflow", description="Name of the workflow.")
    trainable_targets: List[str] | None = Field(
        default=None,
        description="List of agents to be fine-tuned. When None, all agents are trainable.",
    )

    async def execute(self, workflow_task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        """Run the workflow on a given task."""
        raise NotImplementedError


"""
How to define a trainable workflow 🚀:

1. Single agent scenario 🤖:

    Simply set `model` argument to `model_tuner` when initializing your agent.
    This is a helpful example when you:
    - 🌟 Know exactly which agents should be trained, or the number of agents are small;
    - ✨ Already finished basic debugging of your workflow using a fixed model such as qwen-max;
    - 🎇 Do not requires changing which agents to be trained on the fly.

    ----- EXAMPLE -----

    - Suppose you have a react agent that looks like this:

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


    - Then all you have to do is to wrap it in a workflow class:

        [+] class ExampleMathLearn(Workflow):
        [+]    name: str = "math_agent_workflow"
        [+]    async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
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


2. Multi-agent scenario 🤝:

    Use `register_model` method of `ModelTuner` to register different agent targets.
    This is extremely helpful when you want to
    - 🌟 Achieve fine-grained control over which agents to be fine-tuned;
    - ✨ Define what model agents should use when they are NOT being tuned;
    - ⚡ Change which trainable agent targets on the fly without modifying the workflow code.

    ----- EXAMPLE -----

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



"""
