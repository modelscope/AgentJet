from agentscope.message import Msg
from loguru import logger

from agentscope_tuner import ModelTuner, Workflow, WorkflowOutput, WorkflowTask


def extract_final_answer(result) -> str:
    """Extract the final answer from the agent's response."""
    try:
        if (
            hasattr(result, "metadata")
            and isinstance(result.metadata, dict)
            and "result" in result.metadata
        ):
            return result.metadata["result"]
        if hasattr(result, "content"):
            if isinstance(result.content, dict) and "result" in result.content:
                return result.content["result"]
            return str(result.content)
        return str(result)
    except Exception as e:
        logger.warning(f"Extract final answer error: {e}. Raw: {result}")
        return str(result)


system_prompt = """
You are an agent specialized in solving countdown number puzzles.
Given a target number and a list of source numbers, find a way to reach the target number using basic arithmetic operations (+, -, *, /).
And each source number can only be used once.
Show your step-by-step calculation process.
You should return your final answer within \\boxed{{}}, for example \\boxed{{(1 + 2) * 3}}.
"""


class ExampleCountdownLearn(Workflow):
    name: str = "countdown_agent_workflow"

    async def execute(self, workflow_task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        from agentscope.agent import ReActAgent
        from agentscope.formatter import DashScopeChatFormatter
        from agentscope.memory import InMemoryMemory

        # Extract task information
        # print(workflow_task.task.main_query)
        query_data = workflow_task.task.metadata
        target = query_data.get("target")
        nums = query_data.get("nums")

        # Format the query
        nums_str = ", ".join(map(str, nums))  # type: ignore
        query = f"Target number: {target}\nAvailable numbers: {nums_str}\n\nPlease find a way to reach the target number using the available numbers."

        self.agent = ReActAgent(
            name="countdown_react_agent",
            sys_prompt=system_prompt,
            model=model_tuner,
            formatter=DashScopeChatFormatter(),
            memory=InMemoryMemory(),
            max_iters=2,
        )
        self.agent.set_console_output_enabled(False)

        # Execute agent
        msg = Msg("user", query, role="user")
        result = await self.agent.reply(msg)
        final_answer = extract_final_answer(result)

        return WorkflowOutput(
            reward=None,
            metadata={
                "final_answer": final_answer,
                "target": target,
                "nums": nums,
            },
        )
