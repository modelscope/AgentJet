from astune import ModelTuner, Workflow, WorkflowTask, WorkflowOutput
from agentscope.message import Msg
from pydantic import BaseModel, Field
from loguru import logger


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


class FinalResult(BaseModel):
    result: str = Field(
        description=r"Your solution of the given math problem. Put your final answer in boxed format, e.g., \boxed{42}"
    )


system_prompt = """
You are an agent specialized in solving math problems with tools.
Please solve the math problem given to you.
You can write and execute Python code to perform calculation or verify your answer.
You should return your final answer within \\boxed{{}}.
"""


class ExampleMathLearn(Workflow):

    name: str = "math_agent_workflow"

    async def agentscope_execute(
        self, workflow_task: WorkflowTask, model_tuner: ModelTuner
    ) -> WorkflowOutput:
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
            model=model_tuner,
            formatter=DashScopeChatFormatter(),
            toolkit=self.toolkit,
            memory=InMemoryMemory(),
            max_iters=2,
        )
        self.agent.set_console_output_enabled(False)
        msg = Msg("user", query, role="user")
        result = await self.agent.reply(msg, structured_model=FinalResult)
        final_answer = extract_final_answer(result)
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})
