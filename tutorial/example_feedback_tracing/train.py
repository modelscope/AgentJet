from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.tool import Toolkit, execute_python_code
from loguru import logger
from pydantic import BaseModel, Field

from astuner import ModelTuner, Workflow, WorkflowOutput, WorkflowTask

SYSTEM_PROMPT = """
You are an agent specialized in solving math problems with tools.
If I give problem, please solve the math problem given to you.
You can write and execute Python code to perform calculation or verify your answer.
You should return your final answer within \\boxed{{}}.
"""


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
        description="Your solution of the given math problem. Put your final answer in boxed format, e.g., \\boxed{42}"
    )


class ExampleTracingFeedbackTrain(Workflow):
    name: str = "tracing_feedback_train"

    async def execute(
        self, workflow_task: WorkflowTask, model_tuner: ModelTuner
    ) -> WorkflowOutput:
        query = workflow_task.task.main_query

        tool_kit = Toolkit()
        tool_kit.register_tool_function(execute_python_code)

        agent = ReActAgent(
            name="Qwen",
            sys_prompt=SYSTEM_PROMPT,
            model=model_tuner,
            formatter=DashScopeChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=tool_kit,
            print_hint_msg=False,
        )

        msg = Msg("user", query, role="user")
        result = await agent.reply(msg, structured_model=FinalResult)
        final_answer = extract_final_answer(result)
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})
