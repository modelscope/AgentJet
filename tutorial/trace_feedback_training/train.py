from astune import ModelTuner, Workflow, WorkflowTask, WorkflowOutput
from agentscope.message import Msg
from pydantic import BaseModel, Field
from loguru import logger

from agent_deployed import build_agent

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

    async def agentscope_execute(self, workflow_task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        query = workflow_task.task.main_query
        agent=build_agent()
        
        msg = Msg("user", query, role="user")
        result = await agent.reply(msg, structured_model=FinalResult)
        final_answer = extract_final_answer(result)
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})

