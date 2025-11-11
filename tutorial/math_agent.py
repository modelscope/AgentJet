from astune.agentscope_flow import ASTuneProxy
from agentscope.message import Msg
from pydantic import BaseModel, Field
from astune.protocol.agentscope_protocol import AgentScopeLearnProtocol
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
        description="Your solution of the given math problem. Put your final answer in boxed format, e.g., \\boxed{42}"
    )

system_prompt = """
You are an agent specialized in solving math problems with tools. Please solve the math problem given to you. You can write and execute Python code to perform calculation or verify your answer. You should return your final answer within \\boxed{{}}.
"""

class ExampleMathLearn(AgentScopeLearnProtocol):

    trainer: str = Field(default="agentscorpion-trinity")

    async def agentscope_execute(self, init_messages, astune_proxy: ASTuneProxy, config):
        from agentscope.agent import ReActAgent
        from agentscope.formatter import DashScopeChatFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.agent import ReActAgent
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit, execute_python_code

        if len(init_messages) >= 2: first_msg, init_messages = init_messages[0], init_messages[1:]
        else: first_msg = {"content": "You're a helpful assistant."}
        interaction_message = []
        for msg in init_messages:
            interaction_message.append(Msg(name=msg.get("name", "user"), content=msg.get("content", ""), role=msg.get("role", "user")))

        self.toolkit = Toolkit()
        self.toolkit.register_tool_function(execute_python_code)
        self.agent = ReActAgent(
            name="math_react_agent",
            sys_prompt=system_prompt,
            model=astune_proxy,  # type: ignore
            formatter=DashScopeChatFormatter(),
            toolkit=self.toolkit,
            memory=InMemoryMemory(),
        )
        msg = Msg("user", init_messages[0]['content'], role="user")
        result = await self.agent.reply(msg, structured_model=FinalResult)
        final_answer = extract_final_answer(result)
        astune_proxy.update_judge_input_dictionary(final_answer=final_answer)

        return astune_proxy

