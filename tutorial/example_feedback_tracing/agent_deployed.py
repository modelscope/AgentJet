import os
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, execute_python_code
import agentscope


SYSTEM_PROMPT = """
You are an agent specialized in solving math problems with tools.
If I give problem, please solve the math problem given to you.
You can write and execute Python code to perform calculation or verify your answer.
You should return your final answer within \\boxed{{}}.
"""


def build_agent():
    tool_kit=Toolkit()
    tool_kit.register_tool_function(execute_python_code)

    agent = ReActAgent(
        name="Qwen",
        sys_prompt=SYSTEM_PROMPT,
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=True,
        ),
        formatter=DashScopeChatFormatter(),
        memory=InMemoryMemory(),
        toolkit=tool_kit,
        print_hint_msg=False,
    )

    return agent


async def main():
    # init the tracing module
    agentscope.init(studio_url="http://localhost:3000")

    agent = build_agent()

    while True:
        inp = input("User: ")
        print(await agent.reply(Msg("user", inp, role="user")))


if __name__=='__main__':
    import asyncio
    asyncio.run(main())