import asyncio
from typing import TYPE_CHECKING, Any, List, Callable, Literal, Type, Union
from loguru import logger
from pydantic import BaseModel
from ajet.context_tracker.agentscope_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)
from ajet.task_rollout.async_llm_bridge import OpenaiLlmProxyWithTracker
from ajet.utils.magic_mock import SpecialMagicMock
from openai.types.chat.chat_completion import ChatCompletion
from openai.resources.chat.chat import Chat, AsyncChat
from openai.resources.completions import AsyncCompletions
from openai import OpenAI, AsyncOpenAI

if TYPE_CHECKING:
    from ajet import Workflow

# import openai
# client = openai.OpenAI()

# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",  # You can replace this with "gpt-4" if available
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Hello! Tell me a joke about programming."}
#     ],
#     max_tokens=100,  # Limit the response length
#     temperature=0.7  # Control the randomness of the output
# )

class MockAsyncCompletions(AsyncCompletions):
    async def create(self, *args, **kwargs) -> Any: # type: ignore
        return await self._client.create(*args, **kwargs) # type: ignore

class MockAsyncChat(AsyncChat):
    @property
    def completions(self) -> MockAsyncCompletions:  # type: ignore
        return MockAsyncCompletions(self._client)

class OpenaiClientModelTuner(AsyncOpenAI):

    def __init__(
        self,
        config,
        context_tracker: MultiAgentContextTracker,
        workflow: "Workflow",
        agent_name: str,
        debug_model: str,
        use_debug_model: bool = False,
        **kwargs,
    ):
        self.llm_proxy = OpenaiLlmProxyWithTracker(
            context_tracker=context_tracker,
            config=config,
            **kwargs
        )

    @property
    def chat(self) -> MockAsyncChat:    # type: ignore
        return MockAsyncChat(self)

    async def create(
        self,
        messages: List[dict],
        tools: List = [],
        tool_choice: str = "auto",
        *args,
        **kwargs
    ) -> ChatCompletion:

        # call llm model ✨
        response_gen = await self.llm_proxy(
            messages = messages,
            tools = tools,
            tool_choice = tool_choice,
        )
        assert isinstance(response_gen, ChatCompletion)
        return response_gen


