from typing import Any, List, Callable
from ajet.context_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)
from ajet.task_rollout.async_llm_bridge import OpenaiLlmProxyWithTracker
from openai.types.chat.chat_completion import ChatCompletion
from openai.resources.chat.chat import AsyncChat
from openai.resources.completions import AsyncCompletions
from openai import AsyncOpenAI


class MockAsyncCompletions(AsyncCompletions):
    async def create(self, *args, **kwargs) -> Any: # type: ignore
        return await self._client.create(*args, **kwargs) # type: ignore

class MockAsyncChat(AsyncChat):
    @property
    def completions(self) -> MockAsyncCompletions:  # type: ignore
        return MockAsyncCompletions(self._client)

class OpenaiClientModelTuner(AsyncOpenAI):
    """ At this layer, we will determine which model to use:
        - training model
        - debug model assigned by user, used when this target is not being trained
    """
    def __init__(
        self,
        config,
        context_tracker: MultiAgentContextTracker,
        agent_name: str,
        debug_model: str | None = None,
        use_debug_model: bool = False,
        llm_inference_fn: Callable | None = None,
    ):
        self.debug_model = debug_model
        self.agent_name = agent_name
        self.use_debug_model = use_debug_model
        assert llm_inference_fn is not None, "llm_inference_fn must be provided"
        self.llm_proxy = OpenaiLlmProxyWithTracker(
            context_tracker=context_tracker,
            config=config,
            llm_inference_fn=llm_inference_fn,
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

        # route first
        if self.use_debug_model and (self.debug_model is not None):
            client = AsyncOpenAI()
            return await client.chat.completions.create(
                model = self.debug_model,
                messages = messages,    # type: ignore
                tools = tools,
                tool_choice = tool_choice, # type: ignore
            )

        # call llm model ✨
        response_gen = await self.llm_proxy(
            messages = messages,
            tools = tools,
            tool_choice = tool_choice,
        )
        assert isinstance(response_gen, ChatCompletion)
        return response_gen
