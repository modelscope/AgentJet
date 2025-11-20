from litellm import Type
from loguru import logger
from typing import Literal, Any
from pydantic import BaseModel, Field
from astune.utils.dynamic_import import dynamic_import
from astune.task_rollout.async_llm_bridge import LlmProxyForAgentScope
from astune.context_tracker.agentscope_tracker.multiagent_tracking import MultiAgentContextTracking
from agentscope.model import ChatModelBase, ChatResponse, DashScopeChatModel
from agentscope._utils._common import _json_loads_with_repair, _create_tool_from_base_model


class Agent2Proxy(DashScopeChatModel):
    def __init__(self, name: str, proxy, default_model: ChatModelBase):
        self.name = name
        self.proxy = proxy
        self.default_model = default_model

    def __call__(self, *args, **kwargs):
        if self.name not in self.proxy.get_trainable_targets():
            # [DO-NOT-TRAIN] if `trainable_targets` is non-empty,
            # and self.name is not in it, use default model
            return self.default_model(*args, **kwargs)
        else:
            # [TRAIN]
            return self.proxy(*args, **kwargs)


class ModelTuner(DashScopeChatModel):

    def __init__(self, config, context_tracker, **kwargs) -> None:
        self.config = config
        self.context_tracker = context_tracker
        self.target2proxy_registry: dict[str, Agent2Proxy] = {}
        self.llm_proxy = LlmProxyForAgentScope(context_tracker=context_tracker, **kwargs)
        super().__init__(
            model_name='astune',
            api_key='dummy-api-key'
        )


    def register_model(self, target_name: str, default_model: ChatModelBase) -> Agent2Proxy:
        """Register an agent type.
        Args:
            target_name (`str`):
                The name to register the agent type under.
            default_model (`ChatModelBase`):
                The model to use when you are NOT training this agent type.
        Returns:
            Agent2Proxy:
                The agent type instance corresponding to the provided name.
        """
        if target_name in self.target2proxy_registry:
            logger.warning(f"Agent proxy `{target_name}` is already registered. Overwriting `default_model`.")
        self.target2proxy_registry[target_name] = Agent2Proxy(target_name, self, default_model)
        return self.get_model(target_name)


    def get_model(self, target_name: str) -> Agent2Proxy:
        """Get the proxy instance by target_name.
        Args:
            target_name (`str`):
                The name of the agent proxy to retrieve.
        Returns:
            Agent2Proxy:
                The agent proxy corresponding to the provided target_name.
        """
        if target_name not in self.target2proxy_registry:
            raise ValueError(f"Agent proxy '{target_name}' is not registered.")
        else:
            return self.target2proxy_registry[target_name]


    def get_llm_proxy(self) -> LlmProxyForAgentScope:
        """Get the LlmProxyForAgentScope instance.
        Returns:
            LlmProxyForAgentScope:
                The LlmProxyForAgentScope instance used by the ModelTuner.
        """
        return self.llm_proxy


    def get_context_tracker(self) -> MultiAgentContextTracking:
        """Get the context tracker instance.
        Returns:
            LlmProxyForAgentScope:
                The context tracker instance used by the ModelTuner.
        """
        return self.context_tracker


    async def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "any", "required"]
        | str
        | None = None,
        structured_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:

        # For qvq and qwen-vl models, the content field cannot be `None` or
        # `[{"text": None}]`, so we need to convert it to an empty list.
        if self.model_name.startswith("qvq") or "-vl" in self.model_name:
            raise NotImplementedError("Not implemented for qvq and qwen-vl models yet.")

        kwargs = {
            "messages": messages,
            "model": self.model_name,
            "stream": self.stream,
            **self.generate_kwargs,
            **kwargs,
            "result_format": "message",
            # In agentscope, the `incremental_output` must be `True` when
            # `self.stream` is True
            "incremental_output": self.stream,
        }

        if tools:
            kwargs["tools"] = self._format_tools_json_schemas(tools)

        if tool_choice:
            self._validate_tool_choice(tool_choice, tools)
            kwargs["tool_choice"] = self._format_tool_choice(tool_choice)

        if (
            self.enable_thinking is not None
            and "enable_thinking" not in kwargs
        ):
            kwargs["enable_thinking"] = self.enable_thinking

        if structured_model:
            if tools or tool_choice:
                logger.warning(
                    "structured_model is provided. Both 'tools' and "
                    "'tool_choice' parameters will be overridden and "
                    "ignored. The model will only perform structured output "
                    "generation without calling any other tools.",
                )
            format_tool = _create_tool_from_base_model(structured_model)
            kwargs["tools"] = self._format_tools_json_schemas(
                [format_tool],
            )
            kwargs["tool_choice"] = self._format_tool_choice(
                format_tool["function"]["name"],
            )

        # call llm model
        response_gen = await self.llm_proxy(
            api_key=self.api_key,
            structured_model=structured_model,
            **kwargs,
        )

        # Return the AsyncGenerator directly
        return response_gen
