from typing import TYPE_CHECKING, Any, Literal, Type

from agentscope._utils._common import _create_tool_from_base_model
from agentscope.model import ChatModelBase, ChatResponse, DashScopeChatModel
from loguru import logger
from pydantic import BaseModel

from astuner.context_tracker.agentscope_tracker.multiagent_tracking import (
    MultiAgentContextTracking,
)
from astuner.task_rollout.async_llm_bridge import LlmProxyForAgentScope

if TYPE_CHECKING:
    from astuner import Workflow


class Agent2Proxy(DashScopeChatModel):
    """
    Handler for **NAMED** agent trainning targets.
    It stores the target name, and a reference to the ModelTuner.
    When request comes, it switches between default model (dashscope or openai models) and ModelTuner
    """

    def __init__(self, name: str, tuner: "ModelTuner", default_model: ChatModelBase):
        self.name = name
        self.tuner = tuner
        self.default_model = default_model
        super().__init__(
            model_name="astuner",
            api_key="dummy-api-key",
            stream=False,
        )

    def __call__(self, *args, **kwargs):
        if not self.tuner.is_trainable(self.name):
            # [DO-NOT-TRAIN] if `trainable_targets` is non-empty,
            # and self.name is not in it, use default model
            return self.default_model(*args, **kwargs)
        else:
            # [TRAIN]
            return self.tuner(*args, **kwargs)


class ModelTuner(DashScopeChatModel):
    """
    ModelTuner for Agentscope workflow.
    It keeps record of all registered agent types (by their target names),
    And when request comes, it calls `self.llm_proxy` to handle the request.
    """

    def __init__(
        self,
        config,
        context_tracker: MultiAgentContextTracking,
        agentscope_workflow: "Workflow",
        **kwargs,
    ) -> None:
        self.config = config
        self.context_tracker = context_tracker
        self.agentscope_workflow = agentscope_workflow
        self.target2proxy_registry: dict[str, Agent2Proxy] = {}
        self.llm_proxy = LlmProxyForAgentScope(
            context_tracker=context_tracker, config=config, **kwargs
        )
        super().__init__(
            model_name="astuner",
            api_key="dummy-api-key",
            stream=False,
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
            if (
                default_model.model_name
                != self.target2proxy_registry[target_name].default_model.model_name
            ):
                raise ValueError(
                    f"Agent proxy `{target_name}` is already registered with a different model_name.\nWAS [{self.target2proxy_registry[target_name].default_model.model_name}]\nNOW [{default_model.model_name}]."
                )
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

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "any", "required"] | str | None = None,
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

        if self.enable_thinking is not None and "enable_thinking" not in kwargs:
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

        # call llm model ✨
        response_gen = await self.llm_proxy(
            api_key=self.api_key,
            structured_model=structured_model,
            **kwargs,
        )

        # Return the AsyncGenerator directly
        return response_gen

    def is_trainable(self, target_name) -> bool:
        if not self.agentscope_workflow.trainable_targets:
            # always assume trainable when user has never changed trainable_targets
            return True
        if target_name in self.agentscope_workflow.trainable_targets:
            return True
        else:
            return False

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
