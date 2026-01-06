from typing import TYPE_CHECKING, Any, Literal, Type, Union

from agentscope._utils._common import _create_tool_from_base_model
from agentscope.model import ChatModelBase, ChatResponse, DashScopeChatModel
from loguru import logger
from pydantic import BaseModel

from ajet.context_tracker.agentscope_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)
from ajet.task_rollout.async_llm_bridge import AgentScopeLlmProxy
from ajet.tuner_lib.weight_tuner import as_agentscope_model, as_oai_sdk_model
from ajet.utils.magic_mock import SpecialMagicMock

if TYPE_CHECKING:
    from ajet import Workflow
    from openai.types.chat.chat_completion import ChatCompletion
    from ajet.tuner_lib.weight_tuner.ajet import AgentScopeModelTuner



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
            model_name="ajet",
            api_key="dummy-api-key",
            stream=False,
        )

    def __call__(self, *args, **kwargs):
        if not self.tuner.is_target_trainable(self.name):
            # [DO-NOT-TRAIN] if `trainable_targets` is non-empty,
            # and self.name is not in it, use default model
            return self.default_model(*args, **kwargs)
        else:
            # [TRAIN]
            return self.tuner(*args, **kwargs)




class OpenaiClientModelTuner:

    def __init__(self):
        # Create a custom callable object for chat.completions.create
        self.chat = SpecialMagicMock(allowed_attributes="completions")  # Empty object for `chat`
        self.chat.completions = SpecialMagicMock(allowed_attributes="create")  # Empty object for `completions`
        self.chat.completions.create = self.create  # Redirect to create

    def create(
        self,
        *,
        messages,
        model,
        audio,
        frequency_penalty,
        function_call,
        functions,
        logit_bias,
        logprobs,
        max_completion_tokens,
        max_tokens,
        metadata,
        modalities,
        n,
        parallel_tool_calls,
        prediction,
        presence_penalty,
        reasoning_effort,
        response_format,
        seed,
        service_tier,
        stop,
        store,
        stream,
        stream_options,
        temperature,
        tool_choice,
        tools,
        top_logprobs,
        top_p,
        user,
        web_search_options,
        extra_headers,
        extra_query,
        extra_body,
        timeout,
    ) -> ChatCompletion:
        # call llm model ✨
        response_gen = await self.llm_proxy(
            api_key=self.api_key,
            structured_model=structured_model,
            **kwargs,
        )

TunerTypeUnion = Union[Type["TunerV2"], Type["AgentScopeModelTuner"], Type["OpenaiClientModelTuner"]]

class TunerV2(BaseModel):

    def __init__(
        self,
        config,
        context_tracker: MultiAgentContextTracker,
        workflow: "Workflow",
        **kwargs,
    ) -> None:
        self.config = config
        self.workflow = workflow
        self.context_tracker = context_tracker
        self.target2proxy_registry: dict[str, TunerTypeUnion] = {}


    def as_agentscope_model(
        self,
        agent_name="default_agent_name",
        target_tag="default_target_tag",
        debug_model=None
    ) -> "AgentScopeModelTuner":
        """Convert to ModelTuner instance for Agentscope workflow.
        Returns:
            ModelTuner:
                The ModelTuner instance for Agentscope workflow.
        """
        explicit_tuner_as_modelscope_model = AgentScopeModelTuner(
            config=self.config,
            context_tracker=self.context_tracker,
            workflow=self.workflow,
            agent_name=agent_name,
            debug_model=debug_model,
            use_debug_model=(not self.is_target_trainable(target_tag)),
        )
        self.register_model(target_tag, explicit_tuner_as_modelscope_model)
        return explicit_tuner_as_modelscope_model


    def as_raw_openai_sdk_client(
        self,
        agent_name="default_agent_name",
        target_tag="default_target_tag",
        debug_model=None
    ) -> Any:
        """Convert to raw OpenAI SDK client for advanced usage.
        Returns:
            Any:
                The raw OpenAI SDK client.
        """
        return OpenaiClientModelTuner(

        )


    def __call__(self, **kwargs):
        """This method is **deprecated**.
        The current behavior of this method is pretend as a agentscope model
        """
        explicit_tuner = AgentScopeModelTuner(
            config=self.config,
            context_tracker=self.context_tracker,
            workflow=self.workflow,
        )(**kwargs)
        self.register_model("default_target_tag", explicit_tuner)
        return explicit_tuner



    # ------------------------------------------------------------------------
    # other helper methods
    # ------------------------------------------------------------------------

    def register_model(self, target_name: str, explicit_tuner: TunerTypeUnion) -> TunerTypeUnion:
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
        self.target2proxy_registry[target_name] = explicit_tuner
        return explicit_tuner

    def is_target_trainable(self, target_name) -> bool:
        """Determine whether user have used `trainable_targets` to explicitly control training targets.
        """
        if self.workflow.trainable_targets is None:
            # always assume trainable when user has never changed trainable_targets
            return True
        if not self.workflow.trainable_targets:
            # always assume trainable when trainable_targets is []
            return True
        if target_name in self.workflow.trainable_targets:
            return True
        else:
            return False

    def get_llm_proxy(self) -> AgentScopeLlmProxy:
        """Get the LlmProxyForAgentScope instance.
        Returns:
            LlmProxyForAgentScope:
                The LlmProxyForAgentScope instance used by the ModelTuner.
        """
        return self.llm_proxy

    def get_context_tracker(self) -> MultiAgentContextTracker:
        """Get the context tracker instance.
        Returns:
            LlmProxyForAgentScope:
                The context tracker instance used by the ModelTuner.
        """
        return self.context_tracker