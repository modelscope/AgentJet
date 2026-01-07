from typing import TYPE_CHECKING, Any, Literal, Type, Union

from loguru import logger

from ajet.context_tracker.agentscope_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)
from ajet.task_rollout.async_llm_bridge import OpenaiLlmProxyWithTracker

from ajet.tuner_lib.weight_tuner import AgentScopeModelTuner
from ajet.tuner_lib.weight_tuner import OpenaiClientModelTuner
if TYPE_CHECKING:
    from ajet import Workflow

TunerTypeUnion = Union[AgentScopeModelTuner, OpenaiClientModelTuner]

class TunerV2(object):

    def __init__(
        self,
        config,
        context_tracker: MultiAgentContextTracker,
        user_workflow: "Workflow",
        **kwargs,
    ) -> None:
        self.config = config
        self.workflow = user_workflow
        self.context_tracker = context_tracker
        self.target2proxy_registry: dict[str, TunerTypeUnion] = {}
        self.kwargs = kwargs


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
            user_workflow=self.workflow,
            agent_name=agent_name,
            debug_model=debug_model,
            use_debug_model=(not self.is_target_trainable(target_tag)),
            **self.kwargs,
        )
        self.register_model(target_tag, explicit_tuner_as_modelscope_model)
        return explicit_tuner_as_modelscope_model


    def as_raw_openai_sdk_client(
        self,
        agent_name="default_agent_name",
        target_tag="default_target_tag",
        debug_model='gpt-4o',
    ) -> OpenaiClientModelTuner:
        """Convert to raw OpenAI SDK client for advanced usage.
        Returns:
            Any:
                The raw OpenAI SDK client.
        """
        explicit_tuner_as_oai_client = OpenaiClientModelTuner(
            config=self.config,
            context_tracker=self.context_tracker,
            workflow=self.workflow,
            agent_name=agent_name,
            debug_model=debug_model,
            use_debug_model=(not self.is_target_trainable(target_tag)),
            **self.kwargs,
        )
        self.register_model(target_tag, explicit_tuner_as_oai_client)
        return explicit_tuner_as_oai_client


    def __call__(self, **kwargs):
        """This method is **deprecated**.
        The current behavior of this method is pretend as a agentscope model
        """
        # explicit_tuner = AgentScopeModelTuner(
        #     config=self.config,
        #     context_tracker=self.context_tracker,
        #     workflow=self.workflow,
        # )(**kwargs)
        # self.register_model("default_target_tag", explicit_tuner)
        # return explicit_tuner
        raise RuntimeError("This method is deprecated. Please use `as_agentscope_model` instead.")



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

    def get_llm_proxy(self) -> OpenaiLlmProxyWithTracker:
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