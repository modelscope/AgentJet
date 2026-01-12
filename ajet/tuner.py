from typing import TYPE_CHECKING, Any, Literal, Callable, Union

from ajet.context_tracker.multiagent_tracking import (
    MultiAgentContextTracker,
)

from ajet.tuner_lib.weight_tuner import AgentScopeModelTuner
from ajet.tuner_lib.weight_tuner import OpenaiClientModelTuner
from ajet.tuner_lib.weight_tuner.as_oai_baseurl_apikey import OpenaiClientBaseUrlTuner
if TYPE_CHECKING:
    from ajet import Workflow

TunerTypeUnion = Union[AgentScopeModelTuner, OpenaiClientModelTuner]

class AjetTuner(object):

    def __init__(
        self,
        config,
        context_tracker: MultiAgentContextTracker,
        user_workflow: "Workflow",
        llm_inference_fn: Callable,
    ) -> None:
        self.config = config
        self.workflow = user_workflow
        self.context_tracker = context_tracker
        self.llm_inference_fn = llm_inference_fn
        self.target2proxy_registry: dict[str, dict[str,TunerTypeUnion]] = {}
        if config.ajet.enable_experimental_reverse_proxy:
            self._enable_experimental_interchange_server(llm_inference_fn)


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
            use_debug_model=(not self._is_target_trainable(target_tag)),
            llm_inference_fn=self.llm_inference_fn,
        )
        self._register(target_tag, agent_name, explicit_tuner_as_modelscope_model)
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
            use_debug_model=(not self._is_target_trainable(target_tag)),
            llm_inference_fn=self.llm_inference_fn,
        )
        self._register(target_tag, agent_name, explicit_tuner_as_oai_client)
        return explicit_tuner_as_oai_client


    def as_oai_baseurl_apikey(
        self,
        agent_name="default_agent_name",
        target_tag="default_target_tag",
    ):
        """
        Usage:
            ```python
            result = tuner.as_oai_baseurl_apikey()

            # take base_url, api_key, model_name
            base_url = result.base_url
            api_key = result.api_key

            # use base_url, api_key, model_name
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            response = await client.chat.completions.create(
                model='whatever_model_name_you_like',
                messages=messages,
            )
            ```
        """

        assert self.config.ajet.enable_experimental_reverse_proxy, "Please enable `ajet.enable_experimental_reverse_proxy` in yaml config to use `as_oai_baseurl_apikey` feature."
        baseurl_apikey_model = OpenaiClientBaseUrlTuner(
            config=self.config,
            context_tracker=self.context_tracker,
            workflow=self.workflow,
            agent_name=agent_name,
            target_tag=target_tag,
            episode_uuid=self.context_tracker.episode_uuid,
        )
        return baseurl_apikey_model

    def __call__(self, **kwargs):
        """This method is **deprecated**.
        The current behavior of this method is pretend as a agentscope model
        """
        raise RuntimeError("This method is deprecated. Please use `as_agentscope_model` / `as_raw_openai_sdk_client` first.")


    # ------------------------------------------------------------------------
    # other helper methods
    # ------------------------------------------------------------------------

    def _register(self, target_name: str, agent_name: str, explicit_tuner: TunerTypeUnion) -> TunerTypeUnion:
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
        if target_name not in self.target2proxy_registry:
            self.target2proxy_registry[target_name] = {}
        self.target2proxy_registry[target_name][agent_name] = explicit_tuner
        return explicit_tuner

    def _is_target_trainable(self, target_name) -> bool:
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


    def get_context_tracker(self) -> MultiAgentContextTracker:
        """Get the context tracker instance.
        Returns:
            LlmProxyForAgentScope:
                The context tracker instance used by the ModelTuner.
        """
        return self.context_tracker


    def _enable_experimental_interchange_server(self, llm_inference_fn):
        # experimental reverse proxy start
        if self.config.ajet.enable_experimental_reverse_proxy:
            from ajet.tuner_lib.weight_tuner.experimental.as_oai_model_client import InterchangeClient
            self.interchange_client = InterchangeClient(
                episode_uuid=self.context_tracker.episode_uuid,
                context_tracker=self.context_tracker,
                config=self.config,
                llm_inference_fn=llm_inference_fn,
            )


    def terminate_episode(self):
        # experimental reverse proxy cleanup
        if self.config.ajet.enable_experimental_reverse_proxy:
            self.interchange_client._should_terminate = True
