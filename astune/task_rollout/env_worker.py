from loguru import logger
from typing import List, Union
from omegaconf import DictConfig
from astune.workflow_controller.classic_agentflow import AgentFlow
from astune.workflow_controller.classic_agentflow import BaseAgentFlow
from astune.workflow_controller.agentscope_flow import AgentScopeWorkflow
from astune.context_manager.agentflow_cm.cmt_linear import CMTLinear
from astune.env_service_client.env_client_ng import EnvClient as EnvClientNg

class EnvWorker(object):

    def __init__(self, task_core_arg, config: DictConfig):
        self.config = config

        if config.astune.task_reader.type == 'env_service':
            url = config.astune.task_reader.env_service.env_url
            env_type = config.astune.task_reader.env_service.env_type
            self.env = EnvClientNg(base_url=url)
            self.env_params = {}
            self.env_type: str = env_type
        else:
            self.env = None

        self.task_core_arg = task_core_arg
        self.task_id: str = task_core_arg.task_id
        self.tokenizer = task_core_arg.tokenizer
        self.llm_chat_fn = task_core_arg.llm_chat_fn
        self.obs_window = task_core_arg.obs_window


    def execute(self) -> CMTLinear:
        """Choose between classic `AgentFlow` controller and `AgentScopeWorkflow` controller
        """

        # >>>>>>>>>>>>>> create
        init_messages = self._initialize_environment_and_messages()

        # >>>>>>>>>>>>>> simulate
        try:
            if not self.config.astune.rollout.use_agentscope_protocol:
                agent_flow: BaseAgentFlow = AgentFlow(llm_chat_fn=self.llm_chat_fn, tokenizer=self.tokenizer, config=self.config)
            else:
                agent_flow: BaseAgentFlow = AgentScopeWorkflow(llm_chat_fn=self.llm_chat_fn, tokenizer=self.tokenizer, config=self.config)
            cmt = agent_flow.execute(
                init_messages=init_messages,
                env=self.env,   # type:ignore || self.env: Union[EnvClient, EnvClientNg]
                task_core_arg=self.task_core_arg
            )
        except Exception as e:
            logger.bind(exception=True).exception(f"encounter exception in env_worker.agent_flow~ error={e.args}")
            if self.env: self.env.release_instance(self.task_core_arg.task_env_uuid)
            raise e

        cmt.task_batch_index = self.task_core_arg.task_batch_index
        cmt.task_tag = self.task_core_arg.task_tag
        cmt.task_id = self.task_id

        # >>>>>>>>>>>>>> destory
        try:
            if self.env: self.env.release_instance(self.task_core_arg.task_env_uuid)
        except Exception as e:
            logger.bind(exception=True).exception(f"encounter exception in env_worker.release_instance~ error={e.args}")
            raise e

        return cmt


    def _initialize_environment_and_messages(self) -> List[dict]:
        """
        Initialize environment instance and setup initial messages.

        Returns:
            List[dict]: Initial messages for the agent flow

        Raises:
            Exception: If environment creation fails or required task data is missing
        """
        if self.config.astune.task_reader.type == 'env_service':
            if self.env is None:
                raise ValueError("Environment client is None but env_service type is specified")
            try:
                init_response = self.env.create_instance(
                    env_type=self.env_type,
                    task_id=self.task_id,
                    instance_id=self.task_core_arg.task_env_uuid,
                    params=self.env_params
                )
                state_message: dict = init_response["state"]
                _, init_messages = self._get_init_messages(state_message)
            except Exception as e:
                logger.bind(exception=True).exception(f"encounter exception in env_worker.create_instance~ error={e.args}")
                if self.env is not None:
                    self.env.release_instance(self.task_core_arg.task_env_uuid)
                raise e
        else:
            task = self.task_core_arg.task
            if task.init_messages:
                init_messages = task.init_messages
            else:
                assert task.main_query, "You must provide init_messages or main_query in task."
                init_messages = [{"role": "user", "content": task.main_query}]

        return init_messages


    def _get_init_messages(self, state_message) -> tuple:
        """
        Process state_message to extract query and init_messages.

        Args:
            state_message (Union[dict, list]): The state message to process

        Returns:
            tuple: (query, init_messages) where query is a string and init_messages is a list

        Raises:
            ValueError: If state_message is neither dict nor list
        """
        if isinstance(state_message, dict):
            query = state_message["content"]
            init_messages = [state_message]
        elif isinstance(state_message, list):
            assert isinstance(state_message[0], dict)
            query = state_message[-1]["content"]
            init_messages = state_message
        else:
            raise ValueError(f"state_message should be dict or list, but got {type(state_message)}")

        return query, init_messages
