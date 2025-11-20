from loguru import logger
from typing import List, Union
from omegaconf import DictConfig
from recipe.sppo import config
from astune.task_runner.classic_runner import AgentRunner, BaseAgentRunner
from astune.task_runner.agentscope_runner import AgentScopeRunner
from astune.context_tracker.basic_tracker import BasicContextTracker
from astune.utils.env_service_client.env_client_ng import EnvClient as EnvClientNg

class ResourceKeeper(object):

    def __init__(self, task_core_arg, config: DictConfig):
        self.task_core_arg = task_core_arg
        self.config = config

    def __enter__(self):
        self.config = self.config
        self.task_core_arg = self.task_core_arg
        self.task_id: str = self.task_core_arg.task_id
        self.tokenizer = self.task_core_arg.tokenizer
        self.llm_chat_fn = self.task_core_arg.llm_chat_fn
        self.obs_window = self.task_core_arg.obs_window
        if self.config.astune.task_reader.type == 'env_service':
            url = self.config.astune.task_reader.env_service.env_url
            env_type = self.config.astune.task_reader.env_service.env_type
            self.env = EnvClientNg(base_url=url)
            self.env_params = {}
            self.env_type: str = env_type
        else:
            self.env = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.env: self.env.release_instance(self.task_core_arg.task_env_uuid)
        except Exception as e:
            logger.bind(exception=True).exception(f"encounter exception in env_worker.release_instance~ error={e.args}")
            raise e
