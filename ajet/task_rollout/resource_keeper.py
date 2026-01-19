from typing import Any, Dict, List, Tuple

from loguru import logger
from omegaconf import DictConfig

from ajet.schema.task import WorkflowTask
from ajet.utils.env_service_client.env_client_ng import (
    EnvClient as EnvClientNg,
)


class ResourceKeeper(object):
    """
    TODO: integrate with A.S. Runtime
    """

    def __init__(self, workflow_task: WorkflowTask, config: DictConfig):
        self.workflow_task = workflow_task
        self.config = config

    def __enter__(self):
        self.config = self.config
        self.workflow_task = self.workflow_task
        self.task_id: str = self.workflow_task.task_id
        self.tokenizer = self.workflow_task.tokenizer
        self.llm_inference_fn = self.workflow_task.llm_inference_fn
        self.observation_window = self.workflow_task.observation_window
        if self.config.ajet.task_reader.type in ("env_service", "jsonl_with_env_service"):
            url = self.config.ajet.task_reader.env_service.env_url
            env_type = self.config.ajet.task_reader.env_service.env_type
            self.env = EnvClientNg(base_url=url)
            self.env_params = {}
            self.env_type: str = env_type
        else:
            self.env = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.env:
                self.env.release_instance(self.workflow_task.episode_uuid)
        except Exception as e:
            logger.bind(exception=True).exception(
                f"encounter exception in env_worker.release_instance~ error={e.args}"
            )
            raise e

    def prepare(self):
        """
        Prepare the environment and initial messages for the workflow task.

        Returns:
            WorkflowTask: The updated workflow task with initialized environment and messages.
        """
        init_messages = self._initialize_environment_and_messages()
        self.workflow_task.task.init_messages = init_messages
        self.workflow_task.gym_env = self.generate_gym_env(
            self.env,
            self.workflow_task.episode_uuid,
            self.workflow_task.task_thread_index,
            self.workflow_task.observation_window,
        )

        return self.workflow_task

    def _initialize_environment_and_messages(self) -> List[dict]:
        """
        Initialize environment instance and setup initial messages.

        Returns:
            List[dict]: Initial messages for the agent flow

        Raises:
            Exception: If environment creation fails or required task data is missing
        """

        reader_type = self.config.ajet.task_reader.type

        if reader_type == "env_service":
            if self.env is None:
                raise ValueError("Environment client is None but env_service type is specified")
            try:
                init_response = self.env.create_instance(
                    env_type=self.env_type,
                    task_id=self.task_id,
                    instance_id=self.workflow_task.episode_uuid,
                    params=self.env_params,
                )
                state_message: dict = init_response["state"]
                query, init_messages = self._get_init_messages(state_message)
                # Update main_query with actual query from environment
                self.workflow_task.task.main_query = query
            except Exception as e:
                logger.bind(exception=True).exception(
                    f"encounter exception in env_worker.create_instance~ error={e.args}"
                )
                if self.env is not None:
                    self.env.release_instance(self.workflow_task.episode_uuid)
                raise e
        elif reader_type == "jsonl_with_env_service":
            # 新逻辑：调用 create_instance 注册实例，但使用 jsonl 中的 init_messages
            if self.env is None:
                raise ValueError("Environment client is None but jsonl_with_env_service type is specified")
            try:
                # 必须调用 create_instance，让服务端创建实例，后续 step() 才能工作
                self.env.create_instance(
                    env_type=self.env_type,
                    task_id=self.task_id,
                    instance_id=self.workflow_task.episode_uuid,
                    params=self.env_params,
                )
                # 不使用返回的 state，直接用 jsonl 中加载的 init_messages
                task = self.workflow_task.task
                if task.init_messages:
                    init_messages = task.init_messages
                else:
                    assert task.main_query, "jsonl_with_env_service requires init_messages or main_query in jsonl file."
                    init_messages = [{"role": "user", "content": task.main_query}]
            except Exception as e:
                logger.bind(exception=True).exception(
                    f"encounter exception in env_worker.create_instance~ error={e.args}"
                )
                if self.env is not None:
                    self.env.release_instance(self.workflow_task.episode_uuid)
                raise e
        else:
            task = self.workflow_task.task
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

    def generate_gym_env(
        self, env_client: Any, episode_uuid: str, task_thread_index: int, observation_window: Dict
    ) -> "BaseGymEnv":
        return BaseGymEnv(env_client, episode_uuid, task_thread_index, observation_window)


class BaseGymEnv(object):
    """
    TODO: integrate with A.S. Runtime
    """

    def __init__(
        self,
        env_client: EnvClientNg,
        episode_uuid: str,
        task_thread_index: int,
        observation_window: Dict,
    ):
        self.env_client = env_client
        self.task_thread_index = task_thread_index
        self.observation_window = observation_window
        self.episode_uuid = episode_uuid
        if self.env_client:
            self.service_url = self.env_client.base_url

    def step(self, action: dict) -> Tuple[str, float, bool, dict]:
        """Take a step in the gym environment."""
        if not isinstance(action["content"], str):
            # assert isinstance(action['content'], list)
            # assert len(action['content']) == 1
            # assert isinstance(action['content'][0], dict)
            # assert 'type' in action['content'][0]
            # assert 'text' in action['content'][0]
            try:
                action["content"] = action["content"][0]["text"]
            except Exception:
                logger.exception(
                    f"Failed to parse action content from agentscope output. {action['content']}"
                )
                action["content"] = str(action["content"])

        self.observation_window["step"][self.task_thread_index] += 1
        env_output = self.env_client.step(
            instance_id=self.episode_uuid,
            action=action,
        )
        obs = ""
        assert isinstance(env_output, dict)

        if isinstance(env_output["state"], list):
            # 1. If state is a list (new standard format), pass through directly
            obs = env_output["state"]
        else:
            # 2. If state is a dict (old format or error)
            if ("content" not in env_output["state"]) and ("error" in env_output["state"]):
                obs = f"[Error from environment: {env_output['error']}]"
            elif env_output["state"].get("content", "") == "":
                obs = "Warning: the environment does not provide any feedback, please provide valid inpu and try again."
            else:
                obs = env_output["state"]["content"]

        reward = 0
        info = {}
        terminate = env_output["is_terminated"]
        return obs, reward, terminate, info # type: ignore

    def reset(self) -> str:
        """Reset gym environment."""
        raise RuntimeError("Reset is not supported")

    def evaluate(self, episode_uuid, params):
        """Evaluate and get reward."""
        return self.env_client.evaluate(episode_uuid, params)
