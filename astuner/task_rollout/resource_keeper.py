from typing import Any, Dict, List, Tuple

from loguru import logger
from omegaconf import DictConfig

from astuner.schema.task import WorkflowTask
from astuner.utils.env_service_client.env_client_ng import EnvClient as EnvClientNg


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
        self.obs_window = self.workflow_task.obs_window
        if self.config.astuner.task_reader.type == "env_service":
            url = self.config.astuner.task_reader.env_service.env_url
            env_type = self.config.astuner.task_reader.env_service.env_type
            self.env = EnvClientNg(base_url=url)
            self.env_params = {}
            self.env_type: str = env_type
        else:
            self.env = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.env:
                self.env.release_instance(self.workflow_task.task_env_uuid)
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
            self.workflow_task.task_env_uuid,
            self.workflow_task.task_thread_index,
            self.workflow_task.obs_window,
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

        if self.config.astuner.task_reader.type == "env_service":
            if self.env is None:
                raise ValueError("Environment client is None but env_service type is specified")
            try:
                init_response = self.env.create_instance(
                    env_type=self.env_type,
                    task_id=self.task_id,
                    instance_id=self.workflow_task.task_env_uuid,
                    params=self.env_params,
                )
                state_message: dict = init_response["state"]
                _, init_messages = self._get_init_messages(state_message)
            except Exception as e:
                logger.bind(exception=True).exception(
                    f"encounter exception in env_worker.create_instance~ error={e.args}"
                )
                if self.env is not None:
                    self.env.release_instance(self.workflow_task.task_env_uuid)
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
        self, env_client: Any, task_env_uuid: str, task_thread_index: int, obs_window: Dict
    ) -> "BaseGymEnv":
        return BaseGymEnv(env_client, task_env_uuid, task_thread_index, obs_window)


class BaseGymEnv(object):
    """
    TODO: integrate with A.S. Runtime
    """

    def __init__(
        self, env_client: EnvClientNg, task_env_uuid: str, task_thread_index: int, obs_window: Dict
    ):
        self.env_client = env_client
        self.task_thread_index = task_thread_index
        self.obs_window = obs_window
        self.task_env_uuid = task_env_uuid

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

        self.obs_window["step"][self.task_thread_index] += 1
        env_output = self.env_client.step(
            instance_id=self.task_env_uuid,
            action=action,
        )
        obs = ""
        assert isinstance(env_output, dict)
        if ("content" not in env_output["state"]) and ("error" in env_output["state"]):
            obs = f"[Error from environment: {env_output['error']}]"
        elif env_output["state"]["content"] == "":
            obs = "Warning: the environment does not provide any feedback, please provide valid inpu and try again."
        else:
            obs = env_output["state"]["content"]
        reward = 0
        info = {}
        terminate = env_output["is_terminated"]
        return obs, reward, terminate, info

    def reset(self) -> str:
        """Reset gym environment."""
        raise RuntimeError("Reset is not supported")

    def evaluate(self, task_env_uuid, params):
        """Evaluate and get reward."""
        return self.env_client.evaluate(task_env_uuid, params)
