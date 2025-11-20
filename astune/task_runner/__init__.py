from astune.context_tracker.basic_tracker import BasicContextTracker
from typing import Any, Dict, Tuple, Union, Callable
from astune.task_judge.judge_base import JudgeBase


class BaseGymEnv(object):
    def __init__(
        self, env_client: Any, task_env_uuid: str, task_thread_index: int, obs_window: Dict
    ):
        self.env_client = env_client
        self.task_thread_index = task_thread_index
        self.obs_window = obs_window
        self.task_env_uuid = task_env_uuid

    def step(self, action: dict) -> Tuple[str, float, bool, dict]:
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
        raise RuntimeError("Reset is not supported")


class BaseAgentRunner(object):

    def __init__(self, llm_chat_fn: Callable, tokenizer: Any, config, **kwargs):
        self.tokenizer = tokenizer
        self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")
        self.cmt: Union[BasicContextTracker, Any, None] = None
        self.alien_llm_chat_fn: Union[Callable, None] = None
        self.llm_chat_fn: Callable = llm_chat_fn
        self.config = config
        self.max_steps: int = self.config.astune.rollout.multi_turn.max_steps
        self.max_model_len: int = self.config.astune.rollout.max_model_len
        self.max_env_len: int = self.config.astune.rollout.max_env_len

    def generate_gym_env(
        self, env_client: Any, task_env_uuid: str, task_thread_index: int, obs_window: Dict
    ) -> BaseGymEnv:
        return BaseGymEnv(env_client, task_env_uuid, task_thread_index, obs_window)

    def agentscope_runner_hooks(self, obs_window, task_thread_index, workflow_task, env):

        def should_interrupt_fn() -> bool:
            if (obs_window["stop"] is not None) and obs_window["stop"][
                task_thread_index
            ]:  # Check if the thread should stop (because other threads have completed, making this thread useless)
                return True
            return False

        def generated_token_callback_fn(token_array):
            obs_window["token"][task_thread_index] += len(token_array)

        return {
            "should_interrupt_fn": should_interrupt_fn,
            "generated_token_callback_fn": generated_token_callback_fn,
        }

    def get_judge(self) -> JudgeBase:
        judge_protocol = self.config.astune.task_judge.judge_protocol
        return dynamic_import(judge_protocol)(self.config)  # type: ignore
