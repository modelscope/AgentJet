from astune.context_tracker.basic_tracker import BasicContextTracker
from typing import Any, Dict, Tuple, Union, Callable
from astune.task_judge.judge_base import JudgeBase
from astune.utils.dynamic_import import dynamic_import
from astune.utils.env_service_client.env_client_ng import EnvClient as EnvClientNg


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

    def agentscope_runner_hooks(self, obs_window, task_thread_index, workflow_task):

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
