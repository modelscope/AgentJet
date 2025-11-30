from typing import Any, Callable, Union

from astuner.context_tracker.basic_tracker import BasicContextTracker
from astuner.task_judge.judge_base import JudgeBase
from astuner.utils.dynamic_import import dynamic_import
from astuner.utils.utils import run_async_coro__no_matter_what


class BaseAgentRunner(object):
    def __init__(self, llm_chat_fn: Callable, tokenizer: Any, config, **kwargs):
        self.tokenizer = tokenizer
        self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")
        self.cmt: Union[BasicContextTracker, Any, None] = None
        self.alien_llm_chat_fn: Union[Callable, None] = None
        self.llm_chat_fn: Callable = llm_chat_fn
        self.config = config
        self.max_steps: int = self.config.astuner.rollout.multi_turn.max_steps
        self.max_model_len: int = self.config.astuner.rollout.max_model_len
        self.max_env_len: int = self.config.astuner.context_tracker.max_env_len

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

    def get_judge(self) -> JudgeBase:  # type: ignore
        if self.config.astuner.task_judge.judge_type == "customized_protocol":
            judge_protocol = self.config.astuner.task_judge.judge_protocol
            return dynamic_import(judge_protocol)(self.config)  # type: ignore

        elif self.config.astuner.task_judge.judge_type == "rubrics_auto_grader":
            # astuner/task_judge/rm_auto_grader_judge.py
            from astuner.task_judge.rm_auto_grader_judge import RMAutoGraderJudge

            judge = RMAutoGraderJudge(self.config)
            run_async_coro__no_matter_what(judge.load_rubrics_from_cache())
            return judge
