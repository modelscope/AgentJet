from typing import Any, Callable, Union

from ajet.context_tracker.basic_tracker import BaseContextTracker
from ajet.task_judge.base_judge import BaseJudge
from ajet.utils.async_utils import run_async_coroutine_with_timeout
from ajet.utils.dynamic_import import dynamic_import


class BaseAgentRunner(object):
    def __init__(self, llm_inference_fn: Callable, tokenizer: Any, config, **kwargs):
        self.tokenizer = tokenizer
        self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")
        self.tracker: Union[BaseContextTracker, Any, None] = None
        self.external_llm_fn: Union[Callable, None] = None
        self.llm_inference_fn: Callable = llm_inference_fn
        self.config = config
        self.max_steps: int = self.config.ajet.rollout.multi_turn.max_steps
        self.max_model_len: int = self.config.ajet.rollout.max_model_len

    def get_judge(self) -> BaseJudge:  # type: ignore
        if self.config.ajet.task_judge.judge_type == "customized_protocol":
            judge_protocol = self.config.ajet.task_judge.judge_protocol
            return dynamic_import(judge_protocol)(self.config)  # type: ignore

        elif self.config.ajet.task_judge.judge_type == "rubrics_auto_grader":
            # ajet/task_judge/rm_auto_grader_judge.py
            from ajet.task_judge.rm_auto_grader_judge import AutoGraderJudge

            judge = AutoGraderJudge(self.config)
            run_async_coroutine_with_timeout(judge.load_rubrics_from_cache())
            return judge

    def runner_hooks(self, observation_window, task_thread_index, workflow_task):
        def should_interrupt_fn() -> bool:
            if (observation_window["stop"] is not None) and observation_window["stop"][
                task_thread_index
            ]:  # Check if the thread should stop (because other threads have completed, making this thread useless)
                return True
            return False

        def generated_token_callback_fn(token_array):
            observation_window["token"][task_thread_index] += len(token_array)

        return {
            "should_interrupt_fn": should_interrupt_fn,
            "generated_token_callback_fn": generated_token_callback_fn,
        }
