import asyncio
import gc
from typing import Any, Callable, Union, Type
from multiprocessing import Process, Queue
from unittest import result

from ajet.context_tracker.basic_tracker import BaseContextTracker
from ajet.schema.task import WorkflowOutput, WorkflowTask
from ajet.task_judge.base_judge import BaseJudge
from ajet.tuner import AjetTuner
from ajet.utils.async_utils import run_async_coroutine_with_timeout
from ajet.utils.dynamic_import import dynamic_import
from ajet.workflow import Workflow


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

        self.wrapper_type = self.config.ajet.task_runner.wrapper_type
        self.wrapper_multiprocessing_timeout = self.config.ajet.task_runner.wrapper_multiprocessing_timeout
        assert self.wrapper_type in ["asyncio", "asyncio-with-gc", "multi-processing"], \
            f"Unsupported wrapper type: {self.wrapper_type}, available options: ['asyncio', 'asyncio-with-gc', 'multi-processing']"


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


    async def wrapper_type_asyncio(self, workflow_cls: Type[Workflow], workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:
        user_workflow: Workflow = workflow_cls(name="ajet-workflow")
        result = await user_workflow.execute(workflow_task, tuner)
        del user_workflow
        gc.collect()    # force garbage collection
        return result


    def wrapper_type_multiprocessing(self, workflow_cls: Type[Workflow], workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:
        def worker(q: Queue):
            user_workflow: Workflow = workflow_cls(name="ajet-workflow")
            result = asyncio.run(user_workflow.execute(workflow_task, tuner))
            q.put(result)
        q = Queue()
        p = Process(target=worker, args=(q,))
        p.daemon = True
        p.start()
        p.join(timeout=self.wrapper_multiprocessing_timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeoutError(f"Workflow execution timeout after {self.wrapper_multiprocessing_timeout} seconds")
        return q.get()


    def run_user_workflow(
        self,
        workflow_cls: Type[Workflow],
        workflow_task: WorkflowTask,
        tuner: AjetTuner,
    ) -> WorkflowOutput:

        if self.wrapper_type == "asyncio":
            user_workflow: Workflow = workflow_cls(name="ajet-workflow")
            return asyncio.run(user_workflow.execute(workflow_task, tuner))

        if self.wrapper_type == "asyncio-with-gc":
            return asyncio.run(self.wrapper_type_asyncio(workflow_cls, workflow_task, tuner))

        elif self.wrapper_type == "multi-processing":
            return self.wrapper_type_multiprocessing(workflow_cls, workflow_task, tuner)

        else:
            raise ValueError(f"Unsupported wrapper type: {self.wrapper_type}")

