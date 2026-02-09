"""Single worker primitives for environment rollouts."""

import uuid
import time
import threading
from typing import Literal

from loguru import logger
from omegaconf import DictConfig
from typing import Dict, List, Literal
from transformers.tokenization_utils import PreTrainedTokenizer

from ajet.context_tracker.basic_tracker import BaseContextTracker
from ajet.schema.task import Task, WorkflowTask
from ajet.task_rollout.async_llm_bridge import AsyncLlmBridge
from ajet.task_rollout.resource_keeper import ResourceKeeper
from ajet.task_runner.general_runner import GeneralRunner
from ajet.task_runner.swarm_runner import SwarmRunner
from ajet.utils.retry import retry_with_backoff
from ajet.utils.retry import SwarmReceiveAbortException
from ajet.utils.sample import get_sample_params
from ajet.utils.testing_utils import TestFailException, TestSuccessException


class BaseRolloutManager:
    def __init__(
        self,
        config: DictConfig,
        async_rollout_manager,
        max_parallel: int,
        max_llm_retries: int = 3,
        tokenizer: PreTrainedTokenizer = None,  # type: ignore
        llm_mode: Literal["local", "remote", "trinity"] = "local",
        **kwargs,
    ):
        """Initialize common rollout state and helpers.

        Parameters
        ----------
        config : DictConfig
            Configuration object containing rollout and experiment settings.
        async_rollout_manager : Any
            Manager responsible for async LLM interactions.
        max_parallel : int
            Maximum number of parallel environment worker threads.
        max_llm_retries : int, optional
            Maximum retries for LLM calls, by default 3.
        tokenizer : PreTrainedTokenizer, optional
            Tokenizer used for padding and ID conversions.
        llm_mode : Literal["local", "remote", "trinity"], optional
            Indicates backend mode (e.g., 'local', 'remote'), default 'local'.
        **kwargs : Any
            Additional parameters passed through for future extensions.
        """

        self.llm_mode: Literal["local", "remote", "trinity"] = llm_mode
        self.config: DictConfig = config
        self.async_rollout_manager = async_rollout_manager
        self.max_parallel: int = max_parallel
        self.max_llm_retries: int = max_llm_retries
        self.rollout_n = config.ajet.rollout.num_repeat
        self.tokenizer = tokenizer
        self.pad_token_id: int = self.tokenizer.pad_token_id  # type: ignore
        assert isinstance(self.pad_token_id, int), "pad_token_id must be an integer"
        self.current_token = 0
        self.current_global_steps: int | str = "NA"
        self.enable_swarm_mode = config.ajet.enable_swarm_mode
        self.async_llm_bridge = AsyncLlmBridge(
            config=config,
            async_rollout_manager=async_rollout_manager,
            tokenizer=tokenizer,
            llm_mode=llm_mode,
            max_llm_retries=max_llm_retries,
        )

    @retry_with_backoff(max_retry_attr="max_llm_retries")
    def rollout_env_worker(
        self,
        task: Task,
        task_batch_index: int,
        task_tag: str,
        mode: Literal["sample", "validate"],
        task_thread_index: int,
        observation_window: dict,
        **kwargs,
    ) -> BaseContextTracker:
        """Execute one environment rollout worker.

        Handles environment initialization, LLM sampling parameter construction
        (with validation overrides), and robust retry on transient failures.
        """
        sampling_params = get_sample_params(mode, self.config)

        if self.config.ajet.task_runner.llm_infer_submit_method == "sync":
            llm_inference_fn = self.async_llm_bridge.get_llm_inference_fn_sync(
                sampling_params=sampling_params
            )
        else:
            llm_inference_fn = self.async_llm_bridge.get_llm_inference_fn_async(
                sampling_params=sampling_params
            )

        workflow_task = WorkflowTask(
            env_type=task.env_type,
            task_id=task.task_id,
            task_thread_index=task_thread_index,
            task_batch_index=task_batch_index,
            episode_uuid=uuid.uuid4().hex,
            task_tag=task_tag,
            observation_window=observation_window,
            llm_inference_fn=llm_inference_fn,
            tokenizer=self.tokenizer,
            task=task,
        )

        with ResourceKeeper(workflow_task, config=self.config) as resource_keeper:
            try:
                workflow_task = resource_keeper.prepare()
                if self.enable_swarm_mode:
                    agent_runner = SwarmRunner(
                        llm_inference_fn=llm_inference_fn, tokenizer=self.tokenizer, config=self.config
                    )
                else:
                    agent_runner = GeneralRunner(
                        llm_inference_fn=llm_inference_fn, tokenizer=self.tokenizer, config=self.config
                    )
                tracker = agent_runner.execute(
                    workflow_task=workflow_task,
                )
            except SwarmReceiveAbortException as exc:  # noqa: BLE001
                # print('SwarmReceiveAbortException caught in rollout_env_worker')
                return None # type: ignore
            except TestSuccessException as e:
                logger.success(
                    f"env_worker.agent_flow completed with TestSuccessException: {e.args}"
                )
                raise e
            except TestFailException as e:
                logger.error(f"env_worker.agent_flow failed with TestFailException: {e.args}")
                raise e
            except Exception as e:
                logger.bind(exception=True).exception(
                    f"encounter exception in env_worker.agent_flow error={e.args}"
                )
                raise e

        return tracker


    def rollout_env_worker_loop(
        self,
        task: Task,
        task_batch_index: int,
        task_tag: str,
        mode: Literal["sample", "validate"],
        task_thread_index: int,
        observation_window: dict,
        completed_task_id_map_ct: Dict[str, List[BaseContextTracker]],
        executor_lock: threading.Lock,
        **kwargs,
    ):
        try:

            cnt = 1

            while True:

                if observation_window["stop"][task_thread_index]:           # since we use multi-threading, the best way to communicate with main thread is through shared memory.
                    return

                observation_window["info"][task_thread_index] = str(cnt)    # observe how many iterations have been done in the loop

                # Let's begin working on the task, the result `tracker` will contain everything: reward, llm calls, conversation history, etc.
                # Later we will gather all trackers and do post-processing, generating samples for VeRL.
                tracker = self.rollout_env_worker(
                    task=task,
                    task_batch_index=task_batch_index,
                    task_tag=task_tag,
                    mode=mode,
                    task_thread_index=task_thread_index,
                    observation_window=observation_window,
                    **kwargs,
                )

                # avoid write conflict
                if tracker and tracker.reward_structure:
                    with executor_lock:
                        if tracker.task_id not in completed_task_id_map_ct:
                            completed_task_id_map_ct[tracker.task_id] = [tracker]
                        else:
                            completed_task_id_map_ct[tracker.task_id] += [tracker]

                cnt += 1

                if observation_window["stop"][task_thread_index]:
                    return
                else:
                    del tracker

        except Exception as e:
            logger.exception(
                f"encounter exception in env_worker_loop error={e.args}"
            )
            raise e