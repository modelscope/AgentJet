"""Single worker primitives for environment rollouts."""

import os
import time
import uuid
from typing import Any, Literal

from loguru import logger
from omegaconf import DictConfig
from transformers.tokenization_utils import PreTrainedTokenizer

from astune.context_manager.cmt_linear import CMTLinear, CMTBaseAttr
from astune.schema.task import Task, TaskLaunchCoreArgument
from astune.task_rollout.async_llm_bridge import AsyncLlmBridge
from astune.task_rollout.env_worker import EnvWorker


def init_parallel_rollout_logger(experiment_name):
    """Initialize the logger with the given configuration."""
    from beast_logger import register_logger
    if 'BEST_LOGGER_INIT' in os.environ: return  # prevent re-initialization in ray environment
    os.environ['BEST_LOGGER_INIT'] = '1'
    from datetime import datetime
    final_log_path = os.path.join("launcher_record", experiment_name, datetime.now().strftime("%Y_%m_%d_%H_%M"))
    os.environ['BEST_LOGGER_PATH'] = final_log_path
    non_console_mods = ["rollout", "token_clip", "bad_case", "env_clip"]
    register_logger(mods=["evaluation", "exception"], non_console_mods=non_console_mods, auto_clean_mods=[], base_log_path=final_log_path, debug=False)


class BaseParallelEnv:

    def __init__(
        self,
        config: DictConfig,
        async_rollout_manager,
        max_parallel: int,
        max_llm_retries: int = 3,
        tokenizer: PreTrainedTokenizer = None,  # type: ignore
        llm_mode: Literal["local", "remote", "trinity"] = "local",
        **kwargs
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

        init_parallel_rollout_logger(experiment_name=config.astune.experiment_name)
        self.llm_mode: Literal["local", "remote", "trinity"] = llm_mode
        self.config: DictConfig = config
        self.async_rollout_manager = async_rollout_manager
        self.max_parallel: int = max_parallel
        self.max_llm_retries: int = max_llm_retries
        self.rollout_n = config.astune.rollout.num_repeat
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.current_token = 0
        self.current_global_steps = "NA"
        self.async_llm_bridge = AsyncLlmBridge(
            config=config,
            async_rollout_manager=async_rollout_manager,
            tokenizer=tokenizer,
            llm_mode=llm_mode,
            max_llm_retries=max_llm_retries
        )

    def rollout_env_worker(self, task: Task, task_batch_index: int, task_tag: str, mode: Literal["sample", "validate"],
                           task_thread_index: int, obs_window: dict, **kwargs) -> CMTLinear:
        """Execute one environment rollout worker.

        Handles environment initialization, LLM sampling parameter construction
        (with validation overrides), and robust retry on transient failures.

        Parameters
        ----------
        task : Task
            The task object to roll out.
        task_batch_index : int
            Index of the task within the provided batch.
        task_tag : str
            Human-readable tag identifying task and rollout repetition.
        mode : Literal['sample','validate']
            Rollout mode selecting sampling hyperparameters.
        task_thread_index : int
            Global thread index for obs_window bookkeeping.
        obs_window : dict
            Shared progress structure updated by the worker.
        **kwargs : Any
            Forwarded for future extensibility.

        Returns
        -------
        CMTLinear
            Collected trajectory container for this rollout.
        """
        def get_sample_params():
            response_length_eps = 16  # Reserve a few tokens for later handling of special tokens like lm_start.
            if self.config.astune.rollout.name == 'vllm':
                sampling_params = dict(
                    n=1,
                    max_tokens=self.config.astune.rollout.max_response_length_in_one_turn - response_length_eps,
                    min_tokens=1,   # Must output at least 1 token.
                    temperature=self.config.astune.rollout.temperature,
                    top_p=self.config.astune.rollout.top_p
                )
            else:
                sampling_params = dict(
                    n=1,
                    max_new_tokens=self.config.astune.rollout.max_response_length_in_one_turn,
                    temperature=self.config.astune.rollout.temperature,
                    top_p=self.config.astune.rollout.top_p
                )

            if mode == "validate":
                sampling_params["temperature"] = self.config.astune.rollout.val_kwargs.temperature
                sampling_params["top_k"] = self.config.astune.rollout.val_kwargs.top_k
                sampling_params["top_p"] = self.config.astune.rollout.val_kwargs.top_p
            return sampling_params

        max_retry = 3
        for retry in range(max_retry):
            try:
                llm_chat_fn = self.async_llm_bridge.get_llm_chat_fn(get_sample_params())
                cmt: CMTBaseAttr = EnvWorker(
                    task_core_arg=TaskLaunchCoreArgument(
                        env_type=task.env_type,
                        task_id=task.task_id,
                        task_thread_index=task_thread_index,
                        task_batch_index=task_batch_index,
                        task_env_uuid=uuid.uuid4().hex,
                        task_tag=task_tag,
                        obs_window=obs_window,
                        llm_chat_fn=llm_chat_fn,
                        tokenizer=self.tokenizer,
                        task=task
                    ),
                    config=self.config
                ).execute()
                break
            except Exception as e:
                if retry < max_retry - 1:
                    logger.bind(exception=True).exception(f"rollout_env_worker error: {e.args}, retrying {retry + 1}/{max_retry}")
                    time.sleep(2 ** retry)
                else:
                    logger.bind(exception=True).exception(f"rollout_env_worker failed after {max_retry} retries: {e.args}")
                    raise e

        return cmt  # type: ignore
