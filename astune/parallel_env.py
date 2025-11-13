"""Parallel environment rollout utilities.

This module contains classes that manage parallel LLM-based environment rollouts
for tasks. Core responsibilities include:

1. Submitting and monitoring multiple environment worker threads.
2. Dynamically oversampling rollouts and performing early termination when
    sufficient statistical confidence is reached.
3. Aggregating and converting collected trajectories into sample objects and
    finally into a `DataProto` batch ready for training.
4. Providing progress instrumentation (token generation rate, step buckets).

None of the implementation logic is modified here; only documentation and
English translations for previously non‑English comments are added.
"""

import os
import copy
import time
import numpy as np
import torch
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Literal, Callable, Any
from urllib.parse import quote
from loguru import logger
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from astune.workflow_controller.classic_agentflow import AgentFlow
from astune.workflow_controller.classic_agentflow import BaseAgentFlow
from astune.task_rollout.env_worker import EnvWorker
from astune.schema.task import Task, TaskLaunchCoreArgument
from astune.schema.trajectory import Sample
from astune.context_manager.cmt_linear import CMTLinear, CMTBaseAttr
from astune.workflow_controller.agentscope_flow import AgentScopeWorkflow
from astune.utils.utils import run_async_coro__no_matter_what
from astune.schema.logprob import TokenAndProb
from transformers.tokenization_utils import PreTrainedTokenizer
from astune.task_rollout.async_llm_bridge import AsyncLlmBridge

def init_logger(experiment_name):
    """Initialize the logger with the given configuration."""
    from beast_logger import register_logger
    if 'BEST_LOGGER_INIT' in os.environ: return # prevent re-initialization in ray environment
    os.environ['BEST_LOGGER_INIT'] = '1'
    from datetime import datetime
    final_log_path = os.path.join( "launcher_record", experiment_name, datetime.now().strftime("%Y_%m_%d_%H_%M") )
    os.environ['BEST_LOGGER_PATH'] = final_log_path
    non_console_mods = ["rollout", "token_clip", "bad_case", "env_clip"]
    register_logger(mods=["evaluation", "exception"], non_console_mods=non_console_mods, auto_clean_mods=[], base_log_path=final_log_path, debug=False)


class StepPrinter(AsyncLlmBridge):
    """Utility mixin providing periodic progress / throughput printing.

    Tracks token generation speed and thread step distribution. Intended to be
    composed with rollout manager classes that maintain an observation window
    (`obs_window`) containing per-thread step, token and stop flags.
    """

    def __init__(
        self,
        config: DictConfig,
        async_rollout_manager,
        max_parallel: int,
        max_llm_retries: int = 3,
        tokenizer: PreTrainedTokenizer = None,  # type: ignore
        llm_mode="local",
        **kwargs
    ):
        """Initialize the step printer.

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
        llm_mode : str, optional
            Indicates backend mode (e.g., 'local', 'remote'), default 'local'.
        **kwargs : Any
            Additional parameters passed through for future extensions.
        """

        init_logger(experiment_name=config.astune.experiment_name)
        self.llm_mode = llm_mode
        self.config: DictConfig = config
        self.async_rollout_manager = async_rollout_manager
        self.max_parallel: int = max_parallel
        self.max_llm_retries: int = max_llm_retries
        self.rollout_n = config.astune.rollout.num_repeat
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.current_token = 0
        self.current_global_steps = "NA"


    def step_status_printer(self, obs_window):
        """Print aggregated rollout progress.

        Buckets thread steps into 5-step ranges and shows how many threads are
        currently within each bucket plus finished threads. Also reports token
        generation speed.

        Parameters
        ----------
        obs_window : dict
            Shared state tracking 'step', 'token', and 'stop' arrays for each
            active (or finished) thread.
        """
        # Histogram buckets: obs_window['step'] 0~5 / 5~10 / 10~15 / ...
        step_counter = {}
        current_token = sum(obs_window['token'])
        current_time = time.time()
        delta_token = current_token - self.current_token
        if delta_token < 0: delta_token = current_token # when next rollout begins, obs_window['token'] should reset
        delta_time = current_time - self.current_token_count_time
        self.current_token = current_token
        self.current_token_count_time = current_time
        token_gen_per_sec_str = f"{delta_token/delta_time:.2f} tokens/s" if delta_time > 0 else "N/A"

        for step in obs_window['step']:
            if step == -1:
                step_counter[(-1, 'terminated')] = step_counter.get((-1, 'terminated'), 0) + 1
                continue
            else:
                start = (step // 5) * 5
                end = start + 5
                step_counter[(start, end)] = step_counter.get((start, end), 0) + 1

        step_counter = dict(sorted(step_counter.items(), key=lambda x: x[0][0]))

        print_buf = []
        for (start, end), count in step_counter.items():
            if start != -1:
                print_buf += [f"[{start}-{end}]:{count} threads"]
        for (start, end), count in step_counter.items():
            if start == -1:
                print_buf += [f"[finished]:{count} threads"]
        print(f"Rollout progress ({token_gen_per_sec_str}): " + "  //  ".join(print_buf))

class StaticRollout(StepPrinter, AsyncLlmBridge):
    """Static (non-dynamic) rollout manager.

    Submits a fixed number of rollout threads per task and waits for all to
    finish before collecting results. Provides retry logic for environment
    worker execution and aggregates success metrics.
    """

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
            response_length_eps = 16 # Reserve a few tokens for later handling of special tokens like lm_start.
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
                llm_chat_fn = self.get_llm_chat_fn(get_sample_params())
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

        return cmt # type: ignore


    def rollout(self, tasks: List[Task], mode: Literal["sample", "validate"], epoch: str) -> List[CMTLinear]:
        """Run static rollout over a batch of tasks.

        Parameters
        ----------
        tasks : List[Task]
            Tasks to execute.
        mode : Literal['sample','validate']
            Sampling mode; validation forces deterministic-ish params.
        epoch : str
            Epoch identifier for logging.

        Returns
        -------
        List[CMTLinear]
            List of collected trajectory containers.
        """
        # Step 1: if enable_oversample (handled by DynamicRollout override)
        self.current_token_count_time = time.time()
        # Step 2: otherwise, use legacy rollout method
        cmt_array: List[CMTLinear] = []
        rollout_n = 1 if mode=="validate" else self.rollout_n
        obs_window = {
            'step': [0 for _ in range(len(tasks) * rollout_n)],
            'token': [0 for _ in range(len(tasks) * rollout_n)],
            'stop': [False for _ in range(len(tasks) * rollout_n)],
        }
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            for task_batch_index, task in enumerate(tasks):
                for task_rollout_index in range(rollout_n):
                    task_thread_index = task_batch_index * rollout_n + task_rollout_index
                    future = executor.submit(self.rollout_env_worker,
                                             task=task, task_batch_index=task_batch_index,
                                             task_tag=f"T{task.task_id}#R{task_rollout_index}",
                                             mode=mode,
                                             task_thread_index=task_thread_index,
                                             obs_window=obs_window)
                    futures.append(future)

            while any(future.running() for future in futures):
                self.step_status_printer(obs_window)
                time.sleep(10)

            for future in tqdm(futures, desc=f"epoch{epoch}.collect_rollout"):
                # do not fail silently
                result = future.result()
                cmt_array.append(result)

            task_success_rate = np.mean([cmt.reward_structure.success_rate for cmt in cmt_array])
            for cmt in cmt_array:
                cmt.current_batch_success_rate = float(task_success_rate)

            return cmt_array


class DynamicRollout(StaticRollout):
    """Dynamic rollout supporting oversampling and early termination.

    Extends the static rollout by launching an oversampled set of threads per
    task, continuously monitoring reward variance and convergence conditions to
    decide when to truncate further sampling. Implements greedy selection to
    maximize reward dispersion (std) for training diversity.
    """

    def rollout(self, tasks: List[Task], mode: Literal["sample", "validate"], epoch: str) -> List[CMTLinear]:
        """Dispatch to dynamic or static rollout depending on configuration."""
        if mode=="sample" and (self.rollout_n!=1) and self.config.astune.rollout.enable_oversample:
            return self.rollout_dynamic(tasks, mode, epoch)
        else:
            return super().rollout(tasks, mode, epoch)

    def greedy_max_std_selection(self, samples: List[CMTLinear], n):
        """Select samples promoting maximum reward spread.

        Strategy: Group by unique reward values in order, take extremes from
        both ends when partial groups would exceed quota, optionally duplicate
        selections if fewer unique samples than required, then return sorted by
        absolute reward magnitude.

        Parameters
        ----------
        samples : List[CMTLinear]
            Candidate samples.
        n : int
            Number of samples to select.

        Returns
        -------
        List[CMTLinear]
            Selected subset with high reward variance.
        """
        if len(samples) < n:
            additional_n = n - len(samples)
            n = len(samples)
        else:
            additional_n = 0

        sorted_samples = sorted(samples, key=lambda cmt: abs(cmt.reward_structure.performance_reward))
        value_array = [cmt.reward_structure.performance_reward for cmt in sorted_samples]
        macro_selected_value = []
        macro_selected_index = []
        while len(macro_selected_index) != n:
            selected_value = []
            selected_index = []
            for index, value in enumerate(value_array):
                if (value not in selected_value) and (index not in macro_selected_index):
                    selected_value.append(value)
                    selected_index.append(index)

            if len(selected_value) + len(macro_selected_value) <= n:
                macro_selected_value += selected_value
                macro_selected_index += selected_index

            elif len(selected_value) + len(macro_selected_value) > n:
                preserve_n = n - len(macro_selected_value)
                # Select preserve_n samples from both ends of selected_value and selected_index.
                pick_left = preserve_n // 2
                pick_right = preserve_n - pick_left
                macro_selected_value += selected_value[:pick_left] + selected_value[-pick_right:]
                macro_selected_index += selected_index[:pick_left] + selected_index[-pick_right:]

        if additional_n > 0:
            # randomly select `additional_n` samples from `macro_selected_index`, then concat to `macro_selected_index`
            additional_indices = np.random.choice(macro_selected_index, additional_n, replace=True)
            macro_selected_index += additional_indices.tolist()

        selected_samples = [sorted_samples[i] for i in macro_selected_index]
        sorted_selected_samples = sorted(selected_samples, key=lambda cmt: abs(cmt.reward_structure.performance_reward))
        return sorted_selected_samples


    def rollout_dynamic(self, tasks: List[Task], mode: Literal["sample", "validate"], epoch: str, allow_sample_num_change=True, allow_force_stop=True) -> List[CMTLinear]:
        """Perform dynamic oversampled rollout with adaptive early stopping.

        Launches an oversampled number of rollout threads per task, periodically
        evaluates termination conditions (all finished, sufficient diversity,
        or confirmation of homogeneity), and force-stops remaining threads by
        setting shared stop flags when appropriate.

        Parameters
        ----------
        tasks : List[Task]
            Tasks to roll out.
        mode : Literal['sample','validate']
            Must be 'sample'; 'validate' is delegated to static logic.
        epoch : str
            Epoch identifier used for logging scope.
        allow_sample_num_change : bool, optional
            Whether final sample count may change to fit hardware divisibility.
        allow_force_stop : bool, optional
            Whether unfinished threads can be signaled to halt once criteria
            are met.

        Returns
        -------
        List[CMTLinear]
            Selected trajectories after variance-based filtering / amendment.
        """
        cmt_array: List[CMTLinear] = []
        assert mode != "validate"
        rollout_n = self.rollout_n
        self.current_token_count_time = time.time()
        submit_oversample_multiplier = self.config.astune.rollout.submit_oversample_multiplier
        rollout_n_oversample = int(rollout_n * submit_oversample_multiplier)
        rollout_n_confirm = int(rollout_n * (1 + submit_oversample_multiplier) / 2)
        assert rollout_n < rollout_n_confirm < rollout_n_oversample, \
            f"submit_oversample_multiplier is too small, rollout_n={rollout_n}, rollout_n_confirm={rollout_n_confirm}, rollout_n_oversample={rollout_n_oversample}"

        obs_window = {
            'step': [0 for _ in range(len(tasks) * rollout_n_oversample)],
            'stop': [False for _ in range(len(tasks) * rollout_n_oversample)],
            'token': [0 for _ in range(len(tasks) * rollout_n_oversample)],
        }

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit threads
            futures = []
            for task_batch_index, task in enumerate(tasks):
                task_future_array = []
                for task_rollout_index in range(rollout_n_oversample):
                    task_thread_index = task_batch_index * rollout_n_oversample + task_rollout_index
                    future = executor.submit(self.rollout_env_worker,
                                                task=task,
                                                task_batch_index=task_batch_index,
                                                task_tag=f"T{task.task_id}#R{task_rollout_index}", # task_rollout_index=str(task_rollout_index),
                                                mode=mode,
                                                task_thread_index=task_thread_index,
                                                obs_window=obs_window)
                    task_future_array.append(future)
                futures += [task_future_array]

            tic = -1
            # Record results of completed threads
            while True:
                tic += 1
                can_terminate = [False for _ in futures]
                terminate_status = ['running' for _ in futures]
                for j, task_future_array in enumerate(futures):
                    completed_task_futures = [f for f in task_future_array if f.done()]
                    completed_results = [f.result() for f in completed_task_futures]
                    completed_results = [cmt for cmt in completed_results if not cmt.discarded]
                    reward = [cmt.reward_structure.performance_reward for cmt in completed_results]
                    reward_std = np.std(reward) if reward else 0.0
                    all_finished = (len(completed_task_futures) == len(task_future_array))
                    # finish condition 1: all oversample tasks are finished
                    if all_finished:
                        can_terminate[j] = True
                        terminate_status[j] = f'all_fin({len(completed_results)}/{reward_std:.2f})'
                    num_finished = len(completed_task_futures)
                    task_cmd_reward_array = [cmt.reward_structure.performance_reward for cmt in completed_results]
                    all_equal = all(x == task_cmd_reward_array[0] for x in task_cmd_reward_array)
                    # all_reward_greater_than_one = all(x >= 1 for x in task_cmd_reward_array)
                    if not all_equal:
                        if (num_finished >= rollout_n):
                            # finish condition 2: more than rollout_n tasks are finished, and, reward are not all equal
                            can_terminate[j] = True
                            terminate_status[j] = f'early_end({len(completed_results)}/{reward_std:.2f})'
                        else:
                            pass # keep waiting
                    else:
                        if num_finished >= rollout_n_confirm:
                            # finish condition 3: if more than rollout_n_confirm tasks are finished, we can confirm this task is hopeless (or successful for certainty)
                            can_terminate[j] = True
                            terminate_status[j] = f'confirm_dummy({len(completed_results)}/{reward_std:.2f})'
                            # Take actions to stop future rollout
                            if allow_force_stop:
                                for k in range(j*rollout_n_oversample, j*rollout_n_oversample + rollout_n_oversample):
                                    obs_window['stop'][k] = True
                        else:
                            pass # keep waiting
                # check global status
                terminate_status = '/'.join(terminate_status)
                if all(can_terminate):
                    logger.info(f"epoch{epoch}.collect_rollout: all tasks finished, exiting loop")
                    for i, stop_flag in enumerate(obs_window['stop']): obs_window['stop'][i] = True # all must stop now
                    break
                else:
                    if tic % 10 == 0:
                        self.step_status_printer(obs_window) # print status every 10*5=50 seconds
                        logger.info(f"task complete {sum(can_terminate)}/{len(can_terminate)} tasks: {terminate_status}")
                    time.sleep(5)
            # Wait until all threads finish or are forcibly terminated
            tic = -1
            while any(f.running() for task_future_array in futures for f in task_future_array):
                tic += 1
                if tic % 10 == 0: logger.info('waiting final sync, this will not take long')
                time.sleep(5)

            # Check how many threads completed scheduled tasks
            task_ineffective_thread_cnt = []
            task_completed_thread_cnt = []
            task_extra_thread_cnt = []
            task_need_amend = 0
            for j, task_future_array in enumerate(futures):
                # get number of completed tasks
                completed_task_futures = [f for f in task_future_array if f.done()]
                completed_results = [f.result() for f in completed_task_futures]
                completed_results = [cmt for cmt in completed_results if not cmt.discarded]
                task_cmd_reward_array = [cmt.reward_structure.performance_reward for cmt in completed_results]
                all_equal = all(x == task_cmd_reward_array[0] for x in task_cmd_reward_array)
                # Counting
                completed_task_cnt = len(completed_results)
                if all_equal:
                    task_need_amend += 1
                    task_completed_thread_cnt += [0]
                    task_extra_thread_cnt += [0]
                    task_ineffective_thread_cnt += [completed_task_cnt]
                else:
                    task_need_amend += 0
                    task_completed_thread_cnt += [completed_task_cnt]
                    task_extra_thread_cnt += [completed_task_cnt - rollout_n]
                    task_ineffective_thread_cnt += [0]

            logger.info(f"task_completed_thread_cnt: {task_completed_thread_cnt}")
            logger.info(f"task_extra_thread_cnt: {task_extra_thread_cnt}")

            world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
            total_sample = sum(task_completed_thread_cnt)
            if allow_sample_num_change and (total_sample > world_size*2):
                # Allow sample count changes: we only require divisibility by GPU count.
                # add_count = (world_size - total_sample % world_size)   # If using an addition strategy, number of samples to add.
                add_count = 0  # If using an addition strategy, number of samples to add.
                num_task_to_amend = len(futures)    # num_task
                logger.info(f"allow_sample_num_change policy: world_size: {world_size}, total_sample {total_sample}, add_count: {add_count}, ")
                # 选择 extra 最少的task进行补偿
                while add_count != 0:
                    _task_completed_thread_cnt_find_nonzero_min = [float('inf') if x <=0 else x for x in task_completed_thread_cnt]
                    min_extra_index = _task_completed_thread_cnt_find_nonzero_min.index(min(_task_completed_thread_cnt_find_nonzero_min))
                    task_extra_thread_cnt[min_extra_index] += 1
                    task_completed_thread_cnt[min_extra_index] += 1
                    add_count -= 1
                # logger.info(f"_task_completed_thread_cnt_find_nonzero_min: {_task_completed_thread_cnt_find_nonzero_min}")
                logger.info(f"task_completed_thread_cnt (after remove): {task_completed_thread_cnt}")
                logger.info(f"task_extra_thread_cnt (after remove): {task_extra_thread_cnt}")
            else:
                # Sample count change not allowed; attempt compensation.
                num_task_max_to_amend = sum(task_extra_thread_cnt) // rollout_n
                num_task_to_amend = min(num_task_max_to_amend, task_need_amend)
                extra_num_thread_required = num_task_to_amend * rollout_n
                remove_count = sum(task_extra_thread_cnt) - extra_num_thread_required
                logger.info(f"forbid_sample_num_change policy: num_task_max_to_amend: {num_task_max_to_amend}, num_task_to_amend: {num_task_to_amend}, remove_count: {remove_count}, ")

                # 选择 extra 最多的task进行约束
                while remove_count != 0:
                    max_extra_index = task_extra_thread_cnt.index(max(task_extra_thread_cnt))
                    assert task_extra_thread_cnt[max_extra_index] > 0, "task_extra_thread_cnt should be greater than 0"
                    task_extra_thread_cnt[max_extra_index] -= 1
                    task_completed_thread_cnt[max_extra_index] -= 1
                    remove_count -= 1
                logger.info(f"task_completed_thread_cnt (after remove): {task_completed_thread_cnt}")
                logger.info(f"task_extra_thread_cnt (after remove): {task_extra_thread_cnt}")

            # 筛选出方差最高的样本
            cmt_array = []
            print_buffer = ""
            task_success_rate = []
            for j, task_future_array, avail_extra_cnt in zip(range(len(futures)), futures, task_extra_thread_cnt):
                # get number of completed tasks
                completed_task_futures = [f for f in task_future_array if f.done()]
                completed_results = [f.result() for f in completed_task_futures]
                completed_results: List[CMTLinear] = [cmt for cmt in completed_results if not cmt.discarded]
                task_cmd_reward_array = [cmt.reward_structure.performance_reward for cmt in completed_results]
                success_rate_array = [cmt.reward_structure.success_rate for cmt in completed_results]
                task_success_rate += [np.mean(success_rate_array)]
                need_amend = all(x == task_cmd_reward_array[0] for x in task_cmd_reward_array)
                if need_amend and (num_task_to_amend > 0):
                    num_task_to_amend -= 1
                    print_buffer += f"/(amend)"
                    continue
                else:
                    if need_amend:
                        num_completed = len(completed_results)
                        num_to_be_selected = rollout_n
                    else:
                        num_completed = len(completed_results)
                        num_to_be_selected = rollout_n + avail_extra_cnt
                    # assert num_completed >= num_to_be_selected, f"num_completed={num_completed}, num_to_be_selected={num_to_be_selected}"
                    selected_cmt_array = self.greedy_max_std_selection(completed_results, num_to_be_selected)
                    cmt_array += selected_cmt_array
                    print_buffer += f"/({len(selected_cmt_array)})"
                    if need_amend: print_buffer += "(no-amend)"
            logger.info(print_buffer)

            for cmt in cmt_array:
                cmt.current_batch_success_rate = np.mean(task_success_rate)
            return cmt_array




class ParallelEnvManager(DynamicRollout):
    """High-level manager orchestrating parallel environment rollouts.

    Adds conversion helpers to transform collected trajectories into
    `Sample` objects and finally a padded `DataProto` suitable for model
    training.
    """

    # TODO: define an extra class for trajectory-dataproto converting.
    def to_dataproto(self, cmt_array) -> DataProto:
        """Convert trajectories to a `DataProto` batch.

        Pipeline:
        1. Tokenize grouped trajectories into `Sample` objects.
        2. Pad and batch samples into tensors and auxiliary metadata.
        """
        # Step 1: Convert trajectories to samples: tokenizing
        samples = self.trajectories_to_samples(cmt_array)

        # Step 2: Convert samples to DataProto: padding
        dataproto = self.samples_to_dataproto(samples)

        return dataproto

    def trajectories_to_samples(self, cmt_array: List[CMTLinear]) -> List[Sample]:
        """Convert a list of trajectory containers into flat `Sample` objects.

        Ensures divisibility of sample count by world size (removing random
        excess) for downstream distributed training alignment.
        """
        sample_arr_final = []
        CMTLinear.compute_reference_advantage(cmt_array)
        for cmt in cmt_array:
            try:
                sample_arr = cmt.group_tokenize()
            except Exception as e:
                raise e
            finally:
                cmt.generate_log(global_step=self.current_global_steps)
                if os.environ.get('BEST_LOGGER_PATH', None) and os.environ.get('ASTUNE_DEBUG', None):
                    logger.success(f"View rollout details at [http://localhost:8181/?path={quote(os.path.abspath(os.environ['BEST_LOGGER_PATH']))}]")
            sample_arr_final += sample_arr

        # Step 2: Calculate how many samples need to be removed
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        remainder = len(sample_arr_final) % world_size
        if remainder != 0:
            import random
            remove_indices = random.sample(range(len(sample_arr_final)), remainder)
            # Sort in reverse order to avoid index shifting during removal
            remove_indices.sort(reverse=True)
            for idx in remove_indices:
                sample_arr_final.pop(idx)

        # random remove some samples, so that the number of samples is divisible by 8
        return sample_arr_final

    def samples_to_dataproto(self, samples: list[Sample]) -> DataProto:
        # Initialize lists to store batched data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        messages = []
        # reward_scores = [] # replace with step_reward_scores
        step_reward_scores = []
        task_ids = []
        rollout_ids = []
        reference_advantage = []

        for sample in samples:
            # Validate that all fields have the same length
            assert len(sample.input_ids) == len(sample.attention_mask) == len(sample.position_ids) == len(
                sample.loss_mask), f"Sample {sample.request_id} has mismatched lengths: " \
                                f"{len(sample.input_ids)=}, {len(sample.attention_mask)=}, " \
                                f"{len(sample.position_ids)=}, {len(sample.loss_mask)=}"

            task_ids.append(sample.task_id)
            rollout_ids.append(sample.task_tag)
            # Discard samples with prompt length exceeding limit
            if len(sample.prompt_ids) > self.config.astune.data.max_prompt_length:
                raise RuntimeError(f"Sample has prompt_ids length {len(sample.prompt_ids)} ")

            # Warn if response is longer than expected (but still include it)
            if len(sample.response_ids) > self.config.astune.data.max_response_length:
                raise RuntimeError(f"Sample has prompt_ids length {len(sample.prompt_ids)} ")

            # Append tensors to respective lists
            assert len(sample.prompt_ids) != 0
            assert len(sample.response_ids) != 0
            prompt_ids.append(torch.tensor(sample.prompt_ids, dtype=torch.int))
            response_ids.append(torch.tensor(sample.response_ids, dtype=torch.int))

            prompt_attention_mask.append(torch.tensor(sample.prompt_attention_mask, dtype=torch.int))
            response_attention_mask.append(torch.tensor(sample.response_attention_mask, dtype=torch.int))

            prompt_position_ids.append(torch.tensor(sample.prompt_position_ids, dtype=torch.int))
            response_position_ids.append(torch.tensor(sample.response_position_ids, dtype=torch.int))

            prompt_loss_mask.append(torch.tensor(sample.prompt_loss_mask, dtype=torch.int))
            response_loss_mask.append(torch.tensor(sample.response_loss_mask, dtype=torch.int))

            reference_advantage.append(sample.reference_advantage)

            messages.append({"messages": sample.messages})
            # reward_scores.append(sample.global_reward)
            step_reward_scores.append(sample.step_reward)

        max_prompt_length_this_batch = max([p.shape[-1] for p in prompt_ids])
        assert max_prompt_length_this_batch <= self.config.astune.data.max_prompt_length
        max_response_length_this_batch = max([p.shape[-1] for p in response_ids])
        assert max_response_length_this_batch <= self.config.astune.data.max_response_length

        # Batch and pad sequences
        prompt_ids =            pad_sequence(prompt_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left")
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        prompt_position_ids =   pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
        prompt_loss_mask =      pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")

        prompt_ids =            pad_sequence_to_length(prompt_ids, max_prompt_length_this_batch, self.pad_token_id, left_pad=True)
        prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_position_ids =   pad_sequence_to_length(prompt_position_ids, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_loss_mask =      pad_sequence_to_length(prompt_loss_mask, max_prompt_length_this_batch, 0, left_pad=True)

        response_ids =            pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        response_loss_mask =      pad_sequence(response_loss_mask, batch_first=True, padding_value=0)

        response_ids =            pad_sequence_to_length(response_ids, max_response_length_this_batch, self.pad_token_id)
        response_attention_mask = pad_sequence_to_length(response_attention_mask, max_response_length_this_batch, 0)
        response_loss_mask =      pad_sequence_to_length(response_loss_mask, max_response_length_this_batch, 0)

        delta_position_id = torch.arange(1, response_ids.size(1) + 1, device=response_ids.device).unsqueeze(0).repeat(len(samples), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id

        # Concatenate prompt and response tensors
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

        # Construct the batch using TensorDict
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            },
            batch_size=len(samples),
        )

        return DataProto(batch=batch, non_tensor_batch={
            "task_ids": np.array(task_ids),
            "rollout_ids": np.array(rollout_ids),
            "messages": np.array(messages),
            "reward_scores": np.array(step_reward_scores),
            "reference_advantage": np.array(reference_advantage),
        })
