"""Parallel environment rollout orchestration utilities."""

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Literal
from urllib.parse import quote

import numpy as np
import torch
import threading
from loguru import logger
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length

from ajet.schema.task import Task
from ajet.schema.trajectory import Sample
from ajet.task_rollout.single_worker import BaseRolloutManager
from ajet.context_tracker.basic_tracker import BaseContextTracker
from ajet.tuner_lib.weight_tuner.experimental.interchange_utils import (
    http_change_engine_status,
    http_update_rollout_pool_information,
    CurrentBatchRolloutPoolInformation,
)


def spawn_thread_shared_observation_window(n_threads) -> Dict[str, List[int | bool | str]]:
    observation_window: Dict[str, List[int | bool | str]] = {
        "info":      [""    for _ in range(n_threads + 1)],
        "step":      [0     for _ in range(n_threads)],
        "stop":      [False for _ in range(n_threads)],
        "hard_stop": [False for _ in range(n_threads)],
        "token":     [0     for _ in range(n_threads)],
    }
    return observation_window


class DynamicRolloutManager(BaseRolloutManager):
    """Dynamic rollout supporting oversampling and early termination."""

    def step_status_printer(self, observation_window):
        """Pretty-print thread progress statistics for the shared obs window."""
        # Histogram buckets: observation_window['step'] 0~5 / 5~10 / 10~15 / ...
        step_counter = {}
        current_token = sum(observation_window["token"])
        current_time = time.time()
        delta_token = current_token - self.current_token
        if delta_token < 0:
            delta_token = current_token
        delta_time = current_time - self.current_token_count_time
        self.current_token = current_token
        self.current_token_count_time = current_time
        token_gen_per_sec_str = (
            f"{delta_token/delta_time:.2f} tokens/s" if delta_time > 0 else "N/A"
        )

        for step in observation_window["step"]:
            if step == -1:
                step_counter[(-1, "terminated")] = step_counter.get((-1, "terminated"), 0) + 1
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


    def _write_swarm_rollout_dynamic_log(self, observation_window):
        base_exp_dir = self.config.ajet.experiment_dir # {exp-dir}/{experiment_name}
        fp = f"{base_exp_dir}/swarm_rollout.dynamic.log"
        string_buffer = ""
        for info in observation_window["info"]:
            string_buffer += f"{info}\n"
        with open(fp, "w") as f:
            f.write(string_buffer)
        return


    def rollout_static(
        self,
        tasks: List[Task],
        mode: Literal["sample", "validate"],
        epoch: str,
    ) -> List[BaseContextTracker]:
        """Execute non-dynamic rollouts in parallel and return collected trackers."""
        self.current_token_count_time = time.time()
        tracker_array: List[BaseContextTracker] = []
        rollout_n = 1 if mode == "validate" else self.rollout_n
        observation_window = spawn_thread_shared_observation_window(n_threads=len(tasks)*rollout_n)

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures: List[Future] = []
            for task_batch_index, task in enumerate(tasks):
                for task_rollout_index in range(rollout_n):
                    task_thread_index = task_batch_index * rollout_n + task_rollout_index
                    future = executor.submit(
                        self.rollout_env_worker,
                        task=task,
                        task_batch_index=task_batch_index,
                        task_tag=f"T{task.task_id}#R{task_rollout_index}",
                        mode=mode,
                        task_thread_index=task_thread_index,
                        observation_window=observation_window,
                    )
                    futures.append(future)

            while True:
                if not any(future.running() for future in futures):
                    break

                completed_futures = [f for f in futures if f.done()]
                failed_futures = [f for f in completed_futures if f.exception() is not None]

                if failed_futures:
                    executor.shutdown(wait=False, cancel_futures=True)
                    for f in futures:
                        if not f.done():
                            f.cancel()

                    for f in failed_futures:
                        logger.error(f"Thread failed with exception: {f.exception()}")

                    raise RuntimeError(
                        f"One of the rollout threads has encountered an exception. {len(failed_futures)} threads failed."
                    )

                self.step_status_printer(observation_window)
                time.sleep(10)

            for future in tqdm(futures, desc=f"epoch{epoch}.collect_rollout"):
                result = future.result()
                tracker_array.append(result)

            # TODO: support multi-step reward
            task_success_rate = np.mean(
                [tracker.reward_structure.success_rate for tracker in tracker_array]
            )
            task_scalar_reward = np.mean(
                [tracker.reward_structure.final_scalar_reward for tracker in tracker_array]
            )

            for tracker in tracker_array:
                tracker.current_batch_success_rate = float(task_success_rate)
                tracker.current_batch_reward = float(task_scalar_reward)

            return tracker_array


    def rollout_swarm(  # noqa: C901
        self,
        tasks: List[Task],
        mode: Literal["sample", "validate"],
        epoch: str,
        allow_sample_num_change=True,
        allow_force_stop=True,
    ) -> List[BaseContextTracker]:
        """
        Build a pool of threads to run context trackers in parallel,
        each thread re-spawn after complete, until reaching conditions to stop.
        """

        tracker_array: List[BaseContextTracker] = []
        rollout_n = self.rollout_n
        n_batch_task = len(tasks)
        n_task = min(len(tasks), self.max_parallel // rollout_n)
        assert n_task > 0, f"n_task is not valid, n_task = min(len(tasks), self.max_parallel // rollout_n) = {n_task}"
        self.current_token_count_time = time.time()

        # initialize observation window
        observation_window = spawn_thread_shared_observation_window(n_threads = n_task * rollout_n)
        executor = ThreadPoolExecutor(max_workers=self.max_parallel)
        futures: List[Future] = []
        completed_task_id_map_ct: Dict[str, List[BaseContextTracker]] = {}
        executor_lock = threading.Lock()

        # count tasks to see whether we have reach the finish line for next weight update
        def count_tasks(completed_task_id_map_ct):
            total_completed_episodes = 0
            total_completed_tasks = 0
            total_completed_non_dummy_tasks = 0
            for ct_list in completed_task_id_map_ct.values():
                total_completed_episodes += len(ct_list)
                task_cmd_reward_array = [
                    tracker.reward_structure.performance_reward for tracker in ct_list
                ]
                if (len(ct_list) >= rollout_n):
                    total_completed_tasks += 1
                    all_equal = all(x == task_cmd_reward_array[0] for x in task_cmd_reward_array)
                    if all_equal: continue
                    total_completed_non_dummy_tasks += 1
            return {
                "total_completed_episodes": total_completed_episodes,
                "total_completed_tasks": total_completed_tasks,
                "total_completed_non_dummy_tasks": total_completed_non_dummy_tasks,
            }

        def enough_sample_stop_condition(completed_task_id_map_ct) -> bool:
            # ajet.swarm_mode_sample_collection_method == "rollout_until_finish_enough_episodes"
            counts = count_tasks(completed_task_id_map_ct)
            total_completed_episodes = counts["total_completed_episodes"]
            return (total_completed_episodes >= n_batch_task * rollout_n)

        def enough_finished_task_stop_condition(completed_task_id_map_ct) -> bool:
            # ajet.swarm_mode_sample_collection_method == "rollout_until_finish_enough_tasks"
            counts = count_tasks(completed_task_id_map_ct)
            total_completed_episodes = counts["total_completed_episodes"]
            total_completed_tasks = counts["total_completed_tasks"]
            if total_completed_episodes > (self.config.ajet.swarm_mode_sample_collection_max_cached_episodes // 5 * 4):
                logger.warning(
                    f"Total cached episodes [{total_completed_episodes}] is going to exceed the max cached episodes [{self.config.ajet.swarm_mode_sample_collection_max_cached_episodes}], "
                    f"but we are still not able to meet the stop condition (current finished tasks [{total_completed_tasks}], target tasks [{n_batch_task}]), this may cause memory issues. "
                    f"The current stop condition requires at least [{rollout_n}] episodes for each task, however, among the completed [{total_completed_episodes}] episodes, "
                    f"we only have [{total_completed_tasks}] finished tasks which contain at least [{rollout_n}] episodes. "
                    f"Please make sure your swarm workers are instructed to repeat each task for enough times (current rollout_n=[{rollout_n}])"
                )
            if total_completed_episodes > self.config.ajet.swarm_mode_sample_collection_max_cached_episodes:
                logger.warning(
                    f"Too many cached episodes [{total_completed_episodes}] has exceeded the max cached episodes [{self.config.ajet.swarm_mode_sample_collection_max_cached_episodes}] "
                    f"Deleting cached episodes to release memory..."
                )
                completed_task_id_map_ct = {}
            return (total_completed_tasks >= n_batch_task)

        def enough_non_dummy_task_stop_condition(completed_task_id_map_ct) -> bool:
            # ajet.swarm_mode_sample_collection_method == "rollout_until_finish_enough_non_dummy_tasks"
            counts = count_tasks(completed_task_id_map_ct)
            total_completed_episodes = counts["total_completed_episodes"]
            total_completed_tasks = counts["total_completed_tasks"]
            total_completed_non_dummy_tasks = counts["total_completed_non_dummy_tasks"]
            if total_completed_episodes > (self.config.ajet.swarm_mode_sample_collection_max_cached_episodes // 5 * 4):
                logger.warning(
                    f"Total cached episodes [{total_completed_episodes}] is going to exceed the max cached episodes [{self.config.ajet.swarm_mode_sample_collection_max_cached_episodes}], "
                    f"but we are still not able to meet the stop condition (current finished tasks [{total_completed_non_dummy_tasks}], target tasks [{n_batch_task}]), this may cause memory issues. "
                    f"The current stop condition requires at least [{rollout_n}] episodes for each task, and each task contain at least two episodes that differs in reward, "
                    f"however, among the completed [{total_completed_episodes}] episodes, "
                    f"we only have [{total_completed_tasks}] finished tasks which contain at least [{rollout_n}] episodes, "
                    f"and we only have [{total_completed_non_dummy_tasks}] finished tasks which contain at least two episodes that differs in reward. "
                    f"Please make sure your swarm workers are instructed to repeat each task for enough times (current rollout_n=[{rollout_n}]), "
                    f"and please make sure your task is not too simple or too hard to cause all episodes to always have the same reward (e.g. all 0 or all 1)."
                )
            if total_completed_episodes > self.config.ajet.swarm_mode_sample_collection_max_cached_episodes:
                logger.warning(
                    f"Too many cached episodes [{total_completed_episodes}] has exceeded the max cached episodes [{self.config.ajet.swarm_mode_sample_collection_max_cached_episodes}] "
                    f"Deleting cached episodes to release memory..."
                )
                completed_task_id_map_ct = {}
            return (total_completed_non_dummy_tasks >= n_batch_task)

        # select stop condition function based on config
        if self.config.ajet.swarm_mode_sample_collection_method == "rollout_until_finish_enough_episodes":
            stop_condition = enough_sample_stop_condition
        elif self.config.ajet.swarm_mode_sample_collection_method == "rollout_until_finish_enough_tasks":
            stop_condition = enough_finished_task_stop_condition
        elif self.config.ajet.swarm_mode_sample_collection_method == "rollout_until_finish_enough_non_dummy_tasks":
            stop_condition = enough_non_dummy_task_stop_condition
        else:
            logger.error(f"Invalid swarm_mode_sample_collection_method: {self.config.ajet.swarm_mode_sample_collection_method}, fallback to default method: rollout_until_finish_enough_tasks")
            stop_condition = enough_finished_task_stop_condition

        def is_already_soft_stopped():
            return all(observation_window["stop"])

        # communicate with interchange server to stop new episode, and let threads finish current episode, then collect results and shutdown executor
        def stop_all_threads_soft():
            for k in range(len(observation_window["stop"])): observation_window["stop"][k] = True
            http_change_engine_status(self.config, "ENGINE.ROLLING_POST")
            return

        # communicate with interchange server to stop all threads immediately, and shutdown executor without waiting for threads to finish
        def stop_all_threads_hard():
            for k in range(len(observation_window["hard_stop"])): observation_window["hard_stop"][k] = True
            http_change_engine_status(self.config, "ENGINE.WEIGHT_SYNCING")
            return

        # pass a stop condition callback function to each thread, so that threads can check the stop condition whenever it finishes a cycle, this is faster than polling
        def stop_condition_callback(completed_task_id_map_ct):
            if stop_condition(completed_task_id_map_ct):
                if not is_already_soft_stopped():
                    stop_all_threads_soft()
                    update_rollout_result_array_preview(observation_window, completed_task_id_map_ct)
                return True
            update_rollout_result_array_preview(observation_window, completed_task_id_map_ct)
            return False

        # submit initial tasks
        dummy_task = Task(main_query="dummy task")
        for task_batch_index in range(n_task):
            for task_rollout_index in range(rollout_n):
                task_thread_index = task_batch_index * rollout_n + task_rollout_index
                observation_window["info"][task_thread_index] = f"\n\n\n\n[thread {task_thread_index} submit]\n"
                future = executor.submit(
                    self.rollout_env_worker_loop,
                    task=dummy_task,
                    task_tag="",
                    mode=mode,
                    task_batch_index=task_batch_index,
                    task_thread_index=task_thread_index,
                    observation_window=observation_window,
                    completed_task_id_map_ct=completed_task_id_map_ct,
                    executor_lock=executor_lock,
                    stop_condition_callback=stop_condition_callback,
                )
                futures.append(future)

        def update_rollout_result_array_preview(observation_window, completed_task_id_map_ct: Dict[str, List[BaseContextTracker]]):
            buffer = ""
            completed_tasks_details = {}
            for task_id, tracker_arr in completed_task_id_map_ct.items():
                buffer += f"Task {task_id} (completed {len(tracker_arr)} episodes):\n"
                episode_uuids = []
                for ct in tracker_arr:
                    buffer += f"\tEpisode: {ct.episode_uuid}\tTimelines: {len(ct.saved_timelines)}\tLLM_Calls: {ct.llm_call_cnt}\tReward: {ct.reward_structure.performance_reward}\n"
                    episode_uuids.append(ct.episode_uuid)
                completed_tasks_details[task_id] = episode_uuids
            buffer += f"\n"
            buffer += f"\n"
            counts = count_tasks(completed_task_id_map_ct)
            buffer += f"Total completed episodes: {counts['total_completed_episodes']} (target {n_batch_task * rollout_n})\n"
            buffer += f"Total completed tasks: {counts['total_completed_tasks']} (target {n_batch_task})\n"
            buffer += f"Total completed non-dummy tasks: {counts['total_completed_non_dummy_tasks']} (target {n_batch_task})\n"
            buffer += f"Current stop condition: {self.config.ajet.swarm_mode_sample_collection_method}\n"
            observation_window["info"][-1] = buffer

            # Update rollout pool information via API
            pool_info = CurrentBatchRolloutPoolInformation(
                sample_collection_method=self.config.ajet.swarm_mode_sample_collection_method,
                completed_episodes=counts['total_completed_episodes'],
                completed_episode_target=n_batch_task * rollout_n,
                completed_tasks=counts['total_completed_tasks'],
                completed_task_target=n_batch_task,
                completed_non_dummy_tasks=counts['total_completed_non_dummy_tasks'],
                completed_non_dummy_task_target=n_batch_task,
                task_expected_num_repeat=rollout_n,
                completed_tasks_details=completed_tasks_details,
            )
            http_update_rollout_pool_information(self.config, pool_info)
            return

        # loop and wait until stop condition is met, then stop threads and collect results
        CHECK_STATUS_INTERVAL = 4   # seconds
        PRINT_STATUS_INTERVAL = 12  # seconds
        cnt = 0
        while True:
            cnt += 1
            time.sleep(CHECK_STATUS_INTERVAL)
            if (cnt % ( PRINT_STATUS_INTERVAL//CHECK_STATUS_INTERVAL ) == 0):
                update_rollout_result_array_preview(observation_window, completed_task_id_map_ct)
                self.step_status_printer(observation_window)
            self._write_swarm_rollout_dynamic_log(observation_window)
            meet_stop_condition_after_new_results = stop_condition(completed_task_id_map_ct)
            if meet_stop_condition_after_new_results:
                print("Sending soft stop signal to all threads...")
                stop_all_threads_soft()
                break

        # wait for all threads to complete
        print('Finalizing all threads...')
        executor.shutdown(wait=True)

        # stop all threads hard
        print("Sending hard stop signal to all threads...")
        stop_all_threads_hard()

        # build tracker_array
        print('Collecting results...')
        for ct_list in completed_task_id_map_ct.values():
            tracker_array.extend(ct_list)

        # TODO: support multi-step reward
        task_success_rate = np.mean(
            [tracker.reward_structure.success_rate for tracker in tracker_array]
        )
        task_scalar_reward = np.mean(
            [tracker.reward_structure.final_scalar_reward for tracker in tracker_array]
        )

        for tracker in tracker_array:
            tracker.current_batch_success_rate = float(task_success_rate)
            tracker.current_batch_reward = float(task_scalar_reward)

        update_rollout_result_array_preview(observation_window, completed_task_id_map_ct)
        self._write_swarm_rollout_dynamic_log(observation_window)

        return tracker_array


    def rollout(
        self,
        tasks: List[Task],
        mode: Literal["sample", "validate"],
        epoch: str,
    ) -> List[BaseContextTracker]:
        """Delegate to dynamic rollout when oversampling is enabled."""
        if self.config.ajet.enable_swarm_mode:
            return self.rollout_swarm(tasks, mode, epoch)
        else:
            return self.rollout_static(tasks, mode, epoch)
















class VerlRolloutManager(DynamicRolloutManager):
    """High-level manager orchestrating rollouts and batch conversion."""

    def to_dataproto(self, tracker_array) -> DataProto:
        """Convert completed context trackers into a `DataProto` minibatch."""
        samples = self.trajectories_to_samples(tracker_array)
        dataproto = self.samples_to_dataproto(samples)
        return dataproto

    def trajectories_to_samples(self, tracker_array: List[BaseContextTracker]) -> List[Sample]:
        """Tokenize each tracker into `Sample` objects ready for tensorization."""
        sample_arr_final = []
        BaseContextTracker.compute_reference_advantage(tracker_array)
        for tracker in tracker_array:
            try:
                sample_arr = tracker.group_tokenize()
            except Exception as e:
                logger.bind(exception=True).exception("Error during tracker.group_tokenize()")
                raise e
            finally:
                tracker.generate_log(global_step=self.current_global_steps)
                if os.environ.get("BEST_LOGGER_PATH", None) and os.environ.get(
                    "AJET_DEBUG", None
                ):
                    logger.success(
                        f"View rollout details at [http://localhost:8181/?path={quote(os.path.abspath(os.environ['BEST_LOGGER_PATH']))}]"
                    )
            sample_arr_final += sample_arr

        if self.config.ajet.backbone in ["verl"]:
            world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
            remainder = len(sample_arr_final) % world_size
            if remainder != 0:
                import random

                remove_indices = random.sample(range(len(sample_arr_final)), remainder)
                remove_indices.sort(reverse=True)
                for idx in remove_indices:
                    sample_arr_final.pop(idx)

        return sample_arr_final

    def samples_to_dataproto(self, samples: list[Sample]) -> DataProto:
        """Pad sample fields and pack them into the `DataProto` structure expected by VERL."""
        prompt_ids: torch.Tensor | List[torch.Tensor] = []
        response_ids: torch.Tensor | List[torch.Tensor] = []
        prompt_attention_mask: torch.Tensor | List[torch.Tensor] = []
        response_attention_mask: torch.Tensor | List[torch.Tensor] = []
        prompt_position_ids: torch.Tensor | List[torch.Tensor] = []
        response_position_ids: torch.Tensor | List[torch.Tensor] = []
        prompt_loss_mask: torch.Tensor | List[torch.Tensor] = []
        response_loss_mask: torch.Tensor | List[torch.Tensor] = []

        messages = []
        step_reward_scores = []
        task_ids = []
        rollout_ids = []
        reference_advantage = []

        for sample in samples:
            assert (
                len(sample.input_ids)
                == len(sample.attention_mask)
                == len(sample.position_ids)
                == len(sample.loss_mask)
            ), f"Sample has mismatched lengths: {len(sample.input_ids)=}, {len(sample.attention_mask)=}, {len(sample.position_ids)=}, {len(sample.loss_mask)=}"

            task_ids.append(sample.task_id)
            rollout_ids.append(sample.task_tag)
            if len(sample.prompt_ids) > self.config.ajet.data.max_prompt_length:
                raise RuntimeError(f"Sample has prompt_ids length {len(sample.prompt_ids)} ")

            if len(sample.response_ids) > self.config.ajet.data.max_response_length:
                raise RuntimeError(f"Sample has prompt_ids length {len(sample.prompt_ids)} ")

            assert len(sample.prompt_ids) != 0
            assert len(sample.response_ids) != 0
            prompt_ids.append(torch.tensor(sample.prompt_ids, dtype=torch.int))
            response_ids.append(torch.tensor(sample.response_ids, dtype=torch.int))

            prompt_attention_mask.append(
                torch.tensor(sample.prompt_attention_mask, dtype=torch.int)
            )
            response_attention_mask.append(
                torch.tensor(sample.response_attention_mask, dtype=torch.int)
            )

            prompt_position_ids.append(torch.tensor(sample.prompt_position_ids, dtype=torch.int))
            response_position_ids.append(
                torch.tensor(sample.response_position_ids, dtype=torch.int)
            )

            prompt_loss_mask.append(torch.tensor(sample.prompt_loss_mask, dtype=torch.int))
            response_loss_mask.append(torch.tensor(sample.response_loss_mask, dtype=torch.int))

            reference_advantage.append(sample.reference_advantage)

            messages.append({"messages": sample.messages})
            step_reward_scores.append(sample.step_reward)  # append reward scalar

        max_prompt_length_this_batch = max([p.shape[-1] for p in prompt_ids])
        assert max_prompt_length_this_batch <= self.config.ajet.data.max_prompt_length
        max_response_length_this_batch = max([p.shape[-1] for p in response_ids])
        assert max_response_length_this_batch <= self.config.ajet.data.max_response_length

        prompt_ids = pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        prompt_attention_mask = pad_sequence(
            prompt_attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        prompt_position_ids = pad_sequence(
            prompt_position_ids,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        prompt_loss_mask = pad_sequence(
            prompt_loss_mask,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )

        prompt_ids = pad_sequence_to_length(
            prompt_ids,
            max_prompt_length_this_batch,
            self.pad_token_id,
            left_pad=True,
        )
        prompt_attention_mask = pad_sequence_to_length(
            prompt_attention_mask,
            max_prompt_length_this_batch,
            0,
            left_pad=True,
        )
        prompt_position_ids = pad_sequence_to_length(
            prompt_position_ids, max_prompt_length_this_batch, 0, left_pad=True
        )
        prompt_loss_mask = pad_sequence_to_length(
            prompt_loss_mask, max_prompt_length_this_batch, 0, left_pad=True
        )

        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        response_attention_mask = pad_sequence(
            response_attention_mask, batch_first=True, padding_value=0
        )
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)

        response_ids = pad_sequence_to_length(
            response_ids, max_response_length_this_batch, self.pad_token_id
        )
        response_attention_mask = pad_sequence_to_length(
            response_attention_mask, max_response_length_this_batch, 0
        )
        response_loss_mask = pad_sequence_to_length(
            response_loss_mask, max_response_length_this_batch, 0
        )

        delta_position_id = (
            torch.arange(1, response_ids.size(1) + 1, device=response_ids.device)
            .unsqueeze(0)
            .repeat(len(samples), 1)
        )
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

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

        return DataProto(
            batch=batch,
            non_tensor_batch={
                "task_ids": np.array(task_ids),
                "rollout_ids": np.array(rollout_ids),
                "messages": np.array(messages),
                "reward_scores": np.array(step_reward_scores),
                "reference_advantage": np.array(reference_advantage),
            },
        )
