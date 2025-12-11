"""Parallel environment rollout orchestration utilities."""

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Literal
from urllib.parse import quote

import numpy as np
import torch
from loguru import logger
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length

from astuner.context_tracker.basic_tracker import BasicContextTracker
from astuner.schema.task import Task
from astuner.schema.trajectory import Sample
from astuner.task_rollout.single_worker import BaseRolloutManager


class ClassicRolloutManager(BaseRolloutManager):
    """Static (non-dynamic) rollout manager."""

    def step_status_printer(self, obs_window):
        """Pretty-print thread progress statistics for the shared obs window."""
        # Histogram buckets: obs_window['step'] 0~5 / 5~10 / 10~15 / ...
        step_counter = {}
        current_token = sum(obs_window["token"])
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

        for step in obs_window["step"]:
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

    def rollout(
        self,
        tasks: List[Task],
        mode: Literal["sample", "validate"],
        epoch: str,
    ) -> List[BasicContextTracker]:
        """Execute non-dynamic rollouts in parallel and return collected trackers."""
        self.current_token_count_time = time.time()
        cmt_array: List[BasicContextTracker] = []
        rollout_n = 1 if mode == "validate" else self.rollout_n
        obs_window = {
            "step": [0 for _ in range(len(tasks) * rollout_n)],
            "token": [0 for _ in range(len(tasks) * rollout_n)],
            "stop": [False for _ in range(len(tasks) * rollout_n)],
        }
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
                        obs_window=obs_window,
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

                self.step_status_printer(obs_window)
                time.sleep(10)

            for future in tqdm(futures, desc=f"epoch{epoch}.collect_rollout"):
                result = future.result()
                cmt_array.append(result)

            # TODO: support multi-step reward
            task_success_rate = np.mean([cmt.reward_structure.success_rate for cmt in cmt_array])
            task_scalar_reward = np.mean(
                [cmt.reward_structure.final_scalar_reward for cmt in cmt_array]
            )

            for cmt in cmt_array:
                cmt.current_batch_success_rate = float(task_success_rate)
                cmt.current_batch_reward = float(task_scalar_reward)

            return cmt_array


class DynamicRolloutManager(ClassicRolloutManager):
    """Dynamic rollout supporting oversampling and early termination."""

    def rollout(
        self,
        tasks: List[Task],
        mode: Literal["sample", "validate"],
        epoch: str,
    ) -> List[BasicContextTracker]:
        """Delegate to dynamic rollout when oversampling is enabled."""
        if (
            mode == "sample"
            and (self.rollout_n != 1)
            and self.config.astuner.rollout.enable_oversample
        ):
            return self.rollout_dynamic(tasks, mode, epoch)
        else:
            return super().rollout(tasks, mode, epoch)

    def greedy_max_std_selection(self, samples: List[BasicContextTracker], n):
        """Select samples whose rewards maximize spread to cover diverse rollouts."""
        if len(samples) < n:
            additional_n = n - len(samples)
            n = len(samples)
        else:
            additional_n = 0

        sorted_samples = sorted(
            samples,
            key=lambda cmt: abs(cmt.reward_structure.performance_reward),
        )
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
                pick_left = preserve_n // 2
                pick_right = preserve_n - pick_left
                macro_selected_value += selected_value[:pick_left] + selected_value[-pick_right:]
                macro_selected_index += selected_index[:pick_left] + selected_index[-pick_right:]

        if additional_n > 0:
            additional_indices = np.random.choice(macro_selected_index, additional_n, replace=True)
            macro_selected_index += additional_indices.tolist()

        selected_samples = [sorted_samples[i] for i in macro_selected_index]
        sorted_selected_samples = sorted(
            selected_samples,
            key=lambda cmt: abs(cmt.reward_structure.performance_reward),
        )
        return sorted_selected_samples

    def rollout_dynamic(  # noqa: C901
        self,
        tasks: List[Task],
        mode: Literal["sample", "validate"],
        epoch: str,
        allow_sample_num_change=True,
        allow_force_stop=True,
    ) -> List[BasicContextTracker]:
        """Perform oversampled rollouts with optional early termination heuristics."""

        cmt_array: List[BasicContextTracker] = []
        assert mode != "validate"
        rollout_n = self.rollout_n
        self.current_token_count_time = time.time()
        submit_oversample_multiplier = self.config.astuner.rollout.submit_oversample_multiplier
        rollout_n_oversample = int(rollout_n * submit_oversample_multiplier)
        rollout_n_confirm = int(rollout_n * (1 + submit_oversample_multiplier) / 2)
        assert (
            rollout_n < rollout_n_confirm < rollout_n_oversample
        ), f"submit_oversample_multiplier is too small, rollout_n={rollout_n}, rollout_n_confirm={rollout_n_confirm}, rollout_n_oversample={rollout_n_oversample}"

        obs_window: Dict[str, List[int | bool]] = {
            "step": [0 for _ in range(len(tasks) * rollout_n_oversample)],
            "stop": [False for _ in range(len(tasks) * rollout_n_oversample)],
            "token": [0 for _ in range(len(tasks) * rollout_n_oversample)],
        }

        from vsdb import bp

        bp("POOLX")

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            for task_batch_index, task in enumerate(tasks):
                task_future_array = []
                for task_rollout_index in range(rollout_n_oversample):
                    task_thread_index = task_batch_index * rollout_n_oversample + task_rollout_index
                    future = executor.submit(
                        self.rollout_env_worker,
                        task=task,
                        task_batch_index=task_batch_index,
                        task_tag=f"T{task.task_id}#R{task_rollout_index}",
                        mode=mode,
                        task_thread_index=task_thread_index,
                        obs_window=obs_window,
                    )
                    task_future_array.append(future)
                futures += [task_future_array]

            # A while loop to wait for all task can be terminated
            tic = -1
            while True:
                tic += 1
                can_terminate = [False for _ in futures]
                terminate_status = ["running" for _ in futures]
                for j, task_future_array in enumerate(futures):
                    completed_task_futures = [f for f in task_future_array if f.done()]
                    completed_results = [f.result() for f in completed_task_futures]
                    completed_results = [cmt for cmt in completed_results if not cmt.discarded]
                    reward = [cmt.reward_structure.performance_reward for cmt in completed_results]
                    reward_std = np.std(reward) if reward else 0.0
                    all_finished = len(completed_task_futures) == len(task_future_array)
                    if all_finished:
                        can_terminate[j] = True
                        terminate_status[j] = f"all_fin({len(completed_results)}/{reward_std:.2f})"
                    num_finished = len(completed_task_futures)
                    task_cmd_reward_array = [
                        cmt.reward_structure.performance_reward for cmt in completed_results
                    ]
                    all_equal = all(x == task_cmd_reward_array[0] for x in task_cmd_reward_array)
                    if not all_equal:
                        if num_finished >= rollout_n:
                            can_terminate[j] = True
                            terminate_status[
                                j
                            ] = f"early_end({len(completed_results)}/{reward_std:.2f})"
                        else:
                            pass
                    else:
                        if num_finished >= rollout_n_confirm:
                            can_terminate[j] = True
                            terminate_status[
                                j
                            ] = f"confirm_dummy({len(completed_results)}/{reward_std:.2f})"
                            if allow_force_stop:
                                for k in range(
                                    j * rollout_n_oversample,
                                    j * rollout_n_oversample + rollout_n_oversample,
                                ):
                                    obs_window["stop"][k] = True
                        else:
                            pass
                terminate_status = "/".join(terminate_status)
                if all(can_terminate):
                    logger.info(f"epoch{epoch}.collect_rollout: all tasks finished, exiting loop")
                    for i, stop_flag in enumerate(obs_window["stop"]):
                        obs_window["stop"][i] = True
                    break
                else:
                    if tic % 10 == 0:
                        self.step_status_printer(obs_window)
                        logger.info(
                            f"task complete {sum(can_terminate)}/{len(can_terminate)} tasks: {terminate_status}"
                        )
                    time.sleep(5)

            # We have enough number of samples, but we need to wait for all threads to finish, including discarded threads
            tic = -1
            while any(f.running() for task_future_array in futures for f in task_future_array):
                tic += 1
                if tic % 10 == 0:
                    logger.info("waiting final sync, this will not take long")
                time.sleep(5)

            # find sample group that has identical reward, mark them as need_amend
            task_ineffective_thread_cnt = []
            task_completed_thread_cnt = []  # how many effective threads are obtained per group
            task_extra_thread_cnt = (
                []
            )  # using rollout_n as baseline, how many extra threads are obtained per group
            task_need_amend = 0  # how many groups need amendment due to identical rewards
            for j, task_future_array in enumerate(futures):
                completed_task_futures = [f for f in task_future_array if f.done()]
                completed_results = [f.result() for f in completed_task_futures]
                completed_results = [cmt for cmt in completed_results if not cmt.discarded]
                task_cmd_reward_array = [
                    cmt.reward_structure.performance_reward for cmt in completed_results
                ]
                all_equal = all(x == task_cmd_reward_array[0] for x in task_cmd_reward_array)
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

            # reduce `task_extra_thread_cnt`
            world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
            # the number of all reward-diverse samples
            total_sample = sum(task_completed_thread_cnt)

            # begin to compute a removal plan  (output: `task_extra_thread_cnt` and `num_task_to_amend`)
            # - task_extra_thread_cnt: using rollout_n as baseline, how many extra threads to preserve per group
            # - num_task_to_amend: how many groups can be amended according to removal plan
            if allow_sample_num_change and (total_sample > world_size * 2):
                # When changing the number of samples is ALLOWED
                num_task_to_amend = len(
                    futures
                )  # this means infinate budget to amend, indicating that we throw away all ineffective samples
                task_extra_thread_cnt = task_extra_thread_cnt  # do not change extra thread cnt, we simply take all diverse samples
                # log
                logger.info(
                    f"task_completed_thread_cnt (after remove): {task_completed_thread_cnt}"
                )
                logger.info(f"task_extra_thread_cnt (after remove): {task_extra_thread_cnt}")
            else:
                # When changing the number of samples is NOT ALLOWED (or the number of samples are too small)
                # compute how many valid extra samples are obtained during previous oversampling
                num_task_max_to_amend = sum(task_extra_thread_cnt) // rollout_n
                # compute how many tasks actually need amendment, we fix as many as we can, but not exceed `num_task_max_to_amend`:
                # - num_task_max_to_amend: how many CAN be fixed
                # - task_need_amend:       how many SHOULD be fixed
                num_task_to_amend = min(num_task_max_to_amend, task_need_amend)
                # according to `num_task_to_amend`, how many extra samples should be CONSUMED
                extra_num_thread_required = num_task_to_amend * rollout_n
                # after CONSUME, how many extra samples are really EXTRA and should be REMOVED
                remove_count = sum(task_extra_thread_cnt) - extra_num_thread_required
                logger.info(
                    f"forbid_sample_num_change policy: num_task_max_to_amend: {num_task_max_to_amend}, "
                    f"num_task_to_amend: {num_task_to_amend}, remove_count: {remove_count}, "
                )
                # remove extra samples according to `remove_count`
                while remove_count != 0:
                    # if we should remove some extra samples, we always remove from the group that has the MOST extra samples
                    max_extra_index = task_extra_thread_cnt.index(max(task_extra_thread_cnt))
                    assert (
                        task_extra_thread_cnt[max_extra_index] > 0
                    ), "task_extra_thread_cnt should be greater than 0"
                    task_extra_thread_cnt[max_extra_index] -= 1
                    task_completed_thread_cnt[max_extra_index] -= 1
                    remove_count -= 1

                # now, we have computed the final `task_extra_thread_cnt` and `num_task_to_amend`, which the removal plan deps
                logger.info(
                    f"task_completed_thread_cnt (after remove): {task_completed_thread_cnt}"
                )
                logger.info(f"task_extra_thread_cnt (after remove): {task_extra_thread_cnt}")

            # collect results and get the final cmt_array according to removal plan (`task_extra_thread_cnt` and `num_task_to_amend`)
            cmt_array = []
            print_buffer = ""
            task_success_rate = []
            task_group_reward = []
            for j, task_future_array, avail_extra_cnt in zip(
                range(len(futures)), futures, task_extra_thread_cnt
            ):
                completed_task_futures = [f for f in task_future_array if f.done()]
                completed_results = [f.result() for f in completed_task_futures]
                completed_results = [cmt for cmt in completed_results if not cmt.discarded]
                # in-group success rate and reward
                task_cmd_reward_array = [
                    cmt.reward_structure.performance_reward for cmt in completed_results
                ]
                success_rate_array = [
                    cmt.reward_structure.success_rate for cmt in completed_results
                ]
                task_group_reward += [
                    np.mean([cmt.reward_structure.final_scalar_reward for cmt in completed_results])
                ]
                task_success_rate += [np.mean(success_rate_array)]
                # whether this group need amendment
                need_amend = all(x == task_cmd_reward_array[0] for x in task_cmd_reward_array)
                # if so, whether we have quota (num_task_to_amend) to amend
                if need_amend and (num_task_to_amend > 0):
                    # this group need amendment, so, we discard all its samples
                    # do not worry, other groups will take up the slack
                    num_task_to_amend -= 1
                    print_buffer += "/(amend)"
                    continue
                else:
                    if need_amend:
                        # this group need amendment, but we simply do not have quota to amend
                        # we just accept rollout_n samples from this group
                        num_to_be_selected = rollout_n
                    else:
                        # this group is good and healthy, if it has extra samples, we accept them
                        num_to_be_selected = rollout_n + avail_extra_cnt
                    # if num_to_be_selected > the number of resulting samples, we choose them to maximum reward diversity
                    selected_cmt_array = self.greedy_max_std_selection(
                        completed_results, num_to_be_selected
                    )
                    # good, we have collected selected samples from this group
                    cmt_array += selected_cmt_array
                    # print info
                    print_buffer += f"/({len(selected_cmt_array)})"
                    if need_amend:
                        print_buffer += "(no-amend)"

            logger.info(print_buffer)

            for cmt in cmt_array:
                # average of gourp success rate
                cmt.current_batch_success_rate = np.mean(task_success_rate)
                # average of gourp average reward
                cmt.current_batch_reward = np.mean(task_group_reward)

            return cmt_array


class ParallelEnvManager(DynamicRolloutManager):
    """High-level manager orchestrating rollouts and batch conversion."""

    def to_dataproto(self, cmt_array) -> DataProto:
        """Convert completed context trackers into a `DataProto` minibatch."""
        samples = self.trajectories_to_samples(cmt_array)
        dataproto = self.samples_to_dataproto(samples)
        return dataproto

    def trajectories_to_samples(self, cmt_array: List[BasicContextTracker]) -> List[Sample]:
        """Tokenize each tracker into `Sample` objects ready for tensorization."""
        sample_arr_final = []
        BasicContextTracker.compute_reference_advantage(cmt_array)
        for cmt in cmt_array:
            try:
                sample_arr = cmt.group_tokenize()
            except Exception as e:
                raise e
            finally:
                cmt.generate_log(global_step=self.current_global_steps)
                if os.environ.get("BEST_LOGGER_PATH", None) and os.environ.get(
                    "ASTUNER_DEBUG", None
                ):
                    logger.success(
                        f"View rollout details at [http://localhost:8181/?path={quote(os.path.abspath(os.environ['BEST_LOGGER_PATH']))}]"
                    )
            sample_arr_final += sample_arr

        if self.config.astuner.backbone in ["verl"]:
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
            if len(sample.prompt_ids) > self.config.astuner.data.max_prompt_length:
                raise RuntimeError(f"Sample has prompt_ids length {len(sample.prompt_ids)} ")

            if len(sample.response_ids) > self.config.astuner.data.max_response_length:
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
        assert max_prompt_length_this_batch <= self.config.astuner.data.max_prompt_length
        max_response_length_this_batch = max([p.shape[-1] for p in response_ids])
        assert max_response_length_this_batch <= self.config.astuner.data.max_response_length

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
