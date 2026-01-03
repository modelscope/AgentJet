# Copyright 2025 Alibaba Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import uuid
from collections import defaultdict
from pprint import pprint
from typing import List, Optional

import hydra
import numpy as np
import torch
from beast_logger import print_dict
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    apply_kl_penalty,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from agentscope_tuner.backbone.warm_up import warm_up_process
from agentscope_tuner.context_tracker.basic_tracker import BaseContextTracker
from agentscope_tuner.schema.task import Task
from agentscope_tuner.task_reader import dict_to_astuner_task
from agentscope_tuner.task_rollout.native_parallel_worker import VerlRolloutManager


def parse_reward_from_dataproto(data: DataProto, return_dict=False) -> dict | torch.Tensor:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        return_dict: Whether to return a dictionary or just the reward tensor.

    Returns:
        Tensor of shape (bs, response_len) if return_dict is False,
        or a dict with 'reward_tensor' and 'reward_extra_info'.
    """
    # Within DataFlow, world.execute() will pass a float score, which will be contained in the DataProto.non_tensor_batch('reward_scores')

    # Initialize reward tensor
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # (bs, reslen)
    reward_extra_info = defaultdict(list)

    # Batch-level processing
    prompt_ids_batch = data.batch["prompts"]  # (bs, prompt_len)
    prompt_lengths = prompt_ids_batch.shape[-1]

    # Get attention masks for all items
    attention_masks = data.batch["attention_mask"]  # (bs, total_len)
    response_lengths = attention_masks[:, prompt_lengths:].sum(dim=1)  # (bs, )

    # Get reward scores
    reward_scores_list = [item for item in data.non_tensor_batch["reward_scores"]]
    reward_scores = torch.tensor(
        reward_scores_list, device=reward_tensor.device, dtype=torch.float32
    )  # (bs, )

    # Use advanced indexing to assign rewards (placing reward at the last token position)
    reward_tensor[torch.arange(len(data)), response_lengths - 1] = reward_scores

    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }
    else:
        return reward_tensor


def union_gen_batch_via_task_id(tasks, batch: DataProto, gen_batch_output: DataProto):
    """
    Union the gen_batch_output with the batch based on task_id.
    """
    map_task_id_to_index = {t.task_id: i for i, t in enumerate(tasks)}
    gen_task_task_ids = gen_batch_output.non_tensor_batch["task_ids"]
    indices = [map_task_id_to_index[tid] for tid in gen_task_task_ids]
    batch_extend = batch.select_idxs(indices)
    batch_final = batch_extend.union(gen_batch_output)
    return batch_final


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # If multi-turn, replace the mask with the relevant part of loss_mask
        # Get length from the initial response mask
        response_length = grpo_calculation_mask.size(1)
        # This mask is the one intended for GRPO
        grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class ASTunerRayPPOTrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.
    Slightly modified from RayPPOTrainer in verl.
    """

    # #######################################
    # init
    # #######################################
    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = (
            config.astuner.data.train_batch_size * config.astuner.rollout.num_repeat
        )
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        try:
            actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
            actor_config.validate(
                n_gpus,
                config.astuner.data.train_batch_size,
                config.actor_rollout_ref.model,
            )
        except hydra.errors.InstantiationException:
            raise ValueError(
                "You are using an unsupported VERL version. Please read `documents/backbones.md`"
            )
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.astuner.rollout.log_prob_micro_batch_size,
                config.astuner.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size,
                config.reward_model.micro_batch_size_per_gpu,
                "reward_model",
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.astuner.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.astuner.rollout.val_kwargs.do_sample:
            assert (
                config.astuner.rollout.temperature > 0
            ), "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=critic_cfg
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs[
                "ray_wait_register_center_timeout"
            ] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert (
                OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None
            ), "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        from verl.experimental.agent_loop.agent_loop import (
            AgentLoopManager,
            AsyncLLMServerManager,
        )

        self.async_rollout_mode = True
        agent_loop_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
        )
        self.async_server_list = agent_loop_manager.async_llm_servers
        self.async_rollout_manager = AsyncLLMServerManager(self.config, self.async_server_list)

        self.reward_fn = parse_reward_from_dataproto
        self.val_reward_fn = parse_reward_from_dataproto

        self.parallel_env = VerlRolloutManager(
            config=self.config,
            async_rollout_manager=self.async_rollout_manager,
            max_parallel=self.config.astuner.rollout.max_env_worker,
            tokenizer=self.tokenizer,
        )

    # #######################################
    # training loop
    # #######################################
    def fit(self):  # noqa: C901
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        warm_up_process(self.config)
        self.verl_logger = Tracking(
            project_name=self.config.astuner.project_name,
            experiment_name=self.config.astuner.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # wake and sleep to enforce param sync
        self.async_rollout_manager.wake_up()
        self.async_rollout_manager.sleep()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.verl_logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                batch_dict["index"] = torch.tensor(
                    [i for i in range(len(batch_dict["task_id"]))],
                    dtype=torch.long,
                )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )

                # # pop those keys for generation
                batch_keys_to_pop = ["index"]
                non_tensor_batch_keys_to_pop = [
                    "task_id",
                    "main_query",
                    "env_type",
                    "metadata",
                    "init_messages",
                ]
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    logger.info("=== + rollout step begin ===")
                    with marked_timer("gen", timing_raw, color="red"):
                        assert self.async_rollout_mode
                        logger.info("=== wake up begin ===")
                        self.async_rollout_manager.wake_up()
                        logger.info("=== wake up end ===")
                        tasks: List[Task] = [
                            dict_to_astuner_task(dict(
                                task_id=gen_batch.non_tensor_batch["task_id"][i],
                                main_query=gen_batch.non_tensor_batch["main_query"][i],
                                env_type=gen_batch.non_tensor_batch["env_type"][i],
                                metadata=gen_batch.non_tensor_batch["metadata"][i],
                                init_messages=gen_batch.non_tensor_batch["init_messages"][i],
                            ))
                            for i in range(len(gen_batch))
                        ]
                        logger.info(
                            str(
                                [
                                    gen_batch.non_tensor_batch["task_id"][i]
                                    for i in range(len(gen_batch))
                                ]
                            )
                        )
                        logger.info("=" * 10 + "start fit rollout" + "=" * 10)
                        self.parallel_env.current_global_steps = self.global_steps
                        context_tracker_arr: List[BaseContextTracker] = self.parallel_env.rollout(
                            tasks, mode="sample", epoch=f"train.{epoch}"
                        )
                        logger.info("=" * 10 + "end fit rollout" + "=" * 10)
                        logger.info("begin to convert context_tracker_arr to dataproto")
                        gen_batch_output = self.parallel_env.to_dataproto(context_tracker_arr)
                        logger.info("end convertion")

                        success_rate = [
                            traj.reward_structure.success_rate for traj in context_tracker_arr
                        ]
                        madness_rate = [
                            traj.reward_structure.madness for traj in context_tracker_arr
                        ]
                        # reward = [traj.reward_structure.raw_reward for traj in context_tracker_arr]
                        round_cnt = [traj.round_cnt for traj in context_tracker_arr]
                        metrics.update(
                            {
                                "critic/round_cnt": np.mean(round_cnt),
                                "critic/madness_rate": np.mean(madness_rate),
                                "critic/success_rate": np.mean(success_rate),
                                "critic/real_success_rate": np.mean(
                                    context_tracker_arr[0].current_batch_success_rate
                                ),
                                "critic/real_reward": np.mean(
                                    context_tracker_arr[0].current_batch_reward
                                ),
                            }
                        )
                        if self.config.astuner.execute_test:  # apply a test probe
                            from swanlab.data.run.main import get_run

                            from agentscope_tuner.utils.testing_utils import (
                                _test_if_test_mode,
                            )

                            run_info = get_run().public.json()  # type: ignore
                            data = {
                                "step": self.global_steps,
                                "reward_for_test_robot": metrics["critic/real_reward"],
                                "data_dashboard_url": run_info["cloud"]["experiment_url"],
                            }
                            _test_if_test_mode(key="reward_probe", value=data, config=self.config)

                        logger.info(
                            f"gen_batch_output.info batch.keys={gen_batch_output.batch.keys()}"
                        )
                        self.async_rollout_manager.sleep()
                    logger.info("=== - rollout step end ===")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        raise NotImplementedError("REMAX is not supported in GRPO yet.")

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    batch = union_gen_batch_via_task_id(tasks, batch, gen_batch_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            raise NotImplementedError(
                                "launch_reward_fn_async is not supported in GRPO yet."
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                batch, self.reward_fn
                            )

                    # recompute old_log_probs
                    logger.info("=== + compute log_probs begin ===")
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        assert not torch.isnan(
                            entropy_loss
                        ).item(), "Entropy loss should not be NaN, something must have gone terribly wrong."
                        old_log_prob_metrics = {"actor/entropy": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.astuner.rollout.num_repeat,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = True
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
                )

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                self.verl_logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    # #######################################
    # Validate
    # #######################################
    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_data["index"] = torch.tensor(
                [i for i in range(len(test_data["task_id"]))], dtype=torch.long
            )
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.astuner.rollout.val_kwargs.num_repeat,
                interleave=True,
            )

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            batch_keys_to_pop = ["index"]
            non_tensor_batch_keys_to_pop = [
                "task_id",
                "main_query",
                "env_type",
                "metadata",
                "init_messages",
            ]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            if "extras" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("extras")

            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.astuner.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            logger.info(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            self.async_rollout_manager.wake_up()
            main_val_dataset = self.get_eval_dataset()

            logger.info("=" * 10 + "start validate rollout" + "=" * 10)
            context_tracker_arr, tasks, val_metrics = self.eval_dataset(
                target_dataset=main_val_dataset,
                target_dataset_name="main_val_dataset",
                mode="validate",
                epoch="test.1",
            )
            logger.info("=" * 10 + "end validate rollout" + "=" * 10)
            test_output_gen_batch = self.parallel_env.to_dataproto(context_tracker_arr)
            self.async_rollout_manager.sleep()
            logger.info("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                dtype=object,
            )
            tasks = tasks[: len(main_val_dataset)]
            test_batch = union_gen_batch_via_task_id(tasks, test_batch, test_output_gen_batch)
            # test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            logger.info(
                f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}"
            )
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    logger.info(
                        f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}"
                    )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )
            break  # hack to escape the loop after one batch

        metric_dict = val_metrics

        return metric_dict

    def eval_dataset(self, target_dataset, target_dataset_name, mode, epoch):
        """
        Evaluate a dataset by running rollouts and computing task completion metrics.

        Args:
            target_dataset: The dataset to evaluate
            target_dataset_name: Name for logging purposes
            mode: Evaluation mode ("sample" or "validate")
            epoch: Current epoch for logging

        Returns:
            Tuple of (ctx_trackers, tasks) containing trajectory results and task definitions
        """
        pass_n = self.config.astuner.trainer_common.val_pass_n

        tasks = []
        for _ in range(pass_n):
            tasks += [task for task in target_dataset]

        ctx_trackers = self.parallel_env.rollout(
            tasks=tasks, mode=mode, epoch=epoch
        )  # "sample" or "validate"
        task_results = {}
        for ctx_tracker in ctx_trackers:
            reward = ctx_tracker.reward_structure.raw_reward
            task_id = ctx_tracker.task_id
            if task_id not in task_results:
                task_results[task_id] = {}
                task_results[task_id]["reward_arr"] = []
                task_results[task_id]["tag_arr"] = []
            if reward >= 1:
                ctx_tracker.tag = "success"
            elif reward == 0:
                ctx_tracker.tag = "failure"
            else:
                ctx_tracker.tag = "half_success"
            task_results[task_id]["tag_arr"] += [ctx_tracker.tag]
            task_results[task_id]["reward_arr"] += [ctx_tracker.reward_structure.raw_reward]
            task_results[task_id]["scenario"] = task_id.split("_")[0]

        repeated_success_tasks = 0
        num_all_success_tasks = 0  # number of tasks that is successful among all n attempts
        num_pass_n_tasks = 0  # number of tasks that is successful at least once among n attempts
        for task_id, task_outcomes in task_results.items():
            # Calculate num_all_success_tasks  # The number of tasks where all were successful in n experiments
            # Calculate num_pass_n_tasks       # The number of tasks where at least one was successful in n experiments
            assert len(task_outcomes["tag_arr"]) == pass_n
            if all(tag == "success" for tag in task_outcomes["tag_arr"]):
                num_all_success_tasks += 1
            if any(tag == "success" for tag in task_outcomes["tag_arr"]):
                num_pass_n_tasks += 1
            repeated_success_tasks += task_outcomes["tag_arr"].count("success")

        # record logs
        for ctx_tracker in ctx_trackers:
            ctx_tracker.generate_log()

        rewards = [ctx_tracker.reward_structure.raw_reward for ctx_tracker in ctx_trackers]
        num_tasks = len(task_results)
        assert num_tasks == len(ctx_trackers) // pass_n

        val_metrics = {
            "target dataset name": target_dataset_name,
            "pass_n": pass_n,
            "total_tasks": len(task_results),
            "num_all_success_tasks": num_all_success_tasks,
            f"num_pass_n_tasks(pass@{pass_n})": num_pass_n_tasks,
            "TGC@1": repeated_success_tasks / (num_tasks * pass_n),
            f"TGC@{pass_n}": num_pass_n_tasks / num_tasks,
            f"TGC@{pass_n}-all-pass": num_all_success_tasks / num_tasks,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
        }
        print_dict(
            val_metrics,
            narrow=True,
            header=target_dataset_name,
            mod="evaluation",
        )

        self.verl_logger.log(data=val_metrics, step=self.global_steps)

        return ctx_trackers, tasks, val_metrics

    def get_eval_dataset(self):
        from agentscope_tuner.task_reader import RouterTaskReader

        task_reader = RouterTaskReader(
            self.config.astuner.task_reader.type,
            self.config.astuner.task_reader,
        )
        tasks = task_reader.get_validation_tasks()
        self.main_val_dataset = tasks
        return self.main_val_dataset
