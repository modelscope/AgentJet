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


import asyncio
import json
import os
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Any, List, Optional

import hydra
import numpy as np
import torch
from beast_logger import print_dict
from loguru import logger
from tqdm import tqdm
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
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
    apply_kl_penalty,
    compute_response_mask,
)
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.ray_utils import auto_await

from ajet.backbone.warm_up import warm_up_process
from ajet.context_tracker.single_agent_tracking import SingleAgentContextTracker
from ajet.schema.task import Task
from ajet.task_reader import dict_to_ajet_task
from ajet.task_rollout.native_parallel_worker import VerlRolloutManager
from ajet.utils.metric_helper import save_trajectory_as_json_file, update_metrics


IMAGE_PAD_ID = 151655  # Qwen2.5-VL <|image_pad|>


def capture_training_token_ids(batch: DataProto, tokenizer, out_path: str, global_step: int = 0) -> None:
    """One-shot debug dump of the *training-side* token ids for a batch.

    Writes each sample's real (unpadded) prompt token id array, tagging the
    image-pad (151655) tokens, plus per-sample image counts and grid_thw. Used
    by ``tutorial/claudecode_geo3k_swarm/test_token_ids.py`` to compare the
    training-side tokenization against the vLLM-side capture in
    ``token_id_io.md``. Inert during normal training (gated on an env var by
    the caller).
    """
    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    attention_mask = batch.batch["attention_mask"]
    prompt_len = prompts.shape[1]
    resp_len = responses.shape[1]
    mmi = batch.non_tensor_batch.get("multi_modal_inputs", None)
    task_ids = batch.non_tensor_batch.get("task_ids", None)

    def _real_prompt(i):
        mask = attention_mask[i, :prompt_len].bool()
        return prompts[i][mask].tolist()

    def _real_response(i):
        mask = attention_mask[i, prompt_len:prompt_len + resp_len].bool()
        return responses[i][mask].tolist()

    def _grid(i):
        if mmi is None:
            return None
        d = mmi[i]
        if not isinstance(d, dict):
            return None
        g = d.get("image_grid_thw", None)
        if g is None:
            return None
        try:
            return g.tolist()
        except AttributeError:
            return g

    n = len(prompts)

    def _sample_md(idx: int, header_note: str) -> str:
        """Full markdown dump (summary row + arrays + decoded) for one sample."""
        p = _real_prompt(idx)
        r = _real_response(idx)
        tid = task_ids[idx] if task_ids is not None else "?"
        lines = [
            "# Training-side token ids captured at trainer_verl.py:599",
            "",
            "Final tokenization consumed by training (image already expanded to "
            f"`<|image_pad|>` = {IMAGE_PAD_ID}). Compare against the vLLM-side "
            "ground-truth capture.",
            "",
            f"- global_step: {global_step}",
            f"- {header_note}",
            f"- task_id: {tid}",
            f"- image_tokens: {p.count(IMAGE_PAD_ID)}",
            f"- grid_thw: {_grid(idx)}",
            "",
            f"## Full arrays for sample idx={idx}",
            "",
            f"Input (prompt) token ids — length {len(p)}, image_tokens {p.count(IMAGE_PAD_ID)}:",
            "",
            "```json",
            json.dumps(p),
            "```",
            "",
            f"Output (response) token ids — length {len(r)}:",
            "",
            "```json",
            json.dumps(r),
            "```",
            "",
        ]
        if tokenizer is not None:
            try:
                lines += [
                    "Decoded prompt (special tokens kept):",
                    "",
                    "```",
                    tokenizer.decode(p, skip_special_tokens=False),
                    "```",
                    "",
                ]
            except Exception:
                pass
        return "\n".join(lines)

    # Directory mode: AJET_CAPTURE_TOKEN_IDS points at a directory -> write one
    # <task_id>.debug.md per distinct task_id (first sample of each), so the
    # multimodal test can compare each case independently. File mode keeps the
    # original single-file behavior (first image-bearing sample) for the
    # existing single-image test_token_ids.py.
    is_dir = os.path.isdir(out_path) or out_path.endswith(("/", os.sep))
    if is_dir:
        os.makedirs(out_path, exist_ok=True)
        seen = set()
        for i in range(n):
            tid = str(task_ids[i]) if task_ids is not None else str(i)
            if tid in seen:
                continue
            seen.add(tid)
            safe = tid.replace("/", "_").replace(os.sep, "_")
            with open(os.path.join(out_path, f"{safe}.debug.md"), "w", encoding="utf-8") as f:
                f.write(_sample_md(i, f"batch size: {n} (per-task capture)"))
        return

    chosen = None
    for i in range(n):
        if _real_prompt(i).count(IMAGE_PAD_ID) > 0:
            chosen = i
            break
    if chosen is None:
        chosen = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_sample_md(chosen, f"batch size: {n} (first image-bearing sample)"))


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


def compute_reward(data: DataProto) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """

    reward_result = parse_reward_from_dataproto(data, return_dict=True)
    reward_tensor = reward_result["reward_tensor"]  # type: ignore
    reward_extra_infos_dict = reward_result.get("reward_extra_info", {})  # type: ignore
    return reward_tensor, reward_extra_infos_dict


def union_gen_batch_via_task_id(tasks, batch: DataProto, gen_batch_output: DataProto, discard_original_batch=False):
    """
    Union the gen_batch_output with the batch based on task_id.
    """
    if not discard_original_batch:
        map_task_id_to_index = {t.task_id: i for i, t in enumerate(tasks)}
        gen_task_task_ids = gen_batch_output.non_tensor_batch["task_ids"]
        indices = [map_task_id_to_index[tid] for tid in gen_task_task_ids]
        batch_extend = batch.select_idxs(indices)
        batch_final = batch_extend.union(gen_batch_output)
        return batch_final
    else:
        gen_batch_output.non_tensor_batch['uid'] = gen_batch_output.non_tensor_batch["task_ids"]
        task_id_counter = {}
        for i, tid in enumerate(gen_batch_output.non_tensor_batch["task_ids"]):
            if tid in task_id_counter:
                task_id_counter[tid] += 1
            else:
                task_id_counter[tid] = 1
        logger.info(f'task_id_counter: {task_id_counter}')
        return gen_batch_output

def import_or_export_data_proto(batch: DataProto, direction: str = "export", file: str = "./tmp.pkl") -> DataProto:
    """Import or export a DataProto batch to/from a pickle file.

    Args:
        batch: The DataProto batch object. Used when direction is "export";
               ignored (can be None) when direction is "import".
        direction: "import" to load a batch from file, "export" to save the batch to file.
        file: Path to the pickle file. Defaults to "./tmp.pkl".

    Returns:
        The DataProto batch — either the one just loaded (import) or the one just saved (export).

    Raises:
        ValueError: If direction is not "import" or "export".
        FileNotFoundError: If direction is "import" and the file does not exist.
    """
    import pickle
    if direction == "export":
        with open(file, "wb") as f:
            pickle.dump(batch, f)
        logger.info(f"[import_or_export_data_proto] Exported batch to {file}")
        return batch
    elif direction == "import":
        with open(file, "rb") as f:
            batch = pickle.load(f)
        logger.info(f"[import_or_export_data_proto] Imported batch from {file}")
        return batch
    else:
        raise ValueError(f"direction must be 'import' or 'export', got '{direction}'")

def compute_grpo_episode_level_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    episode_index: np.ndarray,
    norm_adv_by_std_in_grpo: bool = True,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO outcome advantage with the baseline computed at *episode* scope.

    Mirrors ``verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage`` but,
    instead of treating every sample equally when forming the per-task (``uid``)
    baseline, it first reduces every episode (``episode_uuids``) to its mean
    scalar reward and then computes the task baseline mean/std over those
    per-episode means. This way an episode that produced many samples does not
    dominate the baseline of an episode that produced few.

    Example (matches the documented behaviour):
        task T -> episode 1 (2 samples, reward 1) + episode 2 (1 sample, reward 0)
        sample scope baseline  = (1 + 1 + 0) / 3 = 0.667
        episode scope baseline = (mean[1, 1] + mean[0]) / 2 = (1 + 0) / 2 = 0.5

    Args:
        token_level_rewards: (bsz, response_length) reward tensor.
        response_mask: (bsz, response_length) mask of trainable response tokens.
        index: per-sample task id (``non_tensor_batch["uid"]``).
        episode_index: per-sample episode id (``non_tensor_batch["episode_uuids"]``).
        norm_adv_by_std_in_grpo: divide the centred reward by the (episode-level)
            group std when True, otherwise only subtract the group mean.
        epsilon: numerical-stability term added to the std denominator.

    Returns:
        (advantages, returns) - both (bsz, response_length); identical, as in GRPO.
    """
    scores = (token_level_rewards * response_mask).sum(dim=-1)  # (bsz,) scalar reward
    bsz = scores.shape[0]

    # 1) reduce each episode to its mean scalar reward
    episode_score_sum: dict = defaultdict(float)
    episode_score_cnt: dict = defaultdict(int)
    for i in range(bsz):
        ep = episode_index[i]
        episode_score_sum[ep] += scores[i].item()
        episode_score_cnt[ep] += 1
    episode_mean = {ep: episode_score_sum[ep] / episode_score_cnt[ep] for ep in episode_score_sum}

    # 2) collect, per task, the set of distinct episodes it produced
    task2episodes: dict = defaultdict(dict)  # use dict as ordered set
    for i in range(bsz):
        task2episodes[index[i]][episode_index[i]] = None

    # 3) per-task baseline = mean/std over the per-episode means.
    #    Single-episode tasks are degenerate -> follow verl's convention
    #    (mean=0, std=1) so the advantage reduces to the raw score.
    task_mean: dict = {}
    task_std: dict = {}
    for task, episodes in task2episodes.items():
        vals = torch.tensor([episode_mean[ep] for ep in episodes], dtype=torch.float32)
        if vals.numel() == 1:
            task_mean[task] = torch.tensor(0.0)
            task_std[task] = torch.tensor(1.0)
        else:
            task_mean[task] = vals.mean()
            task_std[task] = vals.std()

    # 4) centre (and optionally normalise) every sample against its task baseline
    adv = scores.clone()
    for i in range(bsz):
        task = index[i]
        if norm_adv_by_std_in_grpo:
            adv[i] = (scores[i] - task_mean[task]) / (task_std[task] + epsilon)
        else:
            adv[i] = scores[i] - task_mean[task]

    adv = adv.unsqueeze(-1) * response_mask
    return adv, adv


def compute_episode_level_loss_weight(data: DataProto) -> torch.Tensor:
    """Per-token loss weight that makes every episode contribute equally.

    Each sample belonging to an episode (same ``non_tensor_batch["episode_uuids"]``)
    that produced ``N`` samples receives weight ``1 / N``. The weights of all
    samples of one episode therefore sum to 1, so an episode that emitted many
    samples does not contribute more to the loss than one that emitted few.

    The weight is broadcast across the response dimension so it has the **same
    shape as ``advantages``** ((bsz, response_length)); this lets it multiply
    both the per-token policy-gradient term and the per-token KL term directly.

    Returns:
        A (bsz, response_length) tensor (matching ``data.batch["advantages"]``
        dtype/device) of per-token loss weights, constant along the response
        dimension for a given sample.
    """
    episode_index = data.non_tensor_batch["episode_uuids"]
    bsz = len(episode_index)
    episode_count: dict = defaultdict(int)
    for ep in episode_index:
        episode_count[ep] += 1
    advantages = data.batch["advantages"]  # (bsz, response_length)
    per_sample = torch.tensor(
        [1.0 / episode_count[episode_index[i]] for i in range(bsz)],
        dtype=advantages.dtype,
        device=advantages.device,
    )
    # broadcast per-sample weight to the same shape as advantages
    weights = per_sample.view(-1, 1) * torch.ones_like(advantages)
    return weights


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    advantage_estimation_episode_level: bool = False,
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
        advantage_estimation_episode_level (bool, optional): When True (and using the GRPO estimator),
            the GRPO baseline is computed at episode scope instead of sample scope so every episode
            contributes equally regardless of how many samples it produced. Defaults to False.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    if advantage_estimation_episode_level and adv_estimator != AdvantageEstimator.GRPO:
        raise NotImplementedError(
            "ajet.trainer_common.advantage_estimation_episode_level is only "
            f"supported with the GRPO advantage estimator, got {adv_estimator}."
        )
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
        if advantage_estimation_episode_level:
            # Episode-scope baseline: every episode contributes equally to the
            # per-task baseline regardless of how many samples it produced.
            if "episode_uuids" not in data.non_tensor_batch:
                raise KeyError(
                    "advantage_estimation_episode_level is enabled but "
                    "non_tensor_batch['episode_uuids'] is missing; cannot identify "
                    "same-episode samples."
                )
            advantages, returns = compute_grpo_episode_level_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                episode_index=data.non_tensor_batch["episode_uuids"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        else:
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


class AjetRayPPOTrainer(RayPPOTrainer):
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
            config.ajet.data.train_batch_size * config.ajet.rollout.num_repeat
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
                config.ajet.data.train_batch_size,
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
                config.ajet.rollout.log_prob_micro_batch_size,
                config.ajet.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )


        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            logger.warning("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.ajet.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            logger.warning(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.ajet.rollout.val_kwargs.do_sample:
            assert (
                config.ajet.rollout.temperature > 0
            ), "validation gen temperature should be greater than 0 when enabling do_sample"

        logger.success("[validate_config] All configuration checks passed successfully!")

    def init_workers(self):
        super().init_workers()

        self.reward_fn = parse_reward_from_dataproto
        self.val_reward_fn = parse_reward_from_dataproto

        assert hasattr(self.async_rollout_manager, "agent_loop_workers")
        assert len(self.async_rollout_manager.agent_loop_workers) == 1, "Please set `num_workers = 1` in `ajet/default_config/verl/verl_default.yaml`"

        servers = list(zip(self.async_rollout_manager.server_addresses, self.async_rollout_manager.server_handles, strict=True))
        real_async_rollout_manager: AsyncLLMServerManager  = AsyncLLMServerManager(
            config = self.async_rollout_manager.config,
            servers = servers,
            load_balancer_handle = self.async_rollout_manager.global_load_balancer
        )

        self.parallel_env = VerlRolloutManager(
            config=self.config,
            async_rollout_manager=real_async_rollout_manager,
            max_parallel=self.config.ajet.rollout.max_env_worker,
            tokenizer=self.tokenizer,
            processor=getattr(self, "processor", None),
        )

    def _update_interchange_server_status_flag(self, status: str):
        if self.config.ajet.enable_interchange_server:
            if self.config.ajet.enable_swarm_mode:
                from ajet.tuner_lib.experimental.interchange_utils import http_change_engine_status
                http_change_engine_status(self.config, status, global_step=self.global_steps)

    @auto_await
    async def _sleep_rollout_replicas(self):
        await asyncio.gather(*[replica.abort_all_requests() for replica in self.checkpoint_manager.replicas])
        await self.checkpoint_manager.sleep_replicas()

    # #######################################
    # training loop
    # #######################################
    def fit(self):  # noqa: C901

        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        warm_up_process(self.config)

        self.verl_logger = Tracking(
            project_name=self.config.ajet.project_name,
            experiment_name=self.config.ajet.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        self._sleep_rollout_replicas()

        # [oc] swarm_mode is not compatible with `val_before_train` and `val_only`
        assert not (self.config.ajet.enable_swarm_mode and (self.config.ajet.trainer_common.val_before_train or self.config.ajet.trainer_common.val_only)), \
            "swarm_mode is not compatible with `val_before_train` and `val_only`"


        # perform validation before training
        # currently, we only support validation using the reward_function.
        if (self.val_reward_fn is not None) and (self.config.ajet.trainer_common.val_before_train) and (not self.config.ajet.enable_swarm_mode):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            self.verl_logger.log(data=val_metrics, step=self.global_steps)
            val_print_to_markdown_file_path = self.config.ajet.trainer_common.val_print_to_markdown_file_path
            if val_print_to_markdown_file_path:
                os.makedirs(os.path.dirname(val_print_to_markdown_file_path), exist_ok=True)
                with open(val_print_to_markdown_file_path, mode="a+") as f:
                    f.write(str(val_metrics))
                    f.write('\n')
            if self.config.ajet.trainer_common.val_only:
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

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}


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
                    logger.info("rollout step begin")
                    with marked_timer("gen", timing_raw, color="red"):
                        # assert self.async_rollout_mode
                        logger.info("wake up begin")
                        self.checkpoint_manager.update_weights(self.global_steps)
                        self._update_interchange_server_status_flag("ENGINE.ROLLING")
                        logger.info("wake up end")
                        tasks: List[Task] = [
                            dict_to_ajet_task(dict(
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
                        logger.info("start batch rollout")
                        self.parallel_env.current_global_steps = self.global_steps
                        # rollout stage begin ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
                        context_tracker_arr: List[SingleAgentContextTracker] = self.parallel_env.rollout(
                            tasks, mode="sample", epoch=f"train.{epoch}"
                        )

                        logger.info("end batch rollout")
                        gen_batch_output = self.parallel_env.to_dataproto(context_tracker_arr)
                        logger.info("end dataproto convertion")

                        success_rate = [
                            traj.reward_structure.success_rate for traj in context_tracker_arr
                        ]
                        madness_rate = [
                            traj.reward_structure.madness for traj in context_tracker_arr
                        ]
                        reward = [traj.reward_structure.raw_reward for traj in context_tracker_arr]
                        llm_call_cnt = [traj.llm_call_cnt for traj in context_tracker_arr]
                        metrics.update(
                            {
                                "critic/llm_call_cnt": np.mean(llm_call_cnt),
                                "critic/madness_rate": np.mean(madness_rate),
                                "critic/success_rate": np.mean(success_rate),
                                "critic/real_success_rate": np.mean(
                                    context_tracker_arr[0].current_batch_success_rate
                                ),
                                "critic/real_reward": np.mean(
                                    context_tracker_arr[0].current_batch_reward
                                ),
                                "critic/reward_std": np.std(reward) if reward else 0,
                            }
                        )
                        save_trajectory_as_json_file(context_tracker_arr, self.global_steps, self.config, prefix="train")
                        update_metrics(context_tracker_arr, metrics, prefix="train_")
                        if self.config.ajet.execute_test:  # apply a test probe
                            from swanlab.data.run.main import get_run

                            from ajet.utils.testing_utils import _test_if_test_mode

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
                        self._update_interchange_server_status_flag("ENGINE.WEIGHT_SYNCING")
                        self._sleep_rollout_replicas()
                    logger.info("rollout step end")

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    discard_original_batch = self.config.ajet.enable_swarm_mode
                    batch = union_gen_batch_via_task_id(tasks, batch, gen_batch_output, discard_original_batch)
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
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch)

                    # one-shot debug: dump the training-side token ids (with the
                    # image expanded to <|image_pad|>) so they can be compared
                    # against the vLLM-side capture in token_id_io.md. Inert
                    # unless AJET_CAPTURE_TOKEN_IDS is set.
                    _capture_path = os.environ.get("AJET_CAPTURE_TOKEN_IDS", "")
                    if _capture_path:
                        try:
                            capture_training_token_ids(
                                batch, self.tokenizer, _capture_path, self.global_steps
                            )
                        except Exception:
                            logger.bind(exception=True).exception(
                                "capture_training_token_ids failed"
                            )

                    # recompute old_log_probs
                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

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
                        batch.batch["token_level_scores"] = reward_tensor   # from compute_reward

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        # [AJET] episode-scope advantage baseline (disabled by default)
                        advantage_estimation_episode_level = bool(
                            self.config.ajet.trainer_common.get(
                                "advantage_estimation_episode_level", False
                            )
                        )

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.ajet.rollout.num_repeat,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                            advantage_estimation_episode_level=advantage_estimation_episode_level,
                        )

                        # [AJET] per-sample loss weight that makes every episode
                        # contribute equally to the policy-gradient update
                        # (disabled by default). Consumed in
                        # AjetDataParallelPPOActor.update_policy.
                        if bool(
                            self.config.ajet.trainer_common.get(
                                "loss_weight_normalization_episode_level", False
                            )
                        ):
                            if "episode_uuids" not in batch.non_tensor_batch:
                                raise KeyError(
                                    "loss_weight_normalization_episode_level is enabled but "
                                    "non_tensor_batch['episode_uuids'] is missing; cannot "
                                    "identify same-episode samples."
                                )
                            batch.batch["loss_weight"] = compute_episode_level_loss_weight(batch)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                        and (not self.config.ajet.enable_swarm_mode)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        val_print_to_markdown_file_path = self.config.ajet.trainer_common.val_print_to_markdown_file_path
                        if val_print_to_markdown_file_path:
                            os.makedirs(os.path.dirname(val_print_to_markdown_file_path), exist_ok=True)
                            with open(val_print_to_markdown_file_path, mode="a+") as f:
                                f.write(str(val_metrics))
                                f.write('\n')

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
                            logger.info("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()


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
                train_print_to_markdown_file_path = self.config.ajet.trainer_common.train_print_to_markdown_file_path
                if train_print_to_markdown_file_path:
                    os.makedirs(os.path.dirname(train_print_to_markdown_file_path), exist_ok=True)
                    with open(train_print_to_markdown_file_path, mode="a+") as f:
                        f.write(str(metrics))
                        f.write('\n')
                progress_bar.update(1)
                self.global_steps += 1

                # # when enabled oai request interchange, we need to clear the cache from time to time
                # if self.config.ajet.enable_interchange_server:
                #     from ajet.tuner_lib.experimental.oai_model_server import ensure_dat_interchange_server_cache_clear
                #     ensure_dat_interchange_server_cache_clear()

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
                repeat_times=self.config.ajet.trainer_common.val_pass_n,
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
                "do_sample": self.config.ajet.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            logger.info(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            self.checkpoint_manager.update_weights(self.global_steps)
            main_val_dataset = self.get_val_dataset()

            logger.info("Starting validate rollout")
            context_tracker_arr, tasks, val_metrics = self._rollout_val_dataset(
                target_dataset=main_val_dataset,
                target_dataset_name="main_val_dataset",
                mode="validate",
                epoch="test.1",
            )
            logger.info("Completed validate rollout")
            test_output_gen_batch = self.parallel_env.to_dataproto(context_tracker_arr)
            self._sleep_rollout_replicas()

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
            discard_original_batch = self.config.ajet.enable_swarm_mode
            test_batch = union_gen_batch_via_task_id(tasks, test_batch, test_output_gen_batch, discard_original_batch)
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

    def _rollout_val_dataset(self, target_dataset, target_dataset_name, mode, epoch):
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
        pass_n = self.config.ajet.trainer_common.val_pass_n

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
            assert len(task_outcomes["tag_arr"]) == pass_n, f"expect {pass_n} attempts, but got {len(task_outcomes['tag_arr'])} attempts for task_id={task_id}."
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
            "global_steps": self.global_steps,
            "pass_n": pass_n,
            "total_tasks": len(task_results),
            "num_all_success_tasks": num_all_success_tasks,
            f"num_pass_n_tasks(pass@{pass_n})": num_pass_n_tasks,
            "task_pass_rate@1": repeated_success_tasks / (num_tasks * pass_n),
            f"task_pass_rate@{pass_n}": num_pass_n_tasks / num_tasks,
            f"task_pass_rate@{pass_n}-all-pass": num_all_success_tasks / num_tasks,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
            "std_reward": np.std(rewards) if rewards else 0,
        }
        for k in [2, 4, 8, 16]:
            if pass_n > k:
                num_pass_k = 0
                for task_id, task_outcomes in task_results.items():
                    if any(tag == "success" for tag in task_outcomes["tag_arr"][:k]):
                        num_pass_k += 1
                val_metrics[f"task_pass_rate@{k}"] = num_pass_k / num_tasks

        save_trajectory_as_json_file(ctx_trackers, self.global_steps, self.config, prefix="eval")
        update_metrics(ctx_trackers, val_metrics, prefix="eval_")
        print_dict(
            val_metrics,
            narrow=True,
            header=target_dataset_name,
            mod="evaluation",
        )

        self.verl_logger.log(data=val_metrics, step=self.global_steps)
        val_metrics.update({"target_dataset_name": target_dataset_name})

        return ctx_trackers, tasks, val_metrics

    def get_val_dataset(self):
        from ajet.task_reader import RouterTaskReader

        task_reader = RouterTaskReader(
            self.config.ajet.task_reader.type,
            self.config.ajet.task_reader,
        )
        tasks = task_reader.get_validation_tasks()

        # clip validation tasks if val_max_num_task_each_validation is set
        val_max_num_task = self.config.ajet.trainer_common.val_max_num_task_each_validation
        if val_max_num_task is not None and len(tasks) > val_max_num_task:
            original_size = len(tasks)
            clip_method = self.config.ajet.trainer_common.val_max_num_task_clip_method
            if clip_method == "fix_seed_random_n":
                rng = np.random.RandomState(seed=42)
                indices = rng.choice(len(tasks), val_max_num_task, replace=False)
                tasks = [tasks[i] for i in sorted(indices)]
            elif clip_method == "random_n":
                indices = np.random.choice(len(tasks), val_max_num_task, replace=False)
                tasks = [tasks[i] for i in sorted(indices)]
            elif clip_method == "first_n":
                tasks = tasks[:val_max_num_task]
            else:
                raise ValueError(f"Unknown val_max_num_task_clip_method: {clip_method}, expected 'fix_seed_random_n', 'random_n', or 'first_n'")
            logger.info(f"Clipped validation dataset from {original_size} to {val_max_num_task} tasks using '{clip_method}'")

        self.main_val_dataset = tasks
        return self.main_val_dataset
