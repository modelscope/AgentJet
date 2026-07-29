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


from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from verl import DataProto
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import compute_response_mask


def parse_reward_from_dataproto(data: DataProto) -> torch.Tensor:
    """
    Reward scalar -> token-level reward tensor conversion.
    """
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # (bs, reslen)

    def get_response_lengths():
        # Batch-level processing
        prompt_ids_batch = data.batch["prompts"]  # (bs, prompt_len)
        prompt_lengths = prompt_ids_batch.shape[-1]
        # Get attention masks for all items
        attention_masks = data.batch["attention_mask"]  # (bs, total_len)
        response_lengths = attention_masks[:, prompt_lengths:].sum(dim=1)  # (bs, )
        return response_lengths

    # Get scalar reward scores
    reward_scores = torch.tensor(
        [item for item in data.non_tensor_batch["reward_scores"]],
        device=reward_tensor.device, dtype=torch.float32
    )  # (bs, )

    # Use advanced indexing to assign rewards (placing reward at the last token position)
    # e.g.
    # reward_scores = [1,2,3]
    # response_lengths = [7,3,4]
    # reward_tensor = [
    #     [0,0,0,0,0,0,1,0,0],
    #     [0,0,2,0,0,0,0,0,0],
    #     [0,0,0,3,0,0,0,0,0],
    # ]
    response_lengths = get_response_lengths()
    assert len(data) == reward_tensor.shape[0]
    reward_tensor[torch.arange(reward_tensor.shape[0]), response_lengths - 1] = reward_scores

    return reward_tensor


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
    scores = token_level_rewards.sum(dim=-1)    #  (bs, response_length)
    bsz = scores.shape[0]

    with torch.no_grad():
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
