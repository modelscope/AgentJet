"""Standalone reward/advantage simulation for VERL trainer helpers.

Run from the repository root after activating the VERL environment:

    source /mnt/data_cpfs/qingxu.fu/alpha_auto_research/agentjet_codebase/.venv/bin/activate
    python tests/reward_simulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ajet.backbone.trainer_verl import (  # noqa: E402
    compute_advantage,
    compute_episode_level_loss_weight,
    compute_grpo_episode_level_outcome_advantage,
    compute_reward,
    parse_reward_from_dataproto,
)


def build_mock_config() -> SimpleNamespace:
    return SimpleNamespace(
        algorithm=SimpleNamespace(
            adv_estimator=AdvantageEstimator.GRPO,
            gamma=1.0,
            lam=1.0,
            norm_adv_by_std_in_grpo=False,
        ),
        ajet=SimpleNamespace(
            rollout=SimpleNamespace(num_repeat=1),
            trainer_common={
                "advantage_estimation_episode_level": True,
                "loss_weight_normalization_episode_level": True,
            },
        ),
    )


def build_mock_dataproto() -> DataProto:
    prompts = torch.tensor(
        [
            [101, 11, 12],
            [101, 11, 12],
            [101, 21, 22],
            [101, 21, 22],
            [101, 21, 22],
        ],
        dtype=torch.long,
    )
    responses = torch.tensor(
        [
            [31, 32, 0, 0],
            [33, 34, 35, 0],
            [41, 42, 43, 44],
            [45, 46, 0, 0],
            [47, 48, 49, 0],
        ],
        dtype=torch.long,
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    prompt_mask = torch.ones_like(prompts)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    loss_mask = attention_mask.clone()

    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "response_mask": response_mask,
        },
        batch_size=[prompts.shape[0]],
    )

    return DataProto(
        batch=batch,
        non_tensor_batch={
            "uid": np.array(["task_a", "task_a", "task_b", "task_b", "task_b"]),
            "episode_uuids": np.array(
                ["episode_a1", "episode_a2", "episode_b1", "episode_b1", "episode_b2"]
            ),
            "reward_scores": np.array([1.0, 0.0, 0.5, 0.5, -0.5], dtype=np.float32),
        },
    )


def add_token_level_rewards(data: DataProto) -> tuple[torch.Tensor, dict]:
    reward_result = compute_reward(data)
    if isinstance(reward_result, tuple):
        reward_tensor, reward_extra_info = reward_result
    else:
        reward_tensor, reward_extra_info = reward_result, {}
    data.batch["token_level_rewards"] = reward_tensor
    return reward_tensor, reward_extra_info


def compute_mock_advantage(config: SimpleNamespace, episode_level: bool) -> DataProto:
    data = build_mock_dataproto()
    add_token_level_rewards(data)
    return compute_advantage(
        data,
        adv_estimator=config.algorithm.adv_estimator,
        gamma=config.algorithm.gamma,
        lam=config.algorithm.lam,
        num_repeat=config.ajet.rollout.num_repeat,
        norm_adv_by_std_in_grpo=config.algorithm.norm_adv_by_std_in_grpo,
        config=config.algorithm,
        advantage_estimation_episode_level=episode_level,
    )


def main() -> None:
    config = build_mock_config()
    data = build_mock_dataproto()

    print("uid / episode_uuids / reward:")
    for uid, episode_uuid, reward in zip(
        data.non_tensor_batch["uid"],
        data.non_tensor_batch["episode_uuids"],
        data.non_tensor_batch["reward_scores"],
    ):
        print(f"{uid}\t{episode_uuid}\t{reward}")

    parsed_reward = parse_reward_from_dataproto(data)
    reward_tensor, reward_extra_info = add_token_level_rewards(data)

    response_len = data.batch["responses"].shape[-1]
    grpo_mask = data.batch["loss_mask"][:, -response_len:]
    episode_advantages, episode_returns = compute_grpo_episode_level_outcome_advantage(
        token_level_rewards=reward_tensor,
        response_mask=grpo_mask,
        index=data.non_tensor_batch["uid"],
        episode_index=data.non_tensor_batch["episode_uuids"],
        norm_adv_by_std_in_grpo=config.algorithm.norm_adv_by_std_in_grpo,
    )

    episode_level_data = compute_mock_advantage(config, episode_level=True)
    sample_level_data = compute_mock_advantage(config, episode_level=False)
    episode_level_data.batch["loss_weight"] = compute_episode_level_loss_weight(episode_level_data)

    print("\nreward_scores:")
    print(data.non_tensor_batch["reward_scores"])
    print("\nparse_reward_from_dataproto:")
    print(parsed_reward)
    print("\ncompute_reward:")
    print(reward_tensor)
    print(f"reward_extra_info={dict(reward_extra_info)}")
    print("\ncompute_grpo_episode_level_outcome_advantage:")
    print(episode_advantages)
    print("episode_returns_match=", torch.equal(episode_advantages, episode_returns))
    print("\ncompute_advantage advantage_estimation_episode_level=True:")
    print(episode_level_data.batch["advantages"])
    print(
        "returns_match=",
        torch.equal(
            episode_level_data.batch["advantages"], episode_level_data.batch["returns"]
        ),
    )
    print("\ncompute_advantage advantage_estimation_episode_level=False:")
    print(sample_level_data.batch["advantages"])
    print(
        "returns_match=",
        torch.equal(sample_level_data.batch["advantages"], sample_level_data.batch["returns"]),
    )
    print("\ncompute_episode_level_loss_weight:")
    print(episode_level_data.batch["loss_weight"])


if __name__ == "__main__":
    main()
