"""Standalone simulation for episode-level loss weights.

Run from the repository root after activating the VERL environment:

    source /mnt/data_cpfs/qingxu.fu/alpha_auto_research/agentjet_codebase/.venv/bin/activate
    python tests/episode_loss_weight_simulation.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ajet.backbone.trainer_verl import compute_episode_level_loss_weight  # noqa: E402


def build_mock_dataproto() -> DataProto:
    advantages = torch.tensor(
        [
            [0.1, 0.2, 0.0, 0.0],
            [0.3, 0.4, 0.5, 0.0],
            [0.6, 0.7, 0.8, 0.9],
            [1.0, 1.1, 0.0, 0.0],
            [1.2, 1.3, 1.4, 0.0],
            [1.5, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    episode_uuids = np.array(
        [
            "episode_a",
            "episode_b",
            "episode_b",
            "episode_b",
            "episode_c",
            "episode_c",
        ]
    )

    return DataProto(
        batch=TensorDict({"advantages": advantages}, batch_size=[advantages.shape[0]]),
        non_tensor_batch={"episode_uuids": episode_uuids},
    )


def expected_loss_weight(data: DataProto) -> torch.Tensor:
    episode_uuids = data.non_tensor_batch["episode_uuids"]
    episode_counts = Counter(episode_uuids)
    advantages = data.batch["advantages"]
    per_sample = expected_per_sample_weight(data)
    return per_sample.view(-1, 1).expand_as(advantages)


def expected_per_sample_weight(data: DataProto) -> torch.Tensor:
    episode_uuids = data.non_tensor_batch["episode_uuids"]
    episode_counts = Counter(episode_uuids)
    advantages = data.batch["advantages"]
    return torch.tensor(
        [1.0 / episode_counts[episode_uuid] for episode_uuid in episode_uuids],
        dtype=advantages.dtype,
        device=advantages.device,
    )


def main() -> None:
    data = build_mock_dataproto()
    actual = compute_episode_level_loss_weight(data)
    expected = expected_loss_weight(data)
    per_sample = expected_per_sample_weight(data)

    print("episode_uuids:")
    print(data.non_tensor_batch["episode_uuids"])
    print("\nepisode_counts:")
    print(dict(Counter(data.non_tensor_batch["episode_uuids"])))
    print("\nadvantages:")
    print(data.batch["advantages"])
    print("\nper_sample:")
    print(per_sample)
    print("\nper_sample.view(-1, 1):")
    print(per_sample.view(-1, 1))
    print("\nactual loss_weight:")
    print(actual)
    print("\nexpected loss_weight:")
    print(expected)

    assert actual.shape == data.batch["advantages"].shape
    assert actual.dtype == data.batch["advantages"].dtype
    assert actual.device == data.batch["advantages"].device
    assert torch.allclose(actual, expected)
    print("\nplain_assertions_passed=True")


if __name__ == "__main__":
    main()
