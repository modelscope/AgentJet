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

import torch
from verl import DataProto


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

    # per_sample = tensor([1.0000, 0.3333, 0.3333, 0.3333, 0.5000, 0.5000])
    # broadcast per-sample weight to the same shape as advantages
    weights = per_sample.view(-1, 1) * torch.ones_like(advantages)

    # expected loss_weight:
    # tensor([[1.0000, 1.0000, 1.0000, 1.0000],
    #         [0.3333, 0.3333, 0.3333, 0.3333],
    #         [0.3333, 0.3333, 0.3333, 0.3333],
    #         [0.3333, 0.3333, 0.3333, 0.3333],
    #         [0.5000, 0.5000, 0.5000, 0.5000],
    #         [0.5000, 0.5000, 0.5000, 0.5000]])
    return weights
