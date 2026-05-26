from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from verl.trainer.config import AlgoConfig


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    # token_level_rewards = [
    #     [0,0,0,0,0,0,1,0,0],
    #     [0,0,2,0,0,0,0,0,0],
    #     [0,0,0,3,0,0,0,0,0],
    # ]
    # --->
    # scores = [1,2,3]
    scores = token_level_rewards.sum(dim=-1)    #  (bs, response_length)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():

        # scores = [1,2,3]
        # --->
        # scores = [(1-2)/1, (2-2)/1, (3-2)/1] = [-1, 0, 1]
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        # --->
        # scores = [
        #     [-1,-1,-1,-1,-1,-1,-1, 0, 0],
        #     [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [ 1, 1, 1, 1, 0, 0, 0, 0, 0],
        # ]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
