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


import torch
from verl import DataProto


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
