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

"""
Ajet extensions for verl ActorConfig.
Adds `override_ppo_mini_batch_num` field to control the number of optimizer steps per train-batch-step.
"""

from dataclasses import dataclass, field
from typing import Optional

from verl.workers.config.actor import ActorConfig, FSDPActorConfig


@dataclass
class AjetActorConfig(ActorConfig):
    """ActorConfig extended with ajet-specific fields.

    Additional fields:
        override_ppo_mini_batch_num (Optional[int]): If > 0, overrides ppo_mini_batch_size
            by computing mini_batch_split_size = ceil(batch_size / override_ppo_mini_batch_num).
    """

    override_ppo_mini_batch_num: Optional[int] = None


@dataclass
class AjetFSDPActorConfig(FSDPActorConfig):
    """FSDPActorConfig extended with ajet-specific fields.

    Additional fields:
        override_ppo_mini_batch_num (Optional[int]): If > 0, overrides ppo_mini_batch_size
            by computing mini_batch_split_size = ceil(batch_size / override_ppo_mini_batch_num).
    """

    override_ppo_mini_batch_num: Optional[int] = None
