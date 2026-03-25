from verl.workers.config import FSDPActorConfig
from dataclasses import dataclass, field


@dataclass
class AgentJetFSDPActorConfig(FSDPActorConfig):
    loss_extra_scale_ratio: float = 1.0
    override_ppo_mini_batch_num: int = 1
