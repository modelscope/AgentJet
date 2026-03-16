from verl.workers.config.rollout import MultiTurnConfig
from dataclasses import dataclass, field
from typing import Optional
from verl.base_config import BaseConfig


@dataclass
class AjetMultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_sample_per_task: int = 30
    max_steps: int = 30
    expected_steps: Optional[int] = None
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    interaction_config_path: Optional[str] = None
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"
    num_repeat_rollouts: Optional[int] = None


