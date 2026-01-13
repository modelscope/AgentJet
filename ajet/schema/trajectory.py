from typing import Any, Dict, List, Union

import numpy as np
from pydantic import BaseModel, Field
from loguru import logger


class Reward(BaseModel):
    # raw reward: the original reward from environment
    raw_reward: float = Field(default=0.0)
    # raw step reward: the original step-wise rewards from environment
    raw_step_reward: Union[List[float], None] = Field(default=[])
    # step reward: reward after post-processing, e.g., repeatition penalty
    step_reward_arr: List[float] = Field(default=[])
    # advantage values: mean reward is group-wise averaged. e.g. ( (r11+r12+r13)/3 , (r21+r22)/2 ) / 2
    step_advantage: List[float] = Field(default=[])
    # simple advantage values: mean reward is sample-wise averaged. e.g. (r11, r12, r13, r21, r22) / 5
    step_advantage_simple: List[float] = Field(default=[])
    # the success or not, either 0 or 1. average multiple samples to get success rate
    success_rate: float = Field(default=0.0)
    # llm produce abnormal or illegal output, such as ever repeating the same sentence
    madness: float = Field(default=0.0)
    # description of the reward
    description: str = Field(default="Outcome 1 denotes success, and 0 denotes failure.")
    # metadata for reward
    metadata: dict = Field(default_factory=dict)

    @property
    def performance_reward(self):
        # performance reward is only used in dynamic rollout
        # used to terminate hopeless rollout thread early
        # this reward is NOT used in training
        if (self.step_reward_arr is not None) and len(self.step_reward_arr) > 0:
            res = np.mean(self.step_reward_arr)
            return res
        else:
            return self.raw_reward

    @property
    def final_scalar_reward(self):
        # to compute scalar reward, we average step_reward_arr
        reward = self.step_reward_arr
        reward = float(np.mean(reward))
        return reward


class Sample(BaseModel):
    """The data model for single sample."""

    task_batch_index: int = 0
    task_id: str = ""
    task_tag: str = ""
    messages: List[dict] = []
    extras: Dict[str, Any] = {}
    input_ids: List[int] = []
    prompt_ids: List[int] = []
    response_ids: List[int] = []
    attention_mask: List[int] = []
    prompt_attention_mask: List[int] = []
    response_attention_mask: List[int] = []
    position_ids: List[int] = []
    prompt_position_ids: List[int] = []
    response_position_ids: List[int] = []
    loss_mask: List[int] = []
    prompt_loss_mask: List[int] = []
    response_loss_mask: List[int] = []
    logprobs: List[float] = []
    prompt_logprobs: List[float] = []
    response_logprobs: List[float] = []
    max_model_len: int = -1
    max_prompt_len: int = -1
    max_response_len: int = -1
    step_reward: float = 0.0
    reference_advantage: float = 0.0

    def __init__(self, tracker_tokenized: dict, messages, config, **kwargs):
        super().__init__(**kwargs)

        self.max_prompt_len = config.ajet.data.max_prompt_length
        self.max_response_len = config.ajet.data.max_response_length
        self.max_model_len = (
            config.ajet.data.max_response_length + config.ajet.data.max_prompt_length
        )

        self.input_ids = tracker_tokenized["input_ids"]
        self.attention_mask = tracker_tokenized["attention_mask"]
        self.loss_mask = tracker_tokenized["loss_mask"]
        self.position_ids = tracker_tokenized["position_ids"]
        self.logprobs = tracker_tokenized["logprobs"]

        self.prompt_ids = tracker_tokenized["prompt_ids"]
        self.prompt_attention_mask = tracker_tokenized["prompt_attention_mask"]
        self.prompt_loss_mask = tracker_tokenized["prompt_loss_mask"]
        self.prompt_position_ids = tracker_tokenized["prompt_position_ids"]
        self.prompt_logprobs = tracker_tokenized["prompt_logprobs"]

        self.response_ids = tracker_tokenized["response_ids"]
        self.response_attention_mask = tracker_tokenized["response_attention_mask"]
        self.response_loss_mask = tracker_tokenized["response_loss_mask"]
        self.response_position_ids = tracker_tokenized["response_position_ids"]
        self.response_logprobs = tracker_tokenized["response_logprobs"]

        self.reference_advantage = tracker_tokenized["reference_advantage"]
        self.step_reward = tracker_tokenized["step_reward"]

        self.messages = messages

        self.truncate_output_ids()

        assert len(self.response_ids) != 0, "response_ids should not be empty"

    def truncate_output_ids(self) -> None:
        assert (
            len(self.input_ids)
            == len(self.attention_mask)
            == len(self.position_ids)
            == len(self.loss_mask)
        )
        assert (
            len(self.prompt_ids)
            == len(self.prompt_attention_mask)
            == len(self.prompt_position_ids)
            == len(self.prompt_loss_mask)
            == len(self.prompt_logprobs)
        )
        assert (
            len(self.response_ids)
            == len(self.response_attention_mask)
            == len(self.response_position_ids)
            == len(self.response_loss_mask)
            == len(self.response_logprobs)
        )
        assert (
            isinstance(self.input_ids, list)
            and isinstance(self.prompt_ids, list)
            and isinstance(self.response_ids, list)
        )

        truncate_any = False

        if len(self.prompt_ids) > self.max_prompt_len:
            truncate_any = True
            raise RuntimeError(
                f"Warning: prompt_ids length {len(self.prompt_ids)} exceeds max_prompt_len {self.max_prompt_len}, truncating."
            )

        if len(self.response_ids) > self.max_response_len:
            truncate_any = True
            logger.warning(
                "-------------------------------------------------------------------------------------------------------"
            )
            logger.warning(
                f"Warning: response_ids length {len(self.response_ids)} exceeds max_response_len {self.max_response_len}, truncating."
            )
            logger.warning(
                "-------------------------------------------------------------------------------------------------------"
            )
            self.response_ids = self.response_ids[: self.max_response_len]
            self.response_attention_mask = self.response_attention_mask[: self.max_response_len]
            self.response_position_ids = self.response_position_ids[: self.max_response_len]
            self.response_loss_mask = self.response_loss_mask[: self.max_response_len]
            self.response_logprobs = self.response_logprobs[: self.max_response_len]

        if truncate_any:
            self.input_ids = self.prompt_ids + self.response_ids
            self.attention_mask = self.prompt_attention_mask + self.response_attention_mask
            self.position_ids = self.prompt_position_ids + self.response_position_ids
            self.loss_mask = self.prompt_loss_mask + self.response_loss_mask
            self.logprobs = self.prompt_logprobs + self.response_logprobs

    def discard(self) -> None:
        """
        Discard the experience.
        """
        raise RuntimeError("Never use this method.")
