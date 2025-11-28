from typing import Any, Dict, List, Union

import numpy as np
from pydantic import BaseModel, Field


class Reward(BaseModel):
    raw_reward: float = Field(default=0.0)
    raw_step_reward: Union[List[float], None] = Field(default=[])
    step_reward: List[float] = Field(default=[])
    step_advantage: List[float] = Field(default=[])
    step_advantage_simple: List[float] = Field(default=[])
    success_rate: float = Field(default=0.0)
    madness: float = Field(default=0.0)
    description: str = Field(default="Outcome 1 denotes success, and 0 denotes failure.")
    metadata: dict = Field(default_factory=dict)

    @property
    def performance_reward(self):
        if (self.step_reward is not None) and len(self.step_reward) > 0:
            res = np.mean(self.step_reward)
            # print(f"Performance reward computed as mean of step_reward: {res}")
            return res
        else:
            return self.raw_reward


class Trajectory(BaseModel):
    task_batch_index: int = Field(default=0)
    task_tag: str = Field(default="")

    steps: List[dict] = Field(default_factory=list)
    query: str = Field(default="")

    is_terminated: bool = Field(default=False)
    reward: Reward = Field(default_factory=Reward)

    metadata: dict = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.reward_outcome > 0


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

    def __init__(self, cmt_tokenized: dict, messages, config, **kwargs):
        super().__init__(**kwargs)

        self.max_prompt_len = config.astuner.data.max_prompt_length
        self.max_response_len = config.astuner.data.max_response_length
        self.max_model_len = (
            config.astuner.data.max_response_length + config.astuner.data.max_prompt_length
        )

        self.input_ids = cmt_tokenized["input_ids"]
        self.attention_mask = cmt_tokenized["attention_mask"]
        self.loss_mask = cmt_tokenized["loss_mask"]
        self.position_ids = cmt_tokenized["position_ids"]
        self.logprobs = cmt_tokenized["logprobs"]

        self.prompt_ids = cmt_tokenized["prompt_ids"]
        self.prompt_attention_mask = cmt_tokenized["prompt_attention_mask"]
        self.prompt_loss_mask = cmt_tokenized["prompt_loss_mask"]
        self.prompt_position_ids = cmt_tokenized["prompt_position_ids"]
        self.prompt_logprobs = cmt_tokenized["prompt_logprobs"]

        self.response_ids = cmt_tokenized["response_ids"]
        self.response_attention_mask = cmt_tokenized["response_attention_mask"]
        self.response_loss_mask = cmt_tokenized["response_loss_mask"]
        self.response_position_ids = cmt_tokenized["response_position_ids"]
        self.response_logprobs = cmt_tokenized["response_logprobs"]

        self.reference_advantage = cmt_tokenized["reference_advantage"]
        self.step_reward = cmt_tokenized["step_reward"]

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
            print(
                "-------------------------------------------------------------------------------------------------------"
            )
            print(
                f"Warning: prompt_ids length {len(self.prompt_ids)} exceeds max_prompt_len {self.max_prompt_len}, truncating."
            )
            print(
                "-------------------------------------------------------------------------------------------------------"
            )
            raise RuntimeError(
                "Prompt length exceeds maximum allowed length. Please adjust the input data."
            )
            self.prompt_ids = self.prompt_ids[-self.max_prompt_len :]
            self.prompt_attention_mask = self.prompt_attention_mask[-self.max_prompt_len :]
            self.prompt_position_ids = self.prompt_position_ids[-self.max_prompt_len :]
            self.prompt_loss_mask = self.prompt_loss_mask[-self.max_prompt_len :]
            self.prompt_logprobs = self.prompt_logprobs[-self.max_prompt_len :]

        if len(self.response_ids) > self.max_response_len:
            truncate_any = True
            print(
                "-------------------------------------------------------------------------------------------------------"
            )
            print(
                f"Warning: response_ids length {len(self.response_ids)} exceeds max_response_len {self.max_response_len}, truncating."
            )
            print(
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
