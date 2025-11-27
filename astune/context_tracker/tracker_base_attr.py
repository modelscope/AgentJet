from typing import List, Tuple, Union

from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer

from astune.schema.extended_msg import (
    INVALID_LOG_PROB_VALUE,
    ExtendedMessage,
    find_sublist_indices,
)
from astune.schema.trajectory import Reward


def replace_token_ids(
    place_holder, replace_with, begin, end, raw_logprob
) -> Tuple[List[int], List[int]]:
    _begin_index = find_sublist_indices(place_holder, begin) + len(begin)
    _end_index = find_sublist_indices(place_holder, end, reverse=True)

    if replace_with[-len(end) :] == end:  # remove end token
        replace_with = replace_with[: -len(end)]
        raw_logprob = raw_logprob[: -len(end)]
    if replace_with[: len(begin)] == begin:  # remove begin token
        replace_with = replace_with[len(begin) :]
        raw_logprob = raw_logprob[len(begin) :]

    final = place_holder[:_begin_index] + replace_with + place_holder[_end_index:]
    final_logprob = (
        [INVALID_LOG_PROB_VALUE] * _begin_index
        + raw_logprob
        + [INVALID_LOG_PROB_VALUE] * (len(place_holder) - _end_index)
    )
    return final, final_logprob


class TrackerAttr(object):
    def __init__(self, config, tokenizer, **kwargs):
        self.task_batch_index = kwargs.get("task_batch_index", "undefined")
        self.task_tag = kwargs.get("task_tag", "undefined")
        self.task_id = kwargs.get("task_id", "undefined")
        self.config = config
        self.tokenizer = tokenizer
        self.full_context: List[ExtendedMessage] = []
        self.grouped_steps: List[List[ExtendedMessage]] = []
        self.current_context_status = ""
        max_response_length = self.config.astune.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.astune.rollout.max_model_len
        self.max_seq_length: int = max_model_len - max_response_length
        self.max_env_output_length: int = self.config.astune.rollout.max_env_len
        self.blackout_token_combo = tokenizer.encode("<|im_start|>assistant\n")
        self.generated_token_cnt = 0
        self.terminal_rewards_dict = {}
        self.discarded = False
        self.is_terminated = False
        self.reward_structure: Union[Reward, None] = None
        self.context_time_cost = 0
        self.tag = ""
        self.current_batch_success_rate: float = -1.0
        self.already_mad_flag: bool = False
        self.round_cnt = 0
        self.generation_prompt_token = None

        assert (
            self.config.astune.data.max_prompt_length + self.config.astune.data.max_response_length
            <= max_model_len
        )
