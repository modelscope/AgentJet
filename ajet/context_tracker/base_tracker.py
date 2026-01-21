from typing import Any, Dict, List, Optional, Tuple, Union
from ajet.schema.task import WorkflowTask

from ajet.schema.extended_msg import (
    INVALID_LOG_PROB_VALUE,
    ExtendedMessage,
    find_sublist_indices,
)
from ajet.schema.trajectory import Reward


def replace_token_ids(
    token_container,
    precise_token,
    precise_logprob,
    begin_ids,
    end_ids,
) -> Tuple[List[int], List[int], List[int], bool]:
    """
    Replace token ids

    input   || token_container: [begin_ids, ... ids_that_may_not_precise ... , end_ids, other_ids]
    ==>
    output1 || final_token_ids: [begin_ids, ...       precise_token      ... , end_ids, other_ids]
    output2 || final_logprob:   [NA,        ...       precise_logprob    ... , NA     , NA       ]
    output3 || loss_mask:       [0,         ...       1                  ... , 1      , 0        ]
    output4 || lack_normal_eos: False



    test case:
    ----------- case 1 (with_normal_eos) -----------
    (NA = INVALID_LOG_PROB_VALUE)

    begin_ids = [151644, 77091, 198]
    end_ids   = [151645]
    token_container = [151644, 77091, 198,     1, 1, 1,           151645, 1, 2, 3, 4]
    precise_token =                            [4, 3, 2,          151645]
    precise_logprob =                          [-0.4, -0.3, -0.2, -0.1]

    assert replace_token_ids(
        token_container,
        precise_token,
        precise_logprob,
        begin_ids,
        end_ids,
    ) = (
        [151644, 77091, 198,     4, 3, 2,        151645, 1, 2, 3, 4]
        [NA, NA, NA,             -0.4, -0.3, -0.2, -0.1, NA, NA, NA, NA]
        [0, 0, 0,                1, 1, 1,           1,  0 ,0 ,0 ,0]
    )

    ----------- case 2 (lack_normal_eos) -----------
    begin_ids = [151644, 77091, 198]
    end_ids   = [151645]
    token_container = [151644, 77091, 198,   1, 1, 1,     151645, 1, 2, 3, 4]
    precise_token =                         [3, 2, 1,]
    precise_logprob =                       [-0.3, -0.2, -0.1]

    assert replace_token_ids(
        token_container,
        precise_token,
        precise_logprob,
        begin_ids,
        end_ids,
    ) = (
        [151644, 77091, 198,     3, 2, 1,           151645, 1, 2, 3, 4]
        [NA, NA, NA,             -0.3, -0.2, -0.1,  NA, NA, NA, NA]
        [0, 0, 0,                1, 1, 1,           0,  0 ,0 ,0 ,0]
    )

    """

    _begin_index = find_sublist_indices(token_container, begin_ids) + len(begin_ids)
    _end_index = find_sublist_indices(token_container, end_ids, reverse=True)

    if precise_token[-len(end_ids) :] == end_ids:  # remove end_ids token
        lack_normal_eos = False
        precise_token_center = precise_token[: -len(end_ids)]
        precise_logprob_center = precise_logprob[: -len(end_ids)]
        logprob_eos_tail = precise_logprob[-len(end_ids) :]
    else:
        lack_normal_eos = True
        precise_token_center = precise_token
        precise_logprob_center = precise_logprob
        logprob_eos_tail = []

    if precise_token[: len(begin_ids)] == begin_ids:  # remove begin_ids token
        # precise_token = precise_token[len(begin_ids) :]
        # precise_logprob_center = precise_logprob[len(begin_ids) :]
        raise ValueError(
            "Unexpected situation, wrong llm output (unexpected BOS): please post an github issue."
        )

    final_token_ids = (
        token_container[:_begin_index] + precise_token_center + token_container[_end_index:]
    )
    final_logprob = (
        [INVALID_LOG_PROB_VALUE] * _begin_index
        + precise_logprob_center
        + logprob_eos_tail
        + [INVALID_LOG_PROB_VALUE] * (len(token_container) - _end_index - len(logprob_eos_tail))
    )
    loss_mask = (
        [0] * _begin_index
        + [1] * len(precise_logprob_center)
        + [1] * len(logprob_eos_tail)
        + [0] * (len(token_container) - _end_index - len(logprob_eos_tail))
    )
    return final_token_ids, final_logprob, loss_mask, lack_normal_eos


class BaseTracker(object):
    def __init__(self, config, tokenizer, workflow_task: WorkflowTask, **kwargs):

        self.workflow_task = workflow_task
        self.task_batch_index = self.workflow_task.task_batch_index
        self.task_tag = self.workflow_task.task_tag
        self.task_id = self.workflow_task.task_id
        self.episode_uuid = self.workflow_task.episode_uuid

        self.config = config
        self.tokenizer = tokenizer
        self.saved_timelines: List[List[ExtendedMessage]] = []
        self.current_context_status = ""
        max_response_length = self.config.ajet.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.ajet.rollout.max_model_len
        self.max_seq_length: int = max_model_len - max_response_length
        self.blackout_token_combo = tokenizer.encode("<|im_start|>assistant\n")
        self._im_start_token_id = tokenizer.encode("<|im_start|>")[0]
        self.generated_token_cnt = 0
        self.terminal_rewards_dict = {}
        self.discarded = False
        self.is_terminated = False
        self.reward_structure: Union[Reward, None] = None
        self.context_time_cost = 0
        self.tag = ""
        self.current_batch_success_rate: float = float("-inf")
        self.current_batch_reward: float = float("-inf")
        self.already_mad_flag: bool = False
        self.round_cnt = 0
        self.generation_prompt_token = None
        self.log_metrics: Optional[Dict[str, Union[float, List[float], Dict[str, Any]]]] = None  # Initialize workflow_metadata to store tool statistics

        assert (
            self.config.ajet.data.max_prompt_length
            + self.config.ajet.data.max_response_length
            <= max_model_len
        )

    def group_tokenize(self):
        raise NotImplementedError

    def group_tokenize_multi_group(self):
        raise NotImplementedError

    def group_tokenize_single_group(self, timeline):
        raise NotImplementedError

    def tokenize_steps(
        self, ext_steps: List[ExtendedMessage], index: int, total_steps: int
    ) -> dict:
        raise NotImplementedError
