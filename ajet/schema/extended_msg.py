import uuid
from typing import List

from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer

from ajet.utils.tokenizer import ajet_apply_chat_template

# import numpy as np
# INVALID_LOG_PROB_VALUE = np.inf # when debuging, set to np.inf, if anything goes wrong, we can sense that immediately
INVALID_LOG_PROB_VALUE = 0  # normally, set to 0 is ok
NEED_TRAIN_AUTHORS = ["llm"]
NON_TRAIN_AUTHORS = [
    "env",
    "initialization",
    "user",
    "memory",
    "llm(do_not_train)",
]
DUMMY_MSG = [{"role": "assistant", "content": "dummy text"}]


def find_sublist_indices(large_list, small_list, reverse=False):
    small_len = len(small_list)
    if reverse:
        for i in reversed(range(len(large_list) - small_len + 1)):
            if large_list[i : i + small_len] == small_list:
                return i
    for i in range(len(large_list) - small_len + 1):
        if large_list[i : i + small_len] == small_list:
            return i
    return -1


def blackout_specific_token_ids_first_encounter(mask, arr, token_ids):
    index = find_sublist_indices(arr, token_ids, reverse=False)
    if index >= 0:
        for i in range(index, index + len(token_ids)):
            mask[i] = 0
    return mask


def blackout_everything_after_eos_but_keep_eos(mask, token_arr, eos_token_id):
    eos_position = token_arr.index(eos_token_id) if eos_token_id in token_arr else -1
    if eos_position != -1:
        for i in range(eos_position + 1, len(mask)):
            mask[i] = 0
    return mask


def blackout_everything_after_eos_including_eos(mask, token_arr, eos_token_id):
    eos_position = token_arr.index(eos_token_id) if eos_token_id in token_arr else -1
    if eos_position != -1:
        for i in range(eos_position, len(mask)):
            mask[i] = 0
    return mask


class ExtendedMessage:
    def __init__(
        self,
        author,
        role="assistant",
        content="",
        token_arr=[],
        token_begin_index=-1,
        token_end_index=-1,
        clip=False,
        clip_token_limit=8192,
        tokenizer: PreTrainedTokenizer = None,  # type: ignore
        token_generator="manual",
        build_from_uuid="",
        tools=[],
        tool_calls=[],
        tool_call_id="",
        token_logprob_arr=[],
        name="",  # preserved field, not used currently
        first_message=False,
    ):
        self.author = author
        self.role = role
        self.content = content
        self.token_arr = token_arr
        self.token_logprob_arr = token_logprob_arr
        self.token_begin_index = token_begin_index
        self.token_end_index = token_end_index
        self.invalid_log_prob_value = INVALID_LOG_PROB_VALUE
        self._content_for_future = ""
        self._info = ""
        self.clip = clip
        self.tools = tools
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name  # preserved field, not used currently
        if not isinstance(self.tool_calls, list):
            # agent scope sometimes gives weird type for tool_calls, which is against OpenAI schema
            self.tool_calls = list(self.tool_calls)
        self.uuid = uuid.uuid4().hex
        self.build_from_uuid = build_from_uuid
        self.first_message = first_message
        self.manual_loss_mask_override = []
        self.lack_normal_eos = False

        if not clip:
            self.generate_content_for_future(tokenizer=None, clip=False)
        else:
            self.generate_content_for_future(
                tokenizer=tokenizer,
                clip=True,
                clip_token_limit=clip_token_limit,
            )
        self.eos_token_id = tokenizer.eos_token_id

        if token_generator == "auto":
            self.token_arr = self.auto_tokenize(
                tokenizer=tokenizer,
                tools=tools,
            )

    def auto_tokenize(self, tokenizer, tools):
        if (not self.first_message) and (self.role == "system"):
            raise ValueError("The system message is usually the first message, check program bugs.")
        elif (self.first_message) and (self.role != "system"):
            raise ValueError("The first message is supposed to be the system message, check program bugs, or remove this warning.")
        if not self.first_message:
            self.token_arr = self.auto_tokenize_non_first_message(tokenizer=tokenizer, tools=tools)
        else:
            auto_tokenize_target = {
                "role": self.role,
                "content": self.content_for_future,
            }
            if self.tool_calls:
                auto_tokenize_target.update({"tool_calls": self.tool_calls})
            self.token_arr = ajet_apply_chat_template(
                tokenizer=tokenizer,
                conversation=[auto_tokenize_target],
                tokenize=True,
                tools=tools,
            )
        return self.token_arr

    def auto_tokenize_non_first_message(self, tokenizer, tools):
        try:
            # completion_token_arr will contain generation_prompt header
            auto_tokenize_target = {
                "role": self.role,
                "content": self.content_for_future,
            }
            if self.tool_calls:
                auto_tokenize_target.update({"tool_calls": self.tool_calls})
            if self.tool_call_id:
                auto_tokenize_target.update({"tool_call_id": self.tool_call_id})
            text_frag_to = ajet_apply_chat_template(
                tokenizer=tokenizer,
                conversation=DUMMY_MSG + [auto_tokenize_target],
                tokenize=False,
                tools=tools,
            )
        except Exception as e:
            raise ValueError(f"Cannot tokenize {self.role} --- {self.content_for_future}, \n\n Error: {e}")
        self.token_arr, _ = self.get_inc_simple(
            text_frag_from=ajet_apply_chat_template(
                tokenizer=tokenizer,
                conversation=DUMMY_MSG,
                tokenize=False,
                tools=tools,
            ),
            text_frag_to=text_frag_to,
            tokenizer=tokenizer,
        )
        return self.token_arr

    @property
    def content_for_future(self):
        if self._content_for_future == "":
            if not self.tool_calls:
                logger.exception("content_for_future is not set, or previous llm output is empty!")
                self._content_for_future
        return self._content_for_future

    @property
    def need_training(self):
        assert (self.author in NEED_TRAIN_AUTHORS) or (self.author in NON_TRAIN_AUTHORS) or (self.author.endswith("(discard)")), f"author {self.author} is not identified"
        return self.author in NEED_TRAIN_AUTHORS

    def generate_content_for_future(self, tokenizer, clip, clip_token_limit=-1):
        _content: str = self.content
        if clip:
            assert clip_token_limit > 0, "clip_token_limit must be set when clip is True"
            n_token = len(tokenizer(_content, return_tensors="pt", padding=False)["input_ids"][0])
            if n_token > clip_token_limit:
                # 8000 > 4000
                n_char = len(_content)  # 10,000
                eps = 100  # token
                preserve_percent = (clip_token_limit - eps) / n_token  # 3900 / 8000
                n_char_to_preserve = int(n_char * preserve_percent)
                _content = _content[:n_char_to_preserve] + "... truncate ..."
        self._content_for_future = _content

    def get_loss_mask(self, blackout_token_combo):
        if self.need_training:
            # keep eos, but blackout everything after eos
            msg_token_mask = [1] * len(self.token_arr)
            msg_token_mask = blackout_specific_token_ids_first_encounter(msg_token_mask, self.token_arr, blackout_token_combo)
            # in normal case, we will blackout everything after the EOS token
            # but EOS still participates in the loss calculation
            msg_token_mask = blackout_everything_after_eos_but_keep_eos(
                mask=msg_token_mask,
                token_arr=self.token_arr,
                eos_token_id=self.eos_token_id,
            )
            # however, if the message does not have eos (e.g., finish_reason: length), we will blackout everything after the EOS token
            # including the EOS token
            if self.lack_normal_eos:
                msg_token_mask = blackout_everything_after_eos_including_eos(
                    mask=msg_token_mask,
                    token_arr=self.token_arr,
                    eos_token_id=self.eos_token_id,
                )
            if self.manual_loss_mask_override:
                # assert two list is identical
                assert len(self.manual_loss_mask_override) == len(msg_token_mask)
                assert all(a == b for a, b in zip(self.manual_loss_mask_override, msg_token_mask))

            return msg_token_mask
        else:
            msg_token_mask = [0] * len(self.token_arr)
            return msg_token_mask

    def get_inc_simple(self, text_frag_from, text_frag_to, tokenizer):
        """
        Get the incremental token array from text_frag_from to text_frag_to.
        """
        tokenizer_output = tokenizer(text_frag_from, return_tensors="pt", padding=False)
        tokenizer_input_ids = tokenizer_output["input_ids"][0].tolist()
        token_ids_acc = tokenizer_input_ids

        tokenizer_output = tokenizer(text_frag_to, return_tensors="pt", padding=False)
        input_ids = tokenizer_output["input_ids"][0].tolist()
        # get the new tokens added in this step
        input_id_increment = input_ids[len(token_ids_acc) :]
        FN_DEBUG = False
        if FN_DEBUG:
            overlap_length = 0
            for i in range(len(token_ids_acc)):
                if i < len(token_ids_acc) and input_ids[i] == token_ids_acc[i]:
                    overlap_length += 1
                else:
                    break
            msg = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
        else:
            msg = ""
        return input_id_increment, msg

    @staticmethod
    def check_and_merge_chained_tool_response(ext_msg_array: List["ExtendedMessage"], tokenizer: PreTrainedTokenizer) -> List["ExtendedMessage"]:
        """
        Inside a list of ExtendedMessage,
        Find consecutive ext msg with role=="tool", then merge them into one ExtendedMessage

        Jinja2 template logic for reference:

        {%- elif message.role == \"tool\" %}
            {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}
                {{- '<|im_start|>user' }}
            {%- endif %}
            {{- '\
                <tool_response>\
            ' }}
            {{- message.content }}
            {{- '\
                </tool_response>' }}
                        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}
                            {{- '<|im_end|>\
            ' }}
            {%- endif %}
        {%- endif %}
        """

        def merge_tool_group(group, tokenizer):
            if len(group) == 1:
                return group[0]

            msg0 = group[0]
            merged_content = "".join(f"<tool_response>\n{msg.content}\n</tool_response>\n" for msg in group)
            merged_content = merged_content[len("<tool_response>\n") :]
            merged_content = merged_content[: -len("</tool_response>\n")]
            merged = ExtendedMessage(
                author=msg0.author,
                role=msg0.role,
                content=merged_content,
                tokenizer=tokenizer,
                token_generator="manual",
                build_from_uuid=msg0.uuid,
                tools=msg0.tools,
                tool_calls=msg0.tool_calls,
                token_logprob_arr=msg0.token_logprob_arr,
                first_message=msg0.first_message,
            )
            # re-compute token_arr
            auto_tokenize_targets = [{"role": msg.role, "content": msg.content_for_future} for msg in group]
            merged.token_arr, _ = merged.get_inc_simple(
                text_frag_from=ajet_apply_chat_template(
                    tokenizer=tokenizer,
                    conversation=DUMMY_MSG,
                    tokenize=False,
                    tools=merged.tools,
                    add_generation_prompt=False,
                ),
                text_frag_to=ajet_apply_chat_template(
                    tokenizer,
                    conversation=DUMMY_MSG + auto_tokenize_targets,
                    tokenize=False,
                    tools=merged.tools,
                    add_generation_prompt=False,
                ),
                tokenizer=tokenizer,
            )
            return merged

        groups = []
        current_tool_group = []
        for msg in ext_msg_array:
            if msg.role == "tool":
                current_tool_group.append(msg)
            else:
                if current_tool_group:
                    groups.append(current_tool_group)
                    current_tool_group = []
                groups.append([msg])
        if current_tool_group:
            groups.append(current_tool_group)

        result_ext_msg_array = [merge_tool_group(group, tokenizer) for group in groups]
        return result_ext_msg_array
