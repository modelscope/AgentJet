from typing import List, Union, Tuple
from transformers.tokenization_utils import PreTrainedTokenizer
from astune.utils.tokenizer import astune_apply_chat_templat
from astune.schema.trajectory import Reward
from loguru import logger
import uuid

INVALID_LOG_PROB_VALUE = 0.0


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
        token_logprob_arr=[],
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
        self.uuid = uuid.uuid4().hex
        self.build_from_uuid = build_from_uuid

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
            dummy_msg = [{"role": "assistant", "content": "dummy text"}]
            try:
                # completion_token_arr will contain generation_prompt header
                auto_tokenize_target = {
                    "role": self.role,
                    "content": self.content_for_future,
                }
                if self.tool_calls:
                    auto_tokenize_target.update({"tool_calls": self.tool_calls})
                text_frag_to = astune_apply_chat_templat(
                    tokenizer=tokenizer,
                    conversation=dummy_msg + [auto_tokenize_target],
                    tokenize=False,
                    tools=tools,
                )
            except Exception as e:
                raise ValueError(
                    f"Cannot tokenize {self.role} --- {self.content_for_future}, \n\n Error: {e}"
                )
            self.token_arr, _ = self.get_inc_simple(
                text_frag_from=astune_apply_chat_templat(
                    tokenizer=tokenizer,
                    conversation=dummy_msg,
                    tokenize=False,
                    tools=tools,
                ),
                text_frag_to=text_frag_to,
                tokenizer=tokenizer,
            )

    @property
    def content_for_future(self):
        if self._content_for_future == "":
            if not self.tool_calls:
                # from vsdb import bp; bp("H1")
                logger.exception("content_for_future is not set, or previous llm output is empty!")
                self._content_for_future
        return self._content_for_future

    @property
    def need_training(self):
        NEED_TRAIN_AUTHORS = ["llm"]
        NON_TRAIN_AUTHORS = [
            "env",
            "initialization",
            "user",
            "memory",
            "llm(do_not_train)",
        ]
        assert (
            (self.author in NEED_TRAIN_AUTHORS)
            or (self.author in NON_TRAIN_AUTHORS)
            or (self.author.endswith("(discard)"))
        ), f"author {self.author} is not identified"
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

        if self.need_training:
            msg_token_mask = [1] * len(self.token_arr)
            msg_token_mask = blackout_specific_token_ids_first_encounter(
                msg_token_mask, self.token_arr, blackout_token_combo
            )
            msg_token_mask = blackout_everything_after_eos_but_keep_eos(
                mask=msg_token_mask,
                token_arr=self.token_arr,
                eos_token_id=self.eos_token_id,
            )
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
        input_id_increment = input_ids[
            len(token_ids_acc) :
        ]  # get the new tokens added in this step
        overlap_length = 0
        for i in range(len(token_ids_acc)):
            if i < len(token_ids_acc) and input_ids[i] == token_ids_acc[i]:
                overlap_length += 1
            else:
                break
        msg = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
        # print(msg)
        return input_id_increment, msg
