import os
import traceback
import uuid
from datetime import datetime
from typing import List

from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer

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
        tokenizer: PreTrainedTokenizer = None,  # type: ignore
        token_generator="manual",
        build_from_uuid="",
        tools=[],
        tool_calls=[],
        tool_call_id="",
        token_logprob_arr=[],
        name="",    # preserved field, not used currently
        first_message=False,
        images=None,        # Optional[list] of image refs (PIL / bytes / data URL / path)
        processor=None,     # Optional HuggingFace ProcessorMixin (e.g. Qwen2_5_VLProcessor)
    ):
        self.author = author
        self.role = role
        self.content = content
        self.token_arr = token_arr
        self.token_logprob_arr = token_logprob_arr
        self.token_begin_index = token_begin_index
        self.token_end_index = token_end_index
        self.invalid_log_prob_value = INVALID_LOG_PROB_VALUE
        self._info = ""
        self.tools = tools
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name    # preserved field, not used currently
        if not isinstance(self.tool_calls, list):
            # agent scope sometimes gives weird type for tool_calls, which is against OpenAI schema
            self.tool_calls = list(self.tool_calls)
        self.uuid = uuid.uuid4().hex
        self.build_from_uuid = build_from_uuid
        self.first_message = first_message
        self.manual_loss_mask_override = []
        self.lack_normal_eos = False

        # Multimodal: images attached to this message and the resulting
        # processor-side tensors (pixel_values, image_grid_thw, ...).
        # Only meaningful on the first user message for single-turn VL tasks.
        self.images = images
        self.processor = processor
        self.multi_modal_inputs = None  # dict[str, torch.Tensor] once tokenized

        # text content to compare when timeline merging
        self._text_content_for_compare = ""
        self.generate_content_for_compare(content=self.content)

        self.eos_token_id = tokenizer.eos_token_id

    @property
    def text_content_for_compare(self):
        if self._text_content_for_compare == "":
            if not self.tool_calls:
                logger.exception("text_content_for_compare is not set, or previous llm output is empty!")
        return self._text_content_for_compare

    @property
    def need_training(self):
        assert (
            (self.author in NEED_TRAIN_AUTHORS)
            or (self.author in NON_TRAIN_AUTHORS)
            or (self.author.endswith("(discard)"))
        ), f"author {self.author} is not identified"
        return self.author in NEED_TRAIN_AUTHORS

    def generate_content_for_compare(self, content):
        self._text_content_for_compare = content

    def get_loss_mask(self, blackout_token_combo):
        if self.need_training:
            # keep eos, but blackout everything after eos
            msg_token_mask = [1] * len(self.token_arr)
            msg_token_mask = blackout_specific_token_ids_first_encounter(
                msg_token_mask, self.token_arr, blackout_token_combo
            )
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
                # manual_loss_mask_override: this comes by diff [prompt + answer] with [prompt] (see get_token_inc_from_llm_response)
                # msg_token_mask:            this comes from consuming the generation token prefix (bos + /n, something + think) and eos
                try:
                    assert len(self.manual_loss_mask_override) == len(msg_token_mask), "token mask length mismatch"
                    # assert all(a == b for a, b in zip(self.manual_loss_mask_override, msg_token_mask))
                except AssertionError:
                    error_msg = (
                        "Manual loss mask override mismatch | "
                        f"author={self.author} role={self.role} uuid={self.uuid} | "
                        f"override_len={len(self.manual_loss_mask_override)} mask_len={len(msg_token_mask)} | "
                        f"token_arr_len={len(self.token_arr)} content_preview={self.content[:100]!r} | "
                        f"override={self.manual_loss_mask_override} | "
                        f"generated_mask={msg_token_mask}"
                    )
                    logger.bind(exception=True).error(error_msg)
                    log_dir = "./loss_mask_exception"
                    os.makedirs(log_dir, exist_ok=True)
                    with open(os.path.join(log_dir, "exception.log"), "a") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"[{datetime.now().isoformat()}]\n")
                        f.write(f"{error_msg}\n")
                        f.write(f"Traceback:\n{traceback.format_exc()}\n")

            # returning either is ok, but when lack eos, msg_token_mask is more accurate, otherwise, manual_loss_mask_override is more accurate
            if self.lack_normal_eos:
                return msg_token_mask
            else:
                return self.manual_loss_mask_override
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
        del tokenizer_output  # Free memory immediately

        tokenizer_output = tokenizer(text_frag_to, return_tensors="pt", padding=False)
        input_ids = tokenizer_output["input_ids"][0].tolist()
        del tokenizer_output  # Free memory immediately
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

