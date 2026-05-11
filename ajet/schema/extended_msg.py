import os
import traceback
import uuid
from datetime import datetime
from typing import List

from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer

from ajet.utils.tokenizer import ajet_apply_chat_template
from ajet.utils.multimodal import load_image_to_pil

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
DUMMY_MSG = [{"role": "user", "content": "dummy text"}]


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

        if token_generator == "auto":
            self.token_arr = self.auto_tokenize(
                tokenizer=tokenizer,
                tools=tools,
            )

    def auto_tokenize(self, tokenizer, tools):
        if (not self.first_message) and (self.role == "system"):
            raise ValueError("The system message is usually the first message, check program bugs.")
        elif (self.first_message) and (self.role != "system"):
            raise ValueError(
                "The first message is supposed to be the system message, check program bugs, or remove this warning."
            )

        # Multimodal path: any message carrying images goes through the HF
        # processor (e.g. Qwen2_5_VLProcessor) so image placeholder tokens are
        # expanded and we can capture pixel_values / image_grid_thw for
        # downstream training. We compute the delta against a dummy conversation
        # so tokens for just this message are extracted.
        if self.images and self.processor is not None:
            self.token_arr = self._auto_tokenize_with_processor(
                tokenizer=tokenizer, tools=tools,
            )
            return self.token_arr

        if not self.first_message:
            self.token_arr = self.auto_tokenize_non_first_message(tokenizer=tokenizer, tools=tools)
        else:
            auto_tokenize_target:dict = {
                "role": self.role,
                "content": self.text_content_for_compare,
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

    def _auto_tokenize_with_processor(self, tokenizer, tools):
        """Tokenize this message via the HF processor, producing token ids
        with image placeholder tokens expanded, plus multi_modal_inputs
        (pixel_values, image_grid_thw, ...).

        For non-first messages we tokenize ``DUMMY_MSG + self`` (with images)
        and subtract ``DUMMY_MSG`` (text-only) to get just this message's delta,
        matching the existing non-first tokenization contract.
        """
        pil_images = [load_image_to_pil(im) for im in self.images]
        vision_content = [{"type": "image", "image": p} for p in pil_images]
        vision_content.append({"type": "text", "text": self.text_content_for_compare})
        self_msg = {"role": self.role, "content": vision_content}
        if self.tool_calls:
            self_msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            self_msg["tool_call_id"] = self.tool_call_id

        if self.first_message:
            proc_msgs = [self_msg]
        else:
            # The HF processor needs content to be a list of typed blocks for
            # all messages; DUMMY_MSG's plain-string content would break.
            dummy_msg_processor_style = [
                {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                for m in DUMMY_MSG
            ]
            proc_msgs = dummy_msg_processor_style + [self_msg]

        raw_prompt = self.processor.apply_chat_template(
            proc_msgs, tools=tools or None,
            add_generation_prompt=False, tokenize=False,
        )
        model_inputs = self.processor(
            text=[raw_prompt], images=pil_images, return_tensors="pt",
        )
        model_inputs = dict(model_inputs)
        input_ids = model_inputs.pop("input_ids")[0].tolist()
        model_inputs.pop("attention_mask", None)
        self.multi_modal_inputs = {k: v for k, v in model_inputs.items()}

        if self.first_message:
            return input_ids

        # Subtract DUMMY_MSG prefix to get the delta tokens for just self.
        # DUMMY_MSG has no images, but the processor still needs typed content.
        dummy_msg_processor_style = [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in DUMMY_MSG
        ]
        dummy_ids = self.processor.apply_chat_template(
            dummy_msg_processor_style, tools=tools or None,
            add_generation_prompt=False, tokenize=True,
        )
        if hasattr(dummy_ids, "tolist"):
            dummy_ids = dummy_ids.tolist()
        # apply_chat_template(tokenize=True) returns a batched list[list[int]]
        # when the input is a single convo — unwrap the outer batch dim.
        if dummy_ids and isinstance(dummy_ids[0], list):
            dummy_ids = dummy_ids[0]
        # Sanity: the processor-expanded full_ids should start with dummy_ids.
        if input_ids[: len(dummy_ids)] == dummy_ids:
            return input_ids[len(dummy_ids):]
        # Fallback: find DUMMY prefix by longest-match.
        n = min(len(dummy_ids), len(input_ids))
        match = 0
        for i in range(n):
            if input_ids[i] == dummy_ids[i]:
                match = i + 1
            else:
                break
        return input_ids[match:]

    def auto_tokenize_non_first_message(self, tokenizer, tools):
        try:
            # completion_token_arr will contain generation_prompt header
            auto_tokenize_target:dict = {
                "role": self.role,
                "content": self.text_content_for_compare,
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
            raise ValueError(
                f"Cannot tokenize {self.role} --- {self.text_content_for_compare}, \n\n Error: {e}"
            )
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
                try:
                    assert len(self.manual_loss_mask_override) == len(msg_token_mask)
                    assert all(a == b for a, b in zip(self.manual_loss_mask_override, msg_token_mask))
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

    @staticmethod
    def check_and_merge_chained_tool_response(
        ext_msg_array: List["ExtendedMessage"], tokenizer: PreTrainedTokenizer
    ) -> List["ExtendedMessage"]:
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
            merged_content = "".join(
                f"<tool_response>\n{msg.content}\n</tool_response>\n" for msg in group
            )
            merged_content = merged_content[len("<tool_response>\n") :]
            merged_content = merged_content[: -len("</tool_response>\n")]
            # create merged tool response block
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
            auto_tokenize_targets = [
                {"role": msg.role, "content": msg.text_content_for_compare} for msg in group
            ]
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
