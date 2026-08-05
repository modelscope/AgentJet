# flake8: noqa: F541, F841
import copy
import json
from dataclasses import dataclass, field
from typing import List, Tuple, cast

from beast_logger import NestedJsonItem, SeqItem, print_dict, print_nested
from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer

from ajet.context_tracker.timeline_merging.timeline_merging import (
    merge_tracker_timelines, is_timeline_mergeable
)
from ajet.context_tracker.single_agent_tracking import (
    SingleAgentContextTracker,
    ExtendedMessage,
)
from ajet.schema.extended_msg import INVALID_LOG_PROB_VALUE
from ajet.schema.trajectory import Reward
from ajet.utils.color_hsl import adjust_color_hsl_batch
from ajet.utils.compute_madness import compute_string_madness
from ajet.utils.tokenizer import ajet_apply_chat_template, derive_tool_sep_and_fold, flush_pending_tool_run

@dataclass
class TimelineMergingPolicyConfig:
    timeline_compare_level: str = "text"
    ignore_tools: bool = True


@dataclass
class ContextTrackerConfig:
    timeline_merging_policy: TimelineMergingPolicyConfig = field(
        default_factory=TimelineMergingPolicyConfig
    )
    fix_retokenization_drift: bool = True
    detect_timeline_snap: bool = False




class MultiAgentContextTracker(SingleAgentContextTracker):
    """
    Context tracker is responsible to monitor and process LLM IO.
    Each context tracker is responsible for ONE episode run only.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config,
        should_interrupt_soft_fn,
        should_interrupt_hard_fn,
        generated_token_callback_fn,
        processor=None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, **kwargs)
        self.tokenizer = tokenizer
        self.processor = processor  # HuggingFace ProcessorMixin for VL models, or None
        self.should_interrupt_soft_fn = should_interrupt_soft_fn
        self.should_interrupt_hard_fn = should_interrupt_hard_fn
        self.generated_token_callback_fn = generated_token_callback_fn
        self.context_overflow = False
        self.output_kwargs = {}
        self.input_kwargs = {}
        self.timeline_cache = {}

        # Whether the chat template folds a run of consecutive `tool` messages
        # into ONE <|im_start|>user segment (Qwen3 text -> True) or renders each
        # as its own segment (Qwen2.5-VL -> False), and the inter-tool separator
        # for merging. Derived once at init (cached per-tokenizer in utils).
        self.tool_res_sep, self.tool_res_fold = derive_tool_sep_and_fold(tokenizer)


    def preprocess_tools_field(self, tools: List[dict] = [], disable_toolcalls: bool = False):
        if disable_toolcalls:
            tools = []
        else:
            if tools is not None:
                # rerank tool parameters to improve compatibility
                for i in range(len(tools)):
                    tools[i]["function"]["parameters"] = tools[i]["function"].pop("parameters")
        return tools

    def extract_text_and_image_content_from_content_dict(self, msg):
        # OpenAI vision content blocks, e.g.:
        # msg = {
        #    "role": "user",
        #    "content": [
        #        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        #        {"type": "text",      "text": "What is in this figure?"},
        #    ],
        # }
        # Returns (str_content, should_skip_message, images).
        # Images are surfaced so callers can attach them to the ExtendedMessage.
        str_content = ""
        images: list = []
        for item in msg["content"]:
            item_type = item.get("type", "")
            assert not item_type == "tool_use", f"never observed such protocal yet"
            assert not item_type == "tool_result", f"never observed such protocal yet"

            assert isinstance(item, dict), f"Unsupported non-dict item in message content: {item}. Full message: {msg}"

            if item_type in ("image_url", "image"):
                # OpenAI: {"type": "image_url", "image_url": {"url": "..."}} or HF style {"type":"image","image":<pil>}
                url = None
                if item_type == "image_url":
                    iu = item.get("image_url")
                    if isinstance(iu, dict):
                        url = iu.get("url")
                    elif isinstance(iu, str):
                        url = iu
                else:
                    url = item.get("image") or item.get("url")
                if url is not None:
                    images.append(url)
                continue

            if ("text" not in item):
                logger.warning(
                    f"Non-text, non-image content in message content detected: {item}. Ignoring."
                )
                should_skip_message = True
                return str_content, should_skip_message, images

            if isinstance(item["text"], str):
                str_content += str(item["text"])
            else:
                str_content = ""

        should_skip_message = False
        return str_content, should_skip_message, images


    def step_spawn_timeline(self, messages: List[dict], tools: List = [], disable_toolcalls: bool = False) -> List[ExtendedMessage]:
        """Spawn a timeline from messages.

        Consecutive ``role == "tool"`` messages are handled so the timeline
        stays 1:1 aligned with the chat template's ``<|im_start|>`` segments:

        - If the template FOLDS a run of tool turns into ONE ``<|im_start|>user``
          segment (Qwen3 text), they are MERGED into one ExtendedMessage whose
          content is the per-tool contents joined by the template's inter-tool
          separator (``self.tool_res_sep``), so the merged message renders
          identically to the folded segment.
        - If the template renders each tool as its OWN segment (Qwen2.5-VL:
          ``<|im_start|>tool`` each), they are kept separate, else the timeline
          would be shorter than the segment count.

        Whether to fold is detected once at init via
        ``derive_tool_sep_and_fold`` (cached) and stored on
        ``self.tool_res_fold`` / ``self.tool_res_sep``. The merging itself is
        done by ``flush_pending_tool_run`` (in ajet.utils.tokenizer). This keeps
        ``len(timeline) == segment_count`` so ``patch_prompt_tokens`` (which
        assumes 1:1 message-to-segment) never goes out of range.

        Args:
            messages: List of message dictionaries
            tools: List of tool dictionaries
            disable_toolcalls: Whether to disable tool calls

        Returns:
            List of ExtendedMessage objects representing the timeline
        """
        timeline = []

        consider_roles = ["user", "assistant", "system", "tool"]
        if disable_toolcalls:
            consider_roles.remove("tool")

        previous_message_encounter_user_role = False

        # Buffered run of consecutive tool messages pending merge.
        pending_tool_run: List[dict] = []


        for i, msg in enumerate(messages):

            if (disable_toolcalls) and (not isinstance(msg["content"], str)):
                # Allow vision content blocks through (image_url / image /
                # text); skip only tool_use / tool_result / other exotic blocks.
                content = msg.get("content") or []
                if isinstance(content, list) and all(
                    isinstance(it, dict) and it.get("type") in ("image_url", "image", "text")
                    for it in content
                ):
                    pass  # vision blocks - keep going
                else:
                    continue

            if msg["role"] not in consider_roles:
                continue

            msg_images: list = []
            if not isinstance(msg["content"], str):
                should_skip_message = False

                # fix msg content
                if msg["content"] is None:
                    msg["content"] = ""

                elif isinstance(msg["content"], list):
                    msg["content"], should_skip_message, msg_images = self.extract_text_and_image_content_from_content_dict(msg)

                else:
                    raise ValueError(f"Unsupported non-str message content type: {type(msg['content'])}, Message:\n {msg}")

                if should_skip_message:
                    continue

                if not isinstance(msg["content"], str):
                    msg["content"] = str(msg["content"])  # TODO: better handling mm data

            msg_content = cast(str, msg["content"])

            # Buffer consecutive tool messages; flush on any non-tool message.
            if msg["role"] == "tool":
                pending_tool_run.append({
                    "content": msg_content,
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "first_index": i,
                })
                continue
            flush_pending_tool_run(
                timeline, pending_tool_run, self.tokenizer,
                getattr(self, "processor", None), tools,
                self.tool_res_sep, self.tool_res_fold,
            )

            if msg["role"] == "system":
                author = "initialization"
            else:
                author = "env"

            if msg["role"] == "user":
                previous_message_encounter_user_role = True

            # extract content block from openai-competible messages and convert to ExtendedMessage
            # token_arr is left empty here (token_generator="manual"); the whole
            # timeline is tokenized once and sliced per-message in step_prepare
            # via tokenize_and_slice_timeline, so each message's token_arr is an
            # exact contiguous slice of the full-conversation render (drift-free
            # by construction, matching vLLM's prompt_token_ids).
            timeline += [
                ExtendedMessage(
                    author=author,
                    role=msg["role"],
                    content=msg_content,
                    tokenizer=self.tokenizer,
                    tools=tools,
                    tool_calls=(msg["tool_calls"] if "tool_calls" in msg else ""),
                    tool_call_id=(msg["tool_call_id"] if "tool_call_id" in msg else ""),
                    token_generator="manual",
                    name = (msg["name"] if "name" in msg else ""),
                    first_message=(i == 0),
                    images=msg_images or None,
                    processor=getattr(self, "processor", None),
                )
            ]
            if msg_content.startswith("<think>") and (not previous_message_encounter_user_role):
                logger.warning(f"Warning! Message content contains a think tag, but no prior message has `user` role! This is not a common scenario. Please check your agent loop carefully.")

        flush_pending_tool_run(
            timeline, pending_tool_run, self.tokenizer,
            getattr(self, "processor", None), tools,
            self.tool_res_sep, self.tool_res_fold,
        )
        return timeline

    def tokenize_and_slice_timeline(self, timeline: List[ExtendedMessage], tools: List = [], sep_via_eos: bool = True) -> None:
        """Tokenize the whole timeline once, then slice per message.

        Renders the entire conversation (the OpenAI-style message list produced
        by ``to_role_content``) with the chat template in one shot, tokenizes
        the rendered text, and splits the resulting token-id list into one
        contiguous chunk per message. Each message's ``token_arr`` is then an
        exact slice of the single whole-conversation render — so
        ``concat(token_arr)`` reconstructs the same token stream vLLM produces,
        making retokenization drift impossible by construction
        (``patch_prompt_tokens`` becomes a no-op).

        The template's own ``loop.index0 vs last_query_index`` logic decides
        think-block stripping / tool-block placement on the full render, so we
        no longer need per-message anchors or suffix math.

        ``sep_via_eos`` (default True): split on ``<|im_end|>``
        (via ``_recover_segments_via_im_end``) — a real segment spans
        ``<|im_start|>`` ... (next real ``<|im_start|>``), with the trailing
        ``\\n`` after an ``<|im_end|>`` staying in its segment. This is
        per-segment IDENTICAL to the legacy ``<|im_start|>`` split on clean
        content (pinned by tests), but it also absorbs stray
        ``<|im_start|>`` ids an LLM may generate inside a message's content
        during rollout (degenerate/looping samples) — the legacy im_start split
        would treat those as boundaries and inflate the segment count past the
        timeline length, crashing. Pass False to force the legacy im_start
        split with divergence-detection fallback (im_end recovery only when
        ``len(split_ids) != len(timeline)``).
        """
        if not timeline:
            return

        conversation = self.to_role_content(timeline)

        # VL path: if any message carries images and we have a processor,
        # render+tokenize the whole conversation through the HF processor so
        # image placeholder tokens are expanded and pixel_values / image_grid_thw
        # are captured for the whole conversation in one combined dict.
        has_images = any(getattr(m, "images", None) for m in timeline)
        if has_images and getattr(self, "processor", None) is not None:
            from ajet.utils.multimodal import load_image_to_pil

            pil_images = []
            for m in timeline:
                if m.images:
                    pil_images.extend(load_image_to_pil(im) for im in m.images)
            raw_prompt = self.processor.apply_chat_template(
                conversation, tools=tools or None,
                add_generation_prompt=False, tokenize=False,
            )
            model_inputs = dict(self.processor(
                text=[raw_prompt], images=pil_images, return_tensors="pt",
            ))
            full_ids = model_inputs.pop("input_ids")[0].tolist()
            model_inputs.pop("attention_mask", None)
            # One combined multi_modal_inputs for the whole conversation; attach
            # it to the first message so merge_multi_modal_inputs picks it up.
            # Drop mm_token_type_ids (not dim-0 concatenable; see
            # merge_multi_modal_inputs).
            mmi = {k: v for k, v in model_inputs.items() if k != "mm_token_type_ids"}
            timeline[0].multi_modal_inputs = mmi or None
        else:
            # Text path: render the whole conversation, then tokenize the text.
            prompt_text = ajet_apply_chat_template(
                tokenizer=self.tokenizer,
                conversation=conversation,
                tools=tools,
                add_generation_prompt=False,
                tokenize=False,
            )
            full_ids = self.tokenizer(
                prompt_text, return_tensors="pt", padding=False
            )["input_ids"][0].tolist()

        # Split full_ids on <|im_start|> into one segment per message (mirrors
        # patch_prompt_tokens so the two agree).
        split_ids = self._split_on_im_start(full_ids)

        # Defensive recovery: an LLM may *generate* <|im_start|> as an ordinary
        # output token during rollout (degenerate/looping samples). Those stray
        # ids land inside a message's content and are wrongly treated as segment
        # boundaries, inflating split_ids far beyond the timeline length. Every
        # real chat-template segment spans <|im_start|>...<|im_end|>, so when
        # the im_start split diverges from the timeline, re-split on <|im_end|>
        # (absorbing stray content-internal <|im_start|> ids into the real
        # segment they belong to), restoring 1:1.
        #
        # sep_via_eos=True forces this EOS-based split unconditionally, even
        # when the im_start split already aligns — for templates where im_start
        # is not a reliable per-message boundary but im_end is.
        if sep_via_eos or len(split_ids) != len(timeline):
            split_ids = self._recover_segments_via_im_end(full_ids, timeline, force=sep_via_eos)

        assert len(split_ids) == len(timeline), (
            f"tokenize_and_slice_timeline: {len(split_ids)} segments for "
            f"{len(timeline)} messages after recovery — a template quirk that "
            f"must be handled explicitly."
        )
        for m, seg in zip(timeline, split_ids):
            m.token_arr = seg


    def _split_on_im_start(self, full_ids: List[int]) -> List[List[int]]:
        """Split full_ids on the <|im_start|> token; each segment starts with
        the <|im_start|> id. Mirrors patch_prompt_tokens so the two agree."""
        split_ids: List[List[int]] = []
        tmp: List[int] = []
        for tid in full_ids:
            if tid != self._im_start_token_id:
                tmp.append(tid)
            else:
                if tmp:
                    split_ids.append(tmp)
                tmp = [tid]
        if tmp:
            split_ids.append(tmp)
        return split_ids

    def _recover_segments_via_im_end(
        self, full_ids: List[int], timeline: List[ExtendedMessage],
        force: bool = False,
    ) -> List[List[int]]:
        """Recover 1:1 message↔segment alignment when a message's content
        contains spurious ``<|im_start|>`` tokens (an LLM generated the special
        token during rollout, breaking the im_start-based split).

        Every real chat-template segment spans ``<|im_start|>...<|im_end|>``,
        so re-splitting on ``<|im_end|>`` — keeping each segment from an
        ``<|im_start|>`` up to and including the next ``<|im_end|>`` — restores
        one segment per message regardless of stray ``<|im_start|>`` ids inside
        content.

        ``force`` (set by ``sep_via_eos=True``): skip the spurious-im_start
        precondition check and always re-split on ``<|im_end|>``, even when no
        content contains a literal ``<|im_start|>``. Used when the caller knows
        im_end is the right boundary regardless of why im_start diverged.

        Without ``force``, raises if no message content contains a literal
        ``<|im_start|>``: then the divergence is an unknown template quirk,
        not the recoverable spurious-im_start case, and must surface rather
        than misalign silently.
        """
        # If no content contains a literal <|im_start|>, the im_start split and
        # the timeline disagree for an unknown reason — surface it (unless the
        # caller forced the im_end path via sep_via_eos).
        if not force and not any(
            (getattr(m, "content", "") or "").count("<|im_start|>") > 0
            for m in timeline
        ):
            raise AssertionError(
                f"tokenize_and_slice_timeline: <|im_start|> segment count "
                f"({len(self._split_on_im_start(full_ids))}) != timeline length "
                f"({len(timeline)}), but no message content contains a literal "
                f"<|im_start|> — unknown template quirk, not the recoverable "
                f"spurious-im_start case."
            )

        # Re-split matching _split_on_im_start on well-formed renders (each
        # segment = one <|im_start|> ... up-to-next-<|im_start|>, so the \n
        # after an <|im_end|> stays in its segment) while absorbing stray
        # content-internal <|im_start|> ids emitted by the LLM.
        #
        # A <|im_start|> is a real segment boundary only if it opens a segment
        # that an <|im_end|> eventually closes. Stray <|im_start|> ids inside
        # content are never followed by their own <|im_end|>, so they don't
        # open a segment and are absorbed into the surrounding real segment.
        # We track open/closed state: open at a real <|im_start|>, close at the
        # <|im_end|> that terminates it; between close and the next real
        # <|im_start|> (e.g. the trailing \n) belongs to the just-closed
        # segment, matching _split_on_im_start.
        im_start, im_end = self._im_start_token_id, self._im_end_token_id
        segments: List[List[int]] = []
        cur: List[int] = []
        open_seg = False
        for tid in full_ids:
            if not open_seg:
                if tid == im_start:
                    # Start of a new real segment. (Leading filler, if any,
                    # was already attached to the previous closed segment.)
                    cur = [tid]
                    open_seg = True
                elif segments:
                    # Between segments: trailing filler (e.g. \n after the
                    # last <|im_end|>) belongs to the previous segment.
                    segments[-1].append(tid)
                # else: pre-first-segment filler — drop (none in well-formed).
            else:
                cur.append(tid)
                if tid == im_end:
                    segments.append(cur)
                    cur = []
                    open_seg = False
        # Trailing tokens after the last <|im_end|> without a following real
        # <|im_start|> belong to the last closed segment.
        if cur:
            if segments:
                segments[-1].extend(cur)
            else:
                segments.append(cur)
        return segments


    def step_prepare(self, messages: List[dict], tools: List = [], timeline_uuid: str = ""):
        disable_toolcalls = self.config.ajet.rollout.force_disable_toolcalls
        tools = self.preprocess_tools_field(tools, disable_toolcalls=disable_toolcalls)
        timeline = self.step_spawn_timeline(messages, tools, disable_toolcalls)

        # Tokenize the whole timeline once and slice per message so each
        # token_arr is an exact contiguous slice of the full-conversation
        # render (drift-free by construction; see the method docstring).
        # Consecutive tool messages (which the chat template folds into one
        # <|im_start|>user segment) are handled inside the slicer, so no
        # separate merge step is needed here.
        self.tokenize_and_slice_timeline(timeline, tools)

        # check token overflow (converted_message reflects the timeline the
        # slicer just tokenized, so the overflow check sees the same tokens)
        converted_message = self.to_role_content(timeline)
        context_safe, token_overflow, info = self.check_context_token_num_safe(
            converted_message, tools
        )
        custom_sampling_params = {}
        if not context_safe:
            self.context_overflow = True
            logger.warning(f"[{self.workflow_task.episode_uuid}] Stop tracking timelines because {info}.")


        self.timeline_cache[timeline_uuid] = timeline
        return context_safe, token_overflow, info, converted_message, custom_sampling_params, tools



    def step_track(
        self,
        llm_output,
        context_safe,
        converted_message: List[dict],
        tools: List = [],
        timeline_uuid: str = "",
    ):
        assert timeline_uuid in self.timeline_cache, "Timeline UUID not found in cache. Please ensure `step_prepare` is called before `step_track`."

        # round ++
        self.llm_call_cnt += 1

        # get timeline from cache
        timeline = self.timeline_cache.pop(timeline_uuid, [])
        if not self.already_mad_flag:
            if (
                compute_string_madness(
                    completion=llm_output["content"],
                    checklist=self.config.ajet.rollout.compute_madness_checklist,
                )
                < 0.0
            ):
                self.already_mad_flag = True

        tool_calls = self.detect_tool_call_madness(llm_output)

        # add llm_output to timeline and save
        llm_ext_msg = ExtendedMessage(
            author="llm",
            role="assistant",
            content=llm_output["content"],
            token_generator="manual",
            tool_calls=tool_calls,
            tokenizer=self.tokenizer,
        )
        input_msg_ref = copy.deepcopy(converted_message)
        (
            precise_manual_token,
            token_logprob_arr,
            loss_mask_from_diff,
            lack_normal_eos,
        ) = self.get_token_inc_from_llm_response(input_msg_ref, llm_output, tools=tools)
        llm_ext_msg.token_arr = precise_manual_token
        llm_ext_msg.token_logprob_arr = token_logprob_arr
        llm_ext_msg.lack_normal_eos = lack_normal_eos
        llm_ext_msg.manual_loss_mask_from_diff = loss_mask_from_diff

        assert (
            len(precise_manual_token)
            <= self.config.ajet.rollout.max_response_length_in_one_turn
        ), f"Generated token length {len(precise_manual_token)} exceeds max_response_length_in_one_turn {self.config.ajet.rollout.max_response_length_in_one_turn}"

        # run generated token callback, usually to monitor token output rate ( e.g. 164 tokens/sec )
        self.generated_token_callback_fn(llm_ext_msg.token_arr)

        # take snapshot of current timeline
        if context_safe:
            if (
                "prompt_text" in llm_output and "prompt_token_ids" in llm_output
            ):
                # fix Retokenization Drift
                timeline = self.patch_prompt_tokens(
                    prompt_text=llm_output["prompt_text"],
                    prompt_token_ids=llm_output["prompt_token_ids"],
                    previous_ext_context=timeline,
                )

            self.save_llm_interaction_timeline(tools, llm_ext_msg, timeline)
        return None



    def save_llm_interaction_timeline(self, tools, llm_ext_msg, timeline):
        """Save the LLM interaction timeline by adding the LLM response to `self.saved_timelines`
        """
        timeline += [llm_ext_msg]
        _, length = self.get_context_token_num_and_safety(timeline, tools)
        if length > self.config.ajet.rollout.max_model_len:
            raise RuntimeError(
                    f"Unexpected token overflow after adding LLM response. Full context length {length}, generated token length {len(llm_ext_msg.token_arr)}"
                )

        assert timeline[0].first_message, "First message should be marked as first_message"

        # assert all other message is not first_message
        for i in range(1, len(timeline)):
            assert not timeline[i].first_message

        # no longer write anything
        if self._read_only:
            logger.exception("Timeline is in read-only mode, should not save new timeline. Please report a github issue if you see this error.")
            return

        # save to self.saved_timelines
        self.saved_timelines += [copy.deepcopy(timeline)]

        # warn when merge fails
        timeline_merging_policy: TimelineMergingPolicyConfig = self.config.ajet.context_tracker.timeline_merging_policy
        if (
            self.config.ajet.context_tracker.detect_timeline_snap
            and len(self.saved_timelines) >= 2
            and (
                not is_timeline_mergeable(
                    self.saved_timelines[-1],
                    self.saved_timelines[-2],
                    timeline_merging_policy
                )
            )
        ):
            logger.bind(exception=True).info(f"General Warning: merge failure discovered.\n")
        return


    def detect_tool_call_madness(self, llm_output):
        """Detect whether the tool call format from LLM output is correct or not.
        """
        log_tool = self.config.ajet.context_tracker.log_tool_format_check
        detailed_log = self.config.ajet.context_tracker.log_tool_format_error_detail

        err_type = ""
        if llm_output.get("tool_calls", []):
            # llm_output["tool_calls"] is not None, and is not []
            tool_calls = llm_output["tool_calls"]
            if "wrong_toolcall" in self.config.ajet.rollout.compute_madness_checklist:
                # copy_tool_calls = copy.deepcopy(tool_calls)
                # Shallow copy is sufficient - we're only reading the data
                copy_tool_calls = tool_calls
                wrong_toolcall = False
                for i in range(len(copy_tool_calls)):
                    if ("function" in copy_tool_calls[i]) and (
                        "arguments" in copy_tool_calls[i]["function"]
                    ):
                        try:
                            expect_dict = json.loads(copy_tool_calls[i]["function"]["arguments"])
                            if not isinstance(expect_dict, dict):
                                wrong_toolcall = True
                                err_type = "cannot parse arguments"
                        except Exception:
                            wrong_toolcall = True
                            err_type = "arguments not json"
                    else:
                        wrong_toolcall = True
                        err_type = "no function or no arguments"
                if wrong_toolcall:
                    if detailed_log:
                        logger.bind(exception=True).warning(
                            f"Detected wrong toolcall format from LLM output: \n---*({err_type})*---\n{llm_output['tool_calls']}\n---*-*---\n"
                        )
                    if log_tool:
                        logger.bind(exception=True).warning(
                            f"Detected wrong toolcall format from LLM content"
                        )
                    self.already_mad_flag = True
                else:
                    if log_tool:
                        logger.success("Toolcall format check passed.")

        elif "<tool_call>" in llm_output["content"]:
            if detailed_log:
                logger.bind(exception=True).warning(
                    f"Detected wrong toolcall format from LLM content: \n---*-*---\n{llm_output['content']}\n---*-*---\n"
                )
            if "wrong_toolcall" in self.config.ajet.rollout.compute_madness_checklist:
                if log_tool:
                    logger.bind(exception=True).warning(
                        f"Detected wrong toolcall format from LLM content"
                    )
                self.already_mad_flag = True
            tool_calls = []
        else:
            tool_calls = []
        return tool_calls



    def patch_prompt_tokens(
        self,
        prompt_text: str,
        prompt_token_ids: List[int],
        previous_ext_context: List[ExtendedMessage],
    ) -> List[ExtendedMessage]:
        """
        fix retokenization drift
        prompt_text = llm_output["prompt_text"]:            [this llm call] the prompt in text format used in generation
        prompt_token_ids = llm_output["prompt_token_ids"]:  [this llm call] the prompt token ids used in generation (prompt_text->prompt_token_ids using tokenizer)
        previous_ext_context:                               [from previous context] the context history
        """

        # remove tailing, usually `<|im_start|> assistant`
        if prompt_text.endswith(self.generation_prompt):
            prompt_text = prompt_text[: -len(self.generation_prompt)]
            # prompt_token_ids = prompt_token_ids[: -len(self.generation_prompt_token)]

        # split CURRENT prompt token ids into message level (split_prompt_token_ids is List[List[int]])
        split_prompt_token_ids = []
        tmp = []
        for i in range(len(prompt_token_ids)):
            if prompt_token_ids[i] != self._im_start_token_id:
                tmp += [prompt_token_ids[i]]
            else:
                if len(tmp) > 0:
                    split_prompt_token_ids += [tmp]
                tmp = [prompt_token_ids[i]]
        if len(tmp) > 0:
            split_prompt_token_ids += [tmp]

        # split CURRENT prompt text into message level (corresponding to split_prompt_token_ids)
        prompt_text_split = prompt_text.split("<|im_start|>")
        assert prompt_text_split[0] == "", "Prompt text should start with <|im_start|>"
        prompt_text_split = prompt_text_split[1:]  # remove the first empty string
        for i in range(len(prompt_text_split)):
            prompt_text_split[i] = "<|im_start|>" + prompt_text_split[i]

        # context HISTORY prompt text
        current_prompt_text = []
        for j in range(len(previous_ext_context)):
            current_prompt_text += [self.tokenizer.decode(previous_ext_context[j].token_arr)]

        # HISTORY context length vs CURRENT prompt length
        if len(previous_ext_context) != len(prompt_text_split):
            logger.bind(exception=True).error(f"Length mismatch when patching prompt tokens. Previous ext context length: {len(previous_ext_context)}, prompt text split length: {len(prompt_text_split)}. Replacing all tokens.")

        # try to recover tokens
        if self.config.ajet.context_tracker.fix_retokenization_drift:
            previous_ext_context = self.ensure_retokenization_perfect_match(
                previous_ext_context,   # HISTORY
                split_prompt_token_ids, # CURRENT
                prompt_text_split,      # CURRENT
                current_prompt_text     # HISTORY
            )

        # remove extra messages
        if len(previous_ext_context) != len(prompt_text_split):
            previous_ext_context = previous_ext_context[: len(prompt_text_split)]

        return previous_ext_context


    def ensure_retokenization_perfect_match(self, previous_ext_context, split_prompt_token_ids, prompt_text_split, current_prompt_text):
        """
        Ensure the retokenization is perfectly matched between HISTORY and CURRENT

        previous_ext_context: the context history in ExtendedMessage format, which contains token_arr (token ids)
        split_prompt_token_ids: the prompt token ids of CURRENT prompt, split into message level (List[List[int]])
        prompt_text_split: the prompt text of CURRENT prompt, split into message level (List[str])
        current_prompt_text: the prompt text of HISTORY context, converted from token_arr to text using tokenizer, in message level (List[str])
        """

        for j in range(len(previous_ext_context)):
            vllm_token_array = split_prompt_token_ids[j]
            tracker_token_array = previous_ext_context[j].token_arr
            if vllm_token_array == tracker_token_array:
                # good, everything is perfect
                continue
            else:
                # otherwise, we throw a warning (do not worry, this causes almost no influence in the training)
                print_dict(
                    {
                        "expected_prompt_text": prompt_text_split[j],       # from llm_output["prompt_text"], converted directly from messages using apply_chat_template, passway (messages->apply_chat_template->text)
                        "current_prompt_text": current_prompt_text[j],      # history prompt text converted from token_arr to text using tokenizer, passway (messages->extended_message->incremental apply_chat_template->token_arr->text)
                        "expected_token_ids": vllm_token_array,             # from llm_output["prompt_token_ids"], passway (messages->apply_chat_template->token)
                        "current_token_ids": tracker_token_array,           # from previous_ext_context[j].token_arr, passway (messages->extended_message->incremental apply_chat_template->token_arr)
                    },
                    mod="exception",
                    header="Prompt token ids mismatch (fixing drift by `token_arr=vllm_token_array`).",
                )
                previous_ext_context[j].token_arr = vllm_token_array
        return previous_ext_context


    def process_reward(self, reward_structure: Reward):
        self.reward_structure = reward_structure
        # TODO: support multi-step reward
        # in current implementation, all reward in all step equals
        # we'll implement fine-grained step reward in future versions
        self.reward_structure.step_reward_arr = [
            self.compute_step_level_reward(
                index=i,
                total_steps=len(self.saved_timelines),
            )
            for i in range(len(self.saved_timelines))
        ]


    def generate_log(self, task_id=None, global_step="NA"):
        task_id = self.task_id
        nested_items_print_buffer = {}
        step_reward = 0.0

        for index, ext_steps in enumerate(self.saved_timelines):
            tracker_tokenized = self.tokenize_steps(
                ext_steps=ext_steps,
                index=index,
                total_steps=len(self.saved_timelines),
            )
            text_arr = self.tokenizer.batch_decode([[t] for t in tracker_tokenized["input_ids"]])
            input_id_arr = [str(t) for t in tracker_tokenized["input_ids"]]
            # loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in tracker_tokenized["loss_mask"]]
            logprobs = [INVALID_LOG_PROB_VALUE] * len(
                tracker_tokenized["prompt_ids"]
            ) + tracker_tokenized["response_logprobs"]
            # Create adjusted color array using batch processing for better performance
            base_colors = ["#09ABCF" if mask == 1 else "#D98510" for mask in tracker_tokenized["loss_mask"]]
            loss_mask_color_abl_arr = adjust_color_hsl_batch(base_colors, logprobs)
            logprob_text_arr = [
                (f"{logprob:.4f}" if logprob != INVALID_LOG_PROB_VALUE else "N/A")
                for logprob in logprobs
            ]

            buffer = {
                "text_arr": text_arr,
                "logprob_arr": logprob_text_arr,
                "input_id_arr": input_id_arr,
                "loss_mask_color_arr": loss_mask_color_abl_arr,
            }
            raw_reward = self.reward_structure.raw_reward
            step_reward: float = self.reward_structure.step_reward_arr[index]
            try:
                step_advantage = self.reward_structure.step_advantage[index]
                step_advantage_simple = self.reward_structure.step_advantage_simple[index]
            except Exception:
                step_advantage = 0.0
                step_advantage_simple = 0.0
            task_outcome = str(self.reward_structure.success_rate)
            selectors = [task_id, task_outcome, str(index)]
            len_prompt_ids = len(tracker_tokenized["prompt_ids"])
            len_response_ids = len(tracker_tokenized["response_ids"])
            len_input_ids = len(tracker_tokenized["input_ids"])
            assert (
                len_prompt_ids + len_response_ids == len_input_ids
            ), "len_prompt_ids + len_response_ids should equal to len_input_ids"
            nested_items_print_buffer[".".join(selectors)] = NestedJsonItem(
                item_id="item",  # type: ignore
                outcome=task_outcome,  # type: ignore
                len_prompt_ids=len_prompt_ids,  # type: ignore
                len_response_ids=len_response_ids,  # type: ignore
                len_input_ids=len_input_ids,  # type: ignore
                raw_reward=f"{float(raw_reward):.3f}",  # type: ignore
                step_reward=f"{float(step_reward):.3f}",  # type: ignore
                step_advantage=f"{float(step_advantage):.3f}",  # type: ignore
                step_advantage_simple=f"{float(step_advantage_simple):.3f}",  # type: ignore
                content=SeqItem(
                    text=buffer["text_arr"],  # text content
                    title=buffer["logprob_arr"],  # mouse hover text
                    count=buffer["input_id_arr"],  # highlight text # type: ignore
                    color=buffer["loss_mask_color_arr"],  # color
                ),
            )

        print_nested(
            nested_items_print_buffer,
            main_content="This is the main content of the nested JSON",
            header=f"[{global_step}] Task {task_id} (Reward {float(step_reward):.3f})",  # type: ignore
            mod="rollout",
            narrow=False,
            attach="copy this",  # type: ignore
        )


    def group_merge(self) -> List[List[ExtendedMessage]]:
        timeline_merging_policy: TimelineMergingPolicyConfig = self.config.ajet.context_tracker.timeline_merging_policy
        self.saved_timelines = merge_tracker_timelines(self.saved_timelines, timeline_merging_policy)
        self._read_only = True

        return self.saved_timelines


    def group_tokenize(self, cache=False):
        if hasattr(self, "group_tokenized_cache"):
            return getattr(self, "group_tokenized_cache")
        else:
            result = self.group_tokenize_multi_group()
            if cache:
                setattr(self, "group_tokenized_cache", result)
            return result


    def get_context_token_num_and_safety(self, ext_messages: List[ExtendedMessage], tools: List = []) -> Tuple[bool, int]:  # type: ignore
        dict_messages = self.to_role_content(ext_messages)
        prompt_text = ajet_apply_chat_template(
            tokenizer=self.tokenizer,
            conversation=dict_messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        length = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])  # type: ignore
        max_response_length = self.config.ajet.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.ajet.rollout.max_model_len
        max_seq_length: int = max_model_len - max_response_length

        if length < max_seq_length:
            return True, length
        else:
            return False, length


    def check_context_token_num_safe(
        self, messages: List, tools: List = []
    ) -> Tuple[bool, bool, str]:
        prompt_text = ajet_apply_chat_template(
            tokenizer=self.tokenizer,
            conversation=messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_token_length = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])  # type: ignore
        max_response_length_in_one_turn = self.config.ajet.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.ajet.rollout.max_model_len
        max_seq_length: int = max_model_len - max_response_length_in_one_turn
        # prompt_token_length: the prompt_token_length of current all previous context
        # max_seq_length: max_model_len - max_response_length_in_one_turn
        if prompt_token_length < max_seq_length:
            token_overflow = False
        else:
            token_overflow = True
        if self.should_interrupt_soft_fn():
            ret = (False, token_overflow, "externally_interrupted")
        elif self.already_mad_flag and self.config.ajet.rollout.agent_madness_termination:
            ret = (False, token_overflow, "already_mad")
        elif prompt_token_length < max_seq_length:
            ret = (
                True,
                token_overflow,
                f"safe[{prompt_token_length} < {max_model_len} - {max_response_length_in_one_turn}]",
            )
        else:
            ret = (False, token_overflow,
                   f"token_overflow(prompt_token_length.{prompt_token_length}>=max_model_len.{max_model_len}-max_response_length_in_one_turn.{max_response_length_in_one_turn})")
        return ret
