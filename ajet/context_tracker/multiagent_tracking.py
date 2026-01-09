# flake8: noqa: F541, F841
import copy
import json
from dataclasses import dataclass, field
from typing import List, Tuple

from beast_logger import NestedJsonItem, SeqItem, print_dict, print_nested
from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer

from ajet.context_tracker.timeline_merging.timeline_merging import (
    merge_tracker_timelines, is_timeline_mergeable
)
from ajet.context_tracker.basic_tracker import (
    BaseContextTracker,
    ExtendedMessage,
)
from ajet.schema.extended_msg import INVALID_LOG_PROB_VALUE
from ajet.schema.trajectory import Reward
from ajet.utils.color_hsl import adjust_color_hsl
from ajet.utils.compute_madness import compute_string_madness
from ajet.utils.tokenizer import ajet_apply_chat_template

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




class MultiAgentContextTracker(BaseContextTracker):
    """
    Context tracker is responsible to monitor and process LLM IO.
    Each context tracker is responsible for ONE episode run only.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config,
        should_interrupt_fn,
        generated_token_callback_fn,
        episode_uuid: str,
        **kwargs,
    ):
        super().__init__(config, tokenizer, **kwargs)
        self.tokenizer = tokenizer
        self.should_interrupt_fn = should_interrupt_fn
        self.generated_token_callback_fn = generated_token_callback_fn
        self.context_overflow = False
        self.output_kwargs = {}
        self.input_kwargs = {}
        self.timeline_cache = {}
        self.episode_uuid = episode_uuid


    def preprocess_tools_field(self, tools: List[dict] = [], disable_toolcalls: bool = False):
        if disable_toolcalls:
            tools = []
        else:
            if tools is not None:
                # rerank tool parameters to improve compatibility
                for i in range(len(tools)):
                    tools[i]["function"]["parameters"] = tools[i]["function"].pop("parameters")
        return tools


    def step_spawn_timeline(self, messages: List[dict], tools: List = [], disable_toolcalls: bool = False) -> List[ExtendedMessage]:
        """Spawn a timeline from messages.

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

        for i, msg in enumerate(messages):
            if (disable_toolcalls) and (not isinstance(msg["content"], str)):
                continue
            if msg["role"] not in consider_roles:
                continue
            if not isinstance(msg["content"], str):
                author = "env"
                ignore = False
                str_content = ""

                # fix msg content
                if msg["content"] is None:
                    msg["content"] = ""
                elif isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if "text" not in item:
                            logger.warning(
                                f"Non-text content in message content detected: {item}. Ignoring."
                            )
                            ignore = True
                            break
                        if isinstance(item["text"], str):
                            str_content += str(item["text"])
                        else:
                            str_content = ""
                        msg["content"] = str_content
                else:
                    raise ValueError(
                        f"Unsupported non-str message content type: {type(msg['content'])}, Message:\n {msg}"
                    )

                if ignore:
                    continue
                msg["content"] = str(msg["content"])  # TODO: better handling mm data

            if msg["role"] == "system":
                author = "initialization"

            if msg["role"] == "tool":
                author = "env"
            else:
                author = "env"

            timeline += [
                ExtendedMessage(
                    author=author,
                    role=msg["role"],
                    content=msg["content"],
                    tokenizer=self.tokenizer,
                    tools=tools,
                    tool_calls=(msg["tool_calls"] if "tool_calls" in msg else []),
                    token_generator="auto",
                    first_message=(i == 0),
                )
            ]

        return timeline


    def step_prepare(self, messages: List[dict], tools: List = [], timeline_uuid: str = ""):
        disable_toolcalls = self.config.ajet.rollout.force_disable_toolcalls
        tools = self.preprocess_tools_field(tools, disable_toolcalls=disable_toolcalls)
        timeline = self.step_spawn_timeline(messages, tools, disable_toolcalls)

        # check token overflow
        converted_message = self.to_role_content(timeline)
        timeline = ExtendedMessage.check_and_merge_chained_tool_response(
            timeline, self.tokenizer
        )
        context_safe, token_overflow, info = self.check_context_token_num_safe(
            converted_message, tools
        )
        custom_sampling_params = {}
        if not context_safe:
            self.context_overflow = True

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
        timeline = self.timeline_cache.get(timeline_uuid, [])
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
            loss_mask,
            lack_normal_eos,
        ) = self.get_token_inc_from_llm_response(input_msg_ref, llm_output, tools=tools)
        llm_ext_msg.token_arr = precise_manual_token
        llm_ext_msg.token_logprob_arr = token_logprob_arr
        llm_ext_msg.lack_normal_eos = lack_normal_eos
        llm_ext_msg.manual_loss_mask_override = loss_mask

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
                # currently we make this patch to better compat with Trinity training backend
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

        # save to self.saved_timelines
        self.saved_timelines += [copy.deepcopy(timeline)]

        # DEBUG = True   # warn when merge fails
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
                copy_tool_calls = copy.deepcopy(tool_calls)
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

        # remove tailing
        if prompt_text.endswith(self.generation_prompt):
            prompt_text = prompt_text[: -len(self.generation_prompt)]
            # prompt_token_ids = prompt_token_ids[: -len(self.generation_prompt_token)]

        # split prompt token ids into message level
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

        # split prompt text into message level
        prompt_text_split = prompt_text.split("<|im_start|>")
        assert prompt_text_split[0] == "", "Prompt text should start with <|im_start|>"
        prompt_text_split = prompt_text_split[1:]  # remove the first empty string
        for i in range(len(prompt_text_split)):
            prompt_text_split[i] = "<|im_start|>" + prompt_text_split[i]

        current_prompt_text = []
        for j in range(len(previous_ext_context)):
            current_prompt_text += [self.tokenizer.decode(previous_ext_context[j].token_arr)]

        if len(previous_ext_context) != len(prompt_text_split):
            logger.bind(exception=True).error(
                f"Length mismatch when patching prompt tokens. Previous ext context length: {len(previous_ext_context)}, prompt text split length: {len(prompt_text_split)}. Replacing all tokens."
            )

        # try to recover tokens
        if self.config.ajet.context_tracker.fix_retokenization_drift:
            self.ensure_retokenization_perfect_match(previous_ext_context, split_prompt_token_ids, prompt_text_split, current_prompt_text)

        # remove extra messages
        if len(previous_ext_context) != len(prompt_text_split):
            previous_ext_context = previous_ext_context[: len(prompt_text_split)]

        return previous_ext_context


    def ensure_retokenization_perfect_match(self, previous_ext_context, split_prompt_token_ids, prompt_text_split, current_prompt_text):
        for j in range(len(previous_ext_context)):
            if prompt_text_split[j] != current_prompt_text[j]:
                # if prompt text mismatch, we can replace the tokens
                print_dict(
                    {
                        "expected_prompt_text": prompt_text_split[j],
                        "current_prompt_text": current_prompt_text[j],
                    },
                    mod="exception",
                    header="Prompt text mismatch, Please report a github issue",
                )
                previous_ext_context[j].token_arr = self.tokenizer(
                    prompt_text_split[j], return_tensors="pt", padding=False
                )
            else:
                # if prompt text match
                # we further check whether all token ids matches
                vllm_token_array = split_prompt_token_ids[j]
                tracker_token_array = previous_ext_context[j].token_arr
                if vllm_token_array == tracker_token_array:
                    # good, everything is perfect
                    continue
                else:
                    # otherwise, we throw a warning (do not worry, this causes almost no influence in the training)
                    print_dict(
                        {
                            "expected_token_ids": split_prompt_token_ids[j],
                            "current_token_ids": previous_ext_context[j].token_arr,
                        },
                        mod="exception",
                        header="Prompt token ids mismatch, Please report a github issue",
                    )


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
            text_arr = [self.tokenizer.decode(t) for t in tracker_tokenized["input_ids"]]
            input_id_arr = [str(t) for t in tracker_tokenized["input_ids"]]
            # loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in tracker_tokenized["loss_mask"]]
            logprobs = [INVALID_LOG_PROB_VALUE] * len(
                tracker_tokenized["prompt_ids"]
            ) + tracker_tokenized["response_logprobs"]
            # Create adjusted color array
            loss_mask_color_abl_arr = [
                (
                    adjust_color_hsl("#09ABCF", logprob)
                    if mask == 1
                    else adjust_color_hsl("#D98510", logprob)
                )
                for mask, logprob in zip(tracker_tokenized["loss_mask"], logprobs)
            ]
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
        return self.saved_timelines


    def group_tokenize(self):
        return self.group_tokenize_multi_group()


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
        length = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])  # type: ignore
        max_response_length = self.config.ajet.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.ajet.rollout.max_model_len
        max_seq_length: int = max_model_len - max_response_length
        if length < max_seq_length:
            token_overflow = False
        else:
            token_overflow = True
        if self.should_interrupt_fn():
            ret = (False, token_overflow, "externally_interrupted")
        elif self.already_mad_flag and self.config.ajet.rollout.agent_madness_termination:
            ret = (False, token_overflow, "already_mad")
        elif length < max_seq_length:
            ret = (
                True,
                token_overflow,
                f"safe[{length} < {max_model_len} - {max_response_length}]",
            )
        else:
            ret = (False, token_overflow, "token_overflow")
        return ret
