import copy
from loguru import logger

from agentscope.model import DashScopeChatModel
from astune.schema.trajectory import Reward
from transformers.tokenization_utils import PreTrainedTokenizer
from astune.context_tracker.basic_tracker import (
    BasicContextTracker,
    ExtendedMessage,
    replace_token_ids,
)
from astune.utils.color_hsl import adjust_color_hsl
from astune.utils.tokenizer import astune_apply_chat_templat
from astune.utils.compute_madness import compute_string_madness
from astune.schema.extended_msg import INVALID_LOG_PROB_VALUE
from astune.context_tracker.agentscope_tracker.timeline_merging import (
    merge_tracker_timelines,
    can_merge_steps,
)

from typing import Any, List, Tuple, Union, Dict
from beast_logger import (
    print_nested,
    print_listofdict,
    NestedJsonItem,
    SeqItem,
)
import json

class MultiAgentContextTracking(BasicContextTracker):

    def __init__(
        self,
        llm_chat_fn,
        tokenizer: PreTrainedTokenizer,
        config,
        should_interrupt_fn,
        generated_token_callback_fn,
        **kwargs,
    ):
        super().__init__(config, tokenizer, **kwargs)
        self.llm_chat_fn = llm_chat_fn
        self.tokenizer = tokenizer
        self.should_interrupt_fn = should_interrupt_fn
        self.generated_token_callback_fn = generated_token_callback_fn
        self.context_overflow = False
        self.output_kwargs = {}
        self.input_kwargs = {}

    def step_prepare(self, messages: List[dict], tools: List = []):
        self.full_context = []
        consider_roles = ["user", "assistant", "system", "tool"]
        disable_toolcalls = self.config.astune.rollout.agentscope_disable_toolcalls
        if disable_toolcalls:
            consider_roles.remove("tool")
            tools = []
        else:
            # rerank tool parameters to improve compatibility
            for i in range(len(tools)): tools[i]['function']['parameters'] = tools[i]['function'].pop('parameters')

        for i, msg in enumerate(messages):
            if (disable_toolcalls) and (not isinstance(msg["content"], str)):
                continue
            if msg["role"] not in consider_roles:
                continue
            if not isinstance(msg["content"], str):
                author = "env"
                ignore = False
                str_content = ""
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
                if ignore:
                    continue
                msg["content"] = str(msg["content"])  # TODO: better handling mm data
            if msg["role"] == "system":
                author = "initialization"
            if msg["role"] == "tool":
                author = "env"
            else:
                author = "env"

            self.full_context += [
                ExtendedMessage(
                    author=author,
                    role=msg["role"],
                    content=msg["content"],
                    tokenizer=self.tokenizer,
                    tools=tools,
                    tool_calls=(msg["tool_calls"] if "tool_calls" in msg else []),
                    token_generator="auto",
                )
            ]

        # check token overflow
        converted_message = self.to_role_content(self.full_context)
        context_safe, token_overflow, info = self.check_context_token_num_safe(converted_message, tools)
        custom_sampling_params = {}
        if not context_safe:
            self.context_overflow = True

        return context_safe, token_overflow, info, converted_message, custom_sampling_params, tools

    def step_track(
        self,
        llm_output,
        context_safe,
        converted_message: List[dict],
        tools: List = [],
    ):
        if not self.already_mad_flag:
            if (
                compute_string_madness(
                    completion=llm_output["content"],
                    checklist=self.config.astune.rollout.compute_madness_checklist,
                )
                < 0.0
            ):
                self.already_mad_flag = True

        # dummy response for now
        token_generator = "manual"
        err_type = ""
        if llm_output.get("tool_calls", []): # is not None, and is not []
            tool_calls = llm_output["tool_calls"]
            if ("wrong_toolcall" in self.config.astune.rollout.compute_madness_checklist):
                # check tool call formating
                copy_tool_calls = copy.deepcopy(tool_calls)
                wrong_toolcall = False
                for i in range(len(copy_tool_calls)):
                    if ('function' in copy_tool_calls[i]) and ('arguments' in copy_tool_calls[i]['function']):
                        try:
                            expect_dict = json.loads(copy_tool_calls[i]['function']['arguments'])
                            if not isinstance(expect_dict, dict):
                                wrong_toolcall = True
                                err_type = "cannot parse arguments"
                                from vsdb import bp; bp("UPUP1")
                        except:
                            wrong_toolcall = True
                            err_type = "arguments not json"
                            from vsdb import bp; bp("UPUP3")
                    else:
                        wrong_toolcall = True
                        err_type = "no function or no arguments"
                        from vsdb import bp; bp("UPUP4")
                if wrong_toolcall:
                    logger.bind(exception=True).error(f"Detected wrong toolcall format from LLM output: \n---*({err_type})*---\n{llm_output['tool_calls']}\n---*-*---\n")
                    self.already_mad_flag = True
                else:
                    logger.success("Toolcall format check passed.")
        elif ('<tool_call>' in llm_output["content"]):
            from vsdb import bp; bp("UPUP2")
            logger.bind(exception=True).error(f"Detected wrong toolcall format from LLM content: \n---*-*---\n{llm_output['content']}\n---*-*---\n")
            self.already_mad_flag = True
            tool_calls = []
        else:
            tool_calls = []

        llm_ext_msg = ExtendedMessage(
            author="llm",
            role="assistant",
            content=llm_output["content"],
            token_generator=token_generator,
            tool_calls=tool_calls,
            tokenizer=self.tokenizer,
        )

        if token_generator == "manual":
            input_msg_ref = copy.deepcopy(converted_message)
            token_arr_method2, token_logprob_arr = self.get_token_inc_from_vllm_response(
                input_msg_ref, llm_output, tools=tools
            )
            assert (
                len(token_arr_method2) <= self.config.astune.rollout.max_response_length_in_one_turn
            ), f"Generated token length {len(token_arr_method2)} exceeds max_response_length_in_one_turn {self.config.astune.rollout.max_response_length_in_one_turn}"
            llm_ext_msg.token_arr = token_arr_method2
            llm_ext_msg.token_logprob_arr = token_logprob_arr
            self.generated_token_callback_fn(llm_ext_msg.token_arr)

        # take snapshot of current timeline
        if context_safe:
            self.full_context += [llm_ext_msg]
            is_safe, length = self.get_context_token_num_and_safety(self.full_context, tools)
            if length > self.config.astune.rollout.max_model_len:
                raise RuntimeError(
                    f"Unexpected token overflow after adding LLM response. Full context length {length}, generated token length {len(llm_ext_msg.token_arr)}"
                )
            self.grouped_steps += [copy.deepcopy(self.full_context)]

            # DEBUG = True   # warn when merge fails
            # if (
            #     DEBUG
            #     and len(self.grouped_steps) >= 2
            #     and (not can_merge_steps(self.grouped_steps[-1], self.grouped_steps[-2]))
            # ):
            #     print(f"General Warning: merge failure discovered.")
        return None

    def process_reward(self, reward_structure: Reward):
        self.reward_structure = reward_structure
        ext_steps = self.full_context
        self.reward_structure.step_reward = [
            self.compute_step_level_reward(
                ext_steps=ext_steps,
                index=i,
                total_steps=len(self.grouped_steps),
            )
            for i in range(len(self.grouped_steps))
        ]

    def generate_log(self, task_id=None, global_step="NA"):
        task_id = self.task_id
        nested_items_print_buffer = {}
        step_reward = 0.0

        for index, ext_steps in enumerate(self.grouped_steps):
            cmt_tokenized = self.tokenize_steps(
                ext_steps=ext_steps,
                index=index,
                total_steps=len(self.grouped_steps),
            )
            text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]
            input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]
            # loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in cmt_tokenized["loss_mask"]]
            logprobs = [INVALID_LOG_PROB_VALUE] * len(cmt_tokenized["prompt_ids"]) + cmt_tokenized[
                "response_logprobs"
            ]
            # Create adjusted color array
            loss_mask_color_abl_arr = [
                (
                    adjust_color_hsl("#09ABCF", logprob)
                    if mask == 1
                    else adjust_color_hsl("#D98510", logprob)
                )
                for mask, logprob in zip(cmt_tokenized["loss_mask"], logprobs)
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
            step_reward: float = self.reward_structure.step_reward[index]
            try:
                step_advantage = self.reward_structure.step_advantage[index]
                step_advantage_simple = self.reward_structure.step_advantage_simple[index]
            except:
                step_advantage = 0.0
                step_advantage_simple = 0.0
            task_outcome = str(self.reward_structure.success_rate)
            selectors = [task_id, task_outcome, str(index)]
            len_prompt_ids = len(cmt_tokenized["prompt_ids"])
            len_response_ids = len(cmt_tokenized["response_ids"])
            len_input_ids = len(cmt_tokenized["input_ids"])
            assert (
                len_prompt_ids + len_response_ids == len_input_ids
            ), "len_prompt_ids + len_response_ids should equal to len_input_ids"
            nested_items_print_buffer[f".".join(selectors)] = NestedJsonItem(
                item_id=f"item",  # type: ignore
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
        self.grouped_steps = merge_tracker_timelines(self.grouped_steps)
        return self.grouped_steps

    def group_tokenize(self):
        return self.group_tokenize_multi_group()

    def get_inc(self, text_frag_from, text_frag_to):
        """
        Get the incremental token array from text_frag_from to text_frag_to.
        """
        tokenizer_output = self.tokenizer(text_frag_from, return_tensors="pt", padding=False)
        tokenizer_input_ids = tokenizer_output["input_ids"][0].tolist()  # type: ignore
        token_ids_acc = tokenizer_input_ids

        tokenizer_output = self.tokenizer(text_frag_to, return_tensors="pt", padding=False)
        input_ids = tokenizer_output["input_ids"][0].tolist()  # type: ignore
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

    def get_context_token_num_and_safety(self, ext_messages: List[ExtendedMessage], tools: List = []) -> Tuple[bool, int]:  # type: ignore
        dict_messages = self.to_role_content(ext_messages)
        prompt_text = astune_apply_chat_templat(
            tokenizer=self.tokenizer,
            conversation=dict_messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False
        )
        length = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])  # type: ignore
        max_response_length = self.config.astune.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.astune.rollout.max_model_len
        max_seq_length: int = max_model_len - max_response_length

        if length < max_seq_length:
            ret = [True, length]
        else:
            ret = [False, length]
        return tuple(ret)

    def check_context_token_num_safe(self, messages: List, tools: List = []) -> Tuple[bool, str]:
        prompt_text = astune_apply_chat_templat(
            tokenizer=self.tokenizer,
            conversation=messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False
        )
        length = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])  # type: ignore
        max_response_length = self.config.astune.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.astune.rollout.max_model_len
        max_seq_length: int = max_model_len - max_response_length
        if length < max_seq_length:
            token_overflow = False
        else:
            token_overflow = True
        if self.should_interrupt_fn():
            ret = [False, token_overflow, "externally_interrupted"]
        elif self.already_mad_flag and self.config.astune.rollout.agent_madness_termination:
            ret = [False, token_overflow, "already_mad"]
        elif length < max_seq_length:
            ret = [True, token_overflow, f"safe[{length} < {max_model_len} - {max_response_length}]"]
        else:
            ret = [False, token_overflow, "token_overflow"]
        return tuple(ret)

    def to_role_content(self, ext_msg_array: List[ExtendedMessage]) -> List:
        result = []
        for ext_msg in ext_msg_array:
            d = {
                "role": ext_msg.role,
                "content": ext_msg.content_for_future,
            }
            if ext_msg.tool_calls:
                d.update({"tool_calls": ext_msg.tool_calls})
            result.append(d)
        return result
