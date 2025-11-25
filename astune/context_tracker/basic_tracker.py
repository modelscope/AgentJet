import torch
import copy
from loguru import logger
from collections import defaultdict
from typing import List, Union, Tuple, Optional
from astune.schema.trajectory import Sample, Reward
from astune.utils.compute_madness import compute_string_madness
from astune.utils.tokenizer import astune_apply_chat_templat
from astune.context_tracker.tracker_base_attr import TrackerAttr
from astune.context_tracker.tracker_base_attr import ExtendedMessage
from astune.context_tracker.tracker_base_attr import replace_token_ids
from beast_logger import (
    register_logger,
    print_dict,
    print_listofdict,
    print_nested,
    NestedJsonItem,
    SeqItem,
)


class BasicContextTracker(TrackerAttr):
    """
    A linear context tracker template that handles the conversation flow between LLM and environment.
    This class manages the context window, tokenization, and message history in a linear fashion.

    Attributes:
        config: Configuration object containing environment and model settings
        tokenizer: Tokenizer instance for processing text
        full_context (List[ExtendedMessage]): List of all messages in the conversation
        current_context_status (str): Current status of the context
        max_seq_length (int): Maximum sequence length for the context window
        max_env_output_length (int): Maximum length for environment outputs
        terminal_rewards_dict (dict): Dictionary storing terminal rewards
    """

    def __init__(self, config, tokenizer, **kwargs):
        super().__init__(config, tokenizer, **kwargs)
        self.generation_prompt_token = self.get_generation_prompt_token()


    def prepare_previous_context(self, mod="future"):
        """
        Prepare the input context for future LLM call.

        Returns:
            list: Array of message dictionaries containing role and content_for_future,
                 formatted for LLM input.
        """
        if mod == "future":
            message_arr = [
                {
                    "role": c.role,
                    "content": c.content_for_future,
                    "tool_calls": c.tool_calls,
                }
                for c in self.full_context
            ]
        elif mod == "raw":
            message_arr = [
                {
                    "role": c.role,
                    "content": c.content,
                    "tool_calls": c.tool_calls,
                }
                for c in self.full_context
            ]
        else:
            raise ValueError(
                f"Unknown mod {mod} in prepare_previous_context, only support 'future' and 'raw'"
            )

        # remove tool_calls from messages if empty
        for i in range(len(message_arr)):
            if not message_arr[i]["tool_calls"]:
                message_arr[i].pop("tool_calls")
        return message_arr

    def check_context_token_num_safe(
        self, messages: List[dict], tools: List[dict] = []
    ) -> Tuple[bool, str]:
        def get_seq_length(messages):
            prompt_text = astune_apply_chat_templat(
                tokenizer=self.tokenizer,
                conversation=messages,
                tools=tools,
                add_generation_prompt=False,
                tokenize=False,
            )
            return len(
                self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0]
            )

        if self.already_mad_flag and self.config.astune.rollout.agent_madness_termination:
            return False, "already_mad"
        messages = self.prepare_previous_context(mod="raw")
        if (
            get_seq_length(messages) < self.max_seq_length
        ):  # self.config.env_engine.max_seq_length = 20480
            return True, "safe"
        else:
            return False, "token_overflow"

    def get_inc(self, text_frag_from, text_frag_to):
        """
        Get the incremental token array from text_frag_from to text_frag_to.
        """
        tokenizer_output = self.tokenizer(text_frag_from, return_tensors="pt", padding=False)
        tokenizer_input_ids = tokenizer_output["input_ids"][0].tolist()
        token_ids_acc = tokenizer_input_ids

        tokenizer_output = self.tokenizer(text_frag_to, return_tensors="pt", padding=False)
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

    def remove_last_context(self):
        if len(self.full_context) > 0:
            if self.full_context[-1].author != "llm":
                self.full_context.pop(-1)

    def remove_last_non_llm_msg(self, ext_msg_list: List[ExtendedMessage]):
        if len(ext_msg_list) > 0:
            if ext_msg_list[-1].author != "llm":
                ext_msg_list.pop(-1)
        return ext_msg_list

    @property
    def steps(self):
        return self.prepare_previous_context(mod="future")

    def prepare_next_llm_context(self):
        return self.prepare_previous_context(mod="future")

    def save_init_input(
        self,
        init_input_arr: list,
        add_nothink: bool = False,
        tools: List[dict] = [],
    ):
        """
        Save and process the initial input messages to the context.

        Args:
            init_input_arr (list): Array of initial input messages to be processed
                                  Each message should be a dict with 'role' and 'content'

        Note:
            - Initializes the context with the provided messages
            - Computes token arrays for each message
            - Validates that the context is empty before saving
        """
        # save basic
        assert len(self.full_context) == 0, "full_context should be empty when saving init input"
        for index, llm_msg in enumerate(init_input_arr):
            if (index == len(init_input_arr) - 1) and add_nothink:
                llm_msg["content"] += "\n/no_think"
            ext_msg = ExtendedMessage(
                author="initialization",
                role=llm_msg["role"],
                content=llm_msg["content"],
                token_generator="manual",
                tokenizer=self.tokenizer,
            )
            self.full_context += [ext_msg]

        # compute token array for each message
        token_ids_acc = []
        for llm_msg, ext_msg, index in zip(
            init_input_arr, self.full_context, range(len(init_input_arr))
        ):
            text_with_chat_template = astune_apply_chat_templat(
                tokenizer=self.tokenizer,
                conversation=init_input_arr[: (index + 1)],
                tools=tools,
                add_generation_prompt=False,
                tokenize=False,
            )
            tokenizer_output = self.tokenizer(
                text_with_chat_template, return_tensors="pt", padding=False
            )
            input_ids = tokenizer_output["input_ids"][0].tolist()
            # attention_mask = outputs["attention_mask"][0].tolist()
            input_id_increment = input_ids[
                len(token_ids_acc) :
            ]  # get the new tokens added in this step
            overlap_length = 0
            for i in range(len(token_ids_acc)):
                if (i < len(token_ids_acc)) and (input_ids[i] == token_ids_acc[i]):
                    overlap_length += 1
                else:
                    break
            ext_msg._info = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
            ext_msg.token_arr = input_id_increment
            token_ids_acc += input_ids
        return

    def save_llm_output(self, llm_output, input_msg_ref, auto_register_full_context=True):
        """
        Save the output from the LLM to the full context.

        Args:
            llm_output (dict): The output from the LLM containing 'role', 'content', and 'tokens'
            input_msg_ref: Reference to the input messages for token increment calculation
            out_of_full_context: Register in full_context or not

        Note:
            - Processes the LLM output and adds it to the conversation history
            - Handles token processing and generation prompt management
            - Ensures proper tokenization and context maintenance
        """
        # save basic
        assert isinstance(llm_output, dict)
        token_generator = "manual" if "tokens" in llm_output else "auto"
        ext_msg = ExtendedMessage(
            author="llm",
            role=llm_output["role"],
            content=llm_output["content"],
            token_generator=token_generator,
            tokenizer=self.tokenizer,
        )
        if auto_register_full_context:
            self.full_context += [ext_msg]
            if not self.already_mad_flag:
                if (
                    compute_string_madness(
                        completion=llm_output["content"],
                        checklist=self.config.astune.rollout.compute_madness_checklist,
                    )
                    < 0.0
                ):
                    self.already_mad_flag = True

        if token_generator == "manual":
            token_arr_method2, token_logprob_arr = self.get_token_inc_from_vllm_response(
                input_msg_ref, llm_output
            )
            ext_msg.token_arr = token_arr_method2
            ext_msg.token_logprob_arr = token_logprob_arr

        return ext_msg

    # generate token
    def get_token_inc_from_vllm_response(
        self, input_msg_ref, llm_output, tools: List[dict] = []
    ) -> Tuple[List[int], List[int]]:

        # completion_token_arr will contain generation_prompt header
        llm_output_role_content = {
            "role": llm_output["role"],
            "content": llm_output["content"],
        }
        if llm_output.get("tool_calls", None):
            llm_output_role_content.update({"tool_calls": llm_output.get("tool_calls", [])})

        completion_token_arr, _ = self.get_inc(
            astune_apply_chat_templat(
                tokenizer=self.tokenizer,
                conversation=input_msg_ref,
                tokenize=False, tools=tools, add_generation_prompt=False
            ),
            astune_apply_chat_templat(
                tokenizer=self.tokenizer,
                conversation=input_msg_ref + [llm_output_role_content],
                tokenize=False, tools=tools, add_generation_prompt=False
            ),
        )
        vllm_output_raw_token = [t.token_id for t in llm_output["tokens"]]
        vllm_output_raw_logprob = [t.logprob for t in llm_output["tokens"]]
        self.generated_token_cnt += len(vllm_output_raw_token)
        if not self.generation_prompt_token:
            self.generation_prompt_token = self.get_generation_prompt_token()
        final_token_arr, token_logprob_arr = replace_token_ids(
            place_holder=completion_token_arr,
            replace_with=vllm_output_raw_token,
            begin=self.generation_prompt_token,
            end=[self.tokenizer.eos_token_id],
            raw_logprob=vllm_output_raw_logprob,
        )
        return final_token_arr, token_logprob_arr

    def save_llm_output_do_not_register_full_context(self, llm_output, input_msg_ref):
        return BasicContextTracker.save_llm_output(
            self, llm_output, input_msg_ref, auto_register_full_context=False
        )

    def save_env_output(
        self,
        env_output: dict,
        input_msg_ref: Optional[List[dict]] = None,
        add_nothink=False,
    ):
        """
        Save and process environment output to the context.

        Args:
            env_output (dict): Environment output containing 'content'
            input_msg_ref (List[dict], optional): Reference messages for token calculation

        Note:
            - Clips environment output if it exceeds max_env_output_length
            - Processes the output as a user message in the conversation
            - Computes and stores token arrays for the environment response
        """
        assert isinstance(env_output, dict)
        if ("content" not in env_output) and ("error" in env_output):
            env_output["content"] = f"[Error from environment: {env_output['error']}]"
        elif ("content" not in env_output) or (not env_output["content"]):
            env_output["content"] = (
                "Warning: the environment does not provide any feedback, please provide valid inpu and try again."
            )
        if add_nothink:
            env_output["content"] += " /no_think"
        ext_msg = ExtendedMessage(
            author="env",
            role="user",
            content=env_output["content"],
            clip=True,
            clip_token_limit=self.max_env_output_length,
            token_generator="auto",
            tokenizer=self.tokenizer,
        )
        self.full_context += [ext_msg]
        return

    def to_role_content(self, ext_msg_array: List[ExtendedMessage]) -> List[dict]:
        result = []
        for ext_msg in ext_msg_array:
            d = {
                "role": ext_msg.role,
                "content": ext_msg.content_for_future,
            }
            if ext_msg.tool_calls:
                raise RuntimeError("Not expected, contact developer")
            result.append(d)
        return result

    def prepare_world_interaction(self) -> str:
        """
        Process the latest model content before environment interaction.

        Returns:
            str: Processed content, with code extracted from markdown code blocks if present
                 or the raw content if no code blocks are found

        Note:
            - Extracts Python code from markdown code blocks (```python```)
            - Returns the raw content if no valid code blocks are found
        """
        latest_content = self.full_context[-1].content
        return latest_content

    def filter_context_via_author(self, author: str) -> List[ExtendedMessage]:
        return copy.deepcopy([c for c in self.full_context if c.author == author])

    def filter_context_via_authors(self, authors: List[str]) -> List[ExtendedMessage]:
        return copy.deepcopy([c for c in self.full_context if c.author in authors])

    def filter_context_via_authors_with_limit(
        self, authors: List[str], limit: dict
    ) -> List[ExtendedMessage]:
        """
        limit = {
            "llm": "keep_last@2"
            "env": "keep_first@2"
        }
        """
        filtered_via_authors = copy.deepcopy([c for c in self.full_context if c.author in authors])
        for limit_author, limit_item in limit.items():
            limit_item_command, limit_item_value = limit_item.split("@")
            if limit_item_command == "keep_last":
                limit_item_value = int(limit_item_value)
                # remove all message whose author is `llm_author` except the last `limit_item_value` messages
                num_need_rm = (
                    len([c for c in filtered_via_authors if c.author == limit_author])
                    - limit_item_value
                )
                if num_need_rm > 0:
                    num_already_rm = 0
                    filtered_via_authors_new = []
                    for c in filtered_via_authors:
                        if c.author == limit_author:
                            num_already_rm += 1
                            if num_already_rm <= num_need_rm:
                                continue
                        filtered_via_authors_new += [c]
                    filtered_via_authors = filtered_via_authors_new

            elif limit_item_command == "keep_first":
                limit_item_value = int(limit_item_value)
                # remove all message whose author is `llm_author` except the first `limit_item_value` messages
                num_need_keep = (
                    len([c for c in filtered_via_authors if c.author == limit_author])
                    - limit_item_value
                )
                if num_need_keep > 0:
                    num_already_keep = 0
                    filtered_via_authors_new = []
                    for c in filtered_via_authors:
                        if c.author == limit_author:
                            num_already_keep += 1
                            if num_already_keep > limit_item_value:
                                continue
                        filtered_via_authors_new += [c]
                    filtered_via_authors = filtered_via_authors_new

            else:
                raise ValueError(
                    f"Unknown limit_item_command {limit_item_command} in filter_context_via_authors_with_limit"
                )
        return filtered_via_authors

    def group_tokenize(self):
        sample_arr = []
        ext_steps = self.full_context
        cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps, index=0, total_steps=1)
        sample = Sample(
            cmt_tokenized=cmt_tokenized,
            messages=self.to_role_content(ext_steps),
            config=self.config,
            task_batch_index=self.task_batch_index,
            task_tag=self.task_tag,
            task_id=self.task_id,
        )
        sample.truncate_output_ids()
        sample_arr += [sample]
        return sample_arr

    def group_tokenize_multi_group(self):
        sample_arr = []
        max_num_group = self.config.astune.rollout.multi_turn.max_sample_per_task
        for index, ext_steps in enumerate(self.grouped_steps):
            cmt_tokenized = self.tokenize_steps(
                ext_steps=ext_steps,
                index=index,
                total_steps=len(self.grouped_steps),
            )
            sample = Sample(
                cmt_tokenized=cmt_tokenized,
                messages=self.to_role_content(ext_steps),
                config=self.config,
                task_batch_index=self.task_batch_index,
                task_tag=self.task_tag,
                task_id=self.task_id,
            )
            sample_arr += [sample]

        if len(sample_arr) > max_num_group:
            print(f"Warning: allow {max_num_group} groups, but got {len(sample_arr)} groups")
            import random

            sample_arr = random.sample(sample_arr, max_num_group)  # preserve max_num_group groups

        return sample_arr

    def generate_log(self, task_id=None, global_step="NA"):
        task_id = self.task_id
        nested_items_print_buffer = {}
        ext_steps = self.full_context
        cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps, index=0, total_steps=1)
        text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]
        input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]
        loss_mask_color_arr = [
            "#09ABCF" if mask == 1 else "#D98510" for mask in cmt_tokenized["loss_mask"]
        ]
        buffer = {
            "text_arr": text_arr,
            "input_id_arr": input_id_arr,
            "loss_mask_color_arr": loss_mask_color_arr,
        }
        len_prompt_ids = len(cmt_tokenized["prompt_ids"])
        len_response_ids = len(cmt_tokenized["response_ids"])
        len_input_ids = len(cmt_tokenized["input_ids"])
        raw_reward = self.reward_structure.raw_reward
        step_reward = self.reward_structure.step_reward[0]
        try:
            step_advantage = self.reward_structure.step_advantage[0]
            step_advantage_simple = self.reward_structure.step_advantage_simple[0]
        except:
            step_advantage = 0.0
            step_advantage_simple = 0.0
        task_outcome = str(self.reward_structure.success_rate)
        selectors = [task_id, task_outcome]
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
                title=buffer["text_arr"],  # mouse hover text
                count=buffer["input_id_arr"],  # h highlight text
                color=buffer["loss_mask_color_arr"],  # color
            ),
        )
        print_nested(
            nested_items_print_buffer,
            main_content="This is the main content of the nested JSON",
            header=f"[{global_step}] Task {task_id} (Reward {float(step_reward):.3f})",
            mod="rollout",
            narrow=False,
        )

    def process_reward(self, reward_structure: Reward):
        self.reward_structure = reward_structure
        ext_steps = self.full_context
        # linear mode has only one trajectory
        self.reward_structure.step_reward = [
            self.compute_step_level_reward(ext_steps=ext_steps, index=0, total_steps=1)
        ]

    def ensure_terminate_rollout_stage(self):
        """Nothing need to be done for basic linear cmt at `ensure_terminate_rollout_stage`"""
        pass

    def compute_step_level_reward(
        self, ext_steps: List[ExtendedMessage], index: int, total_steps: int
    ) -> float:
        assert self.reward_structure is not None

        # --------------- global level reward ---------------
        global_reward = self.reward_structure.raw_reward
        gamma = self.config.astune.rollout.gamma
        step_reward_base = global_reward * (gamma ** (total_steps - index - 1))

        # --------------- compute step level reward ---------------
        step_reward = step_reward_base
        if self.already_mad_flag:
            step_reward = self.config.astune.rollout.agent_madness_reward
            self.reward_structure.madness = -1.0

        return step_reward

    def tokenize_steps(
        self, ext_steps: List[ExtendedMessage], index: int, total_steps: int
    ) -> dict:
        """
        Create an Experience object from the current conversation context.

        Returns:
            Experience: An object containing processed conversation data for model training

        Note:
            - Removes the last user message as it's not required in casual model training
            - Processes input IDs, attention masks, and loss masks
            - Separates prompt and response components
            - Handles position IDs and reward scores
            - Truncates output IDs as needed
        """
        from verl.utils.model import compute_position_id_with_mask

        ext_steps = self.remove_last_non_llm_msg(ext_steps)

        # check reward structure
        self.reward_structure: Reward  # type: ignore
        assert (
            self.reward_structure.step_reward is not None
        ), "must call `process_reward` before tokenize_steps"
        assert len(self.reward_structure.step_reward) == total_steps

        # mapping
        input_ids = []
        input_logprobs = []
        attention_mask = []
        loss_mask = []
        split_prompt_reponse_index = -1
        split_point_message_left_index = -1
        input_ids_len = []

        # cat all messages
        for i, ext_msg in enumerate(ext_steps):
            # find split index, this have to be done before input_ids += ext_msg.token_arr
            if (split_prompt_reponse_index == -1) and (ext_msg.need_training):
                split_prompt_reponse_index = len(input_ids)
                split_point_message_left_index = i - 1
                assert (
                    split_point_message_left_index >= 0
                ), "There should be at least one message before the first training message"
                assert split_prompt_reponse_index == input_ids_len[split_point_message_left_index]
                assert (
                    ext_msg.author == "llm"
                ), "The first message after initialization should be from LLM, not from env or user"

            # cat all tokens
            input_ids += ext_msg.token_arr
            if len(ext_msg.token_logprob_arr) == 0:
                input_logprobs += [ext_msg.invalid_log_prob_value] * len(ext_msg.token_arr)
            else:
                input_logprobs += ext_msg.token_logprob_arr
            input_ids_len += [len(input_ids)]
            attention_mask += [1] * len(ext_msg.token_arr)
            loss_mask += ext_msg.get_loss_mask(blackout_token_combo=self.blackout_token_combo)

            if split_prompt_reponse_index == -1:
                # should we begin split point early?
                if input_ids_len[-1] > self.config.astune.data.max_prompt_length:
                    message_dict = self.to_role_content(ext_steps)
                    logger.warning(
                        f"Input ids exceeded max_prompt_length before encountering any training message! trying to fix..."
                    )
                    logger.bind(exception=True).exception(
                        f"Input ids exceeded max_prompt_length before encountering any training message! trying to fix...\n\n"
                        + str(message_dict)
                    )
                    assert (
                        i >= 1
                    ), "There should be at least one message before exceeding max_prompt_length"
                    assert (
                        input_ids_len[-2] <= self.config.astune.data.max_prompt_length
                    ), "The previous message should be within max_prompt_length, something is wrong"
                    split_point_message_left_index = i - 1
                    assert split_point_message_left_index == (len(input_ids_len) - 2), "what?"
                    split_prompt_reponse_index = input_ids_len[split_point_message_left_index]

        # check
        assert len(ext_steps) == len(
            input_ids_len
        ), "length of ext_steps and input_ids_len should be equal"
        assert (
            split_prompt_reponse_index != -1
        ), "split_prompt_reponse_index should not be -1, at least one message should be in the context"
        position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()

        # sperate prompt and response
        prompt_ids = input_ids[:split_prompt_reponse_index]
        prompt_attention_mask = attention_mask[:split_prompt_reponse_index]
        prompt_position_ids = position_ids[:split_prompt_reponse_index]
        prompt_loss_mask = loss_mask[:split_prompt_reponse_index]
        prompt_logprobs = input_logprobs[:split_prompt_reponse_index]

        response_ids = input_ids[split_prompt_reponse_index:]
        response_attention_mask = attention_mask[split_prompt_reponse_index:]
        response_position_ids = position_ids[split_prompt_reponse_index:]
        response_loss_mask = loss_mask[split_prompt_reponse_index:]
        response_logprobs = input_logprobs[split_prompt_reponse_index:]

        cmt_tokenized = {}
        cmt_tokenized["input_ids"] = input_ids
        cmt_tokenized["prompt_ids"] = prompt_ids
        cmt_tokenized["response_ids"] = response_ids
        cmt_tokenized["attention_mask"] = attention_mask
        cmt_tokenized["logprobs"] = input_logprobs
        cmt_tokenized["prompt_attention_mask"] = prompt_attention_mask
        cmt_tokenized["response_attention_mask"] = response_attention_mask
        cmt_tokenized["loss_mask"] = loss_mask
        cmt_tokenized["prompt_loss_mask"] = prompt_loss_mask
        cmt_tokenized["response_loss_mask"] = response_loss_mask
        cmt_tokenized["position_ids"] = position_ids
        cmt_tokenized["prompt_position_ids"] = prompt_position_ids
        cmt_tokenized["response_position_ids"] = response_position_ids
        cmt_tokenized["step_reward"] = self.reward_structure.step_reward[index]
        cmt_tokenized["response_logprobs"] = response_logprobs
        cmt_tokenized["prompt_logprobs"] = prompt_logprobs
        try:
            cmt_tokenized["reference_advantage"] = self.reward_structure.step_advantage[index]
        except:
            cmt_tokenized["reference_advantage"] = 0

        return cmt_tokenized

    @staticmethod
    def compute_reference_advantage(cmt_array: List):
        import numpy as np

        task2cmt = defaultdict(list)
        for cmt in cmt_array:
            task2cmt[cmt.task_id] += [cmt]

        for task_id, cmt_list in task2cmt.items():
            cmt_reward = []

            # compute in-group mean and std
            for cmt in cmt_list:
                cmt_reward += [np.mean(cmt.reward_structure.step_reward)]

            if len(cmt_reward) == 1:
                reward_mean = 0.0
                reward_std = 1.0
            else:
                reward_mean = float(np.mean(cmt_reward))
                reward_std = float(np.std(cmt_reward, ddof=1))
                if reward_std < 0.01:
                    reward_std = 0.01

            # logger.bind(exception=True).info(f"task id {task_id}")
            # logger.bind(exception=True).info(f"reward_mean {reward_mean}, reward_std {reward_std}, cmt_reward {cmt_reward}")
            # compute advantage
            for cmt in cmt_list:
                cmt.reward_structure.step_advantage = []
                for i in range(len(cmt.reward_structure.step_reward)):
                    cmt.reward_structure.step_advantage += [
                        (cmt.reward_structure.step_reward[i] - reward_mean) / (reward_std + 1e-6)
                    ]
                # logger.bind(exception=True).info(f"step reward {cmt.reward_structure.step_reward}")
                # logger.bind(exception=True).info(f"step advantage {cmt.reward_structure.step_advantage}")

        # compute simple advantage (uneven rollout sample count)
        for task_id, cmt_list in task2cmt.items():
            cmt_reward = []
            for cmt in cmt_list:
                cmt_reward.extend(cmt.reward_structure.step_reward)
            if len(cmt_reward) == 1:
                reward_mean = 0.0
                reward_std = 1.0
            else:
                reward_mean = float(np.mean(cmt_reward))
                reward_std = float(np.std(cmt_reward, ddof=1))
                # if reward_std < 0.01:
                #     reward_std = 0.01
            for cmt in cmt_list:
                cmt.reward_structure.step_advantage_simple = []
                for i in range(len(cmt.reward_structure.step_reward)):
                    cmt.reward_structure.step_advantage_simple += [
                        (cmt.reward_structure.step_reward[i] - reward_mean) / (reward_std + 1e-6)
                    ]

        return


    def get_generation_prompt_token(self):
        dummy_msg = [{"role": "assistant", "content": "dummy text"}]
        self.generation_prompt_token, _ = self.get_inc(
            astune_apply_chat_templat(
                tokenizer=self.tokenizer,
                conversation=dummy_msg,
                tools=[],
                add_generation_prompt=False,
                tokenize=False,
            ),
            astune_apply_chat_templat(
                tokenizer=self.tokenizer,
                conversation=dummy_msg,
                tools=[],
                add_generation_prompt=True,
                tokenize=False,
            ),
        )
        return self.generation_prompt_token
