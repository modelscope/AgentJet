import copy
from collections import defaultdict
from typing import List, Optional, Tuple

import torch

from astuner.context_tracker.tracker_base_attr import (
    BaseTracker,
    ExtendedMessage,
    replace_token_ids,
)
from astuner.schema.trajectory import Reward, Sample
from astuner.utils.compute_madness import compute_string_madness
from astuner.utils.tokenizer import astune_apply_chat_template


class BaseContextTracker(BaseTracker):
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
                first_message=(index == 0),
            )
            self.full_context += [ext_msg]

        # compute token array for each message
        token_ids_acc = []
        for llm_msg, ext_msg, index in zip(
            init_input_arr, self.full_context, range(len(init_input_arr))
        ):
            text_with_chat_template = astune_apply_chat_template(
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
                        checklist=self.config.astuner.rollout.compute_madness_checklist,
                    )
                    < 0.0
                ):
                    self.already_mad_flag = True

        if token_generator == "manual":
            (
                precise_manual_token,
                token_logprob_arr,
                loss_mask,
                lack_normal_eos,
            ) = self.get_token_inc_from_vllm_response(input_msg_ref, llm_output)
            ext_msg.token_arr = precise_manual_token
            ext_msg.token_logprob_arr = token_logprob_arr
            ext_msg.lack_normal_eos = lack_normal_eos
            ext_msg.manual_loss_mask_override = loss_mask

        return ext_msg

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
        return input_id_increment, msg

    # generate token
    def get_token_inc_from_vllm_response(
        self, input_msg_ref, llm_output, tools: List[dict] = []
    ) -> Tuple[List[int], List[int], List[int], bool]:
        llm_output_role_content = {
            "role": llm_output["role"],
            "content": llm_output["content"],
        }
        if llm_output.get("tool_calls", None):
            llm_output_role_content.update({"tool_calls": llm_output.get("tool_calls", [])})

        # completion_token_arr will contain generation_prompt header
        completion_token_arr, _ = self.get_inc(
            astune_apply_chat_template(
                tokenizer=self.tokenizer,
                conversation=input_msg_ref,
                tokenize=False,
                tools=tools,
                add_generation_prompt=False,
            ),
            astune_apply_chat_template(
                tokenizer=self.tokenizer,
                conversation=input_msg_ref + [llm_output_role_content],
                tokenize=False,
                tools=tools,
                add_generation_prompt=False,
            ),
        )
        vllm_output_raw_token = [t.token_id for t in llm_output["tokens"]]
        vllm_output_raw_logprob = [t.logprob for t in llm_output["tokens"]]
        self.generated_token_cnt += len(vllm_output_raw_token)
        if not self.generation_prompt_token:
            self.generation_prompt_token = self.get_generation_prompt_token()
        final_token_arr, token_logprob_arr, loss_mask, lack_normal_eos = replace_token_ids(
            token_container=completion_token_arr,
            precise_token=vllm_output_raw_token,
            precise_logprob=vllm_output_raw_logprob,
            begin_ids=self.generation_prompt_token,
            end_ids=[self.tokenizer.eos_token_id],
        )
        return final_token_arr, token_logprob_arr, loss_mask, lack_normal_eos

    def save_llm_output_do_not_register_full_context(self, llm_output, input_msg_ref):
        return BaseContextTracker.save_llm_output(
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
            env_output[
                "content"
            ] = "Warning: the environment does not provide any feedback, please provide valid inpu and try again."
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
        tracker_tokenized = self.tokenize_steps(ext_steps=ext_steps, index=0, total_steps=1)
        sample = Sample(
            tracker_tokenized=tracker_tokenized,
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
        max_num_group = self.config.astuner.rollout.multi_turn.max_sample_per_task
        for index, ext_steps in enumerate(self.grouped_steps):
            tracker_tokenized = self.tokenize_steps(
                ext_steps=ext_steps,
                index=index,
                total_steps=len(self.grouped_steps),
            )
            sample = Sample(
                tracker_tokenized=tracker_tokenized,
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

    def compute_step_level_reward(
        self, ext_steps: List[ExtendedMessage], index: int, total_steps: int
    ) -> float:
        # TODO: support multi-step reward
        assert self.reward_structure is not None

        # --------------- global level reward ---------------
        global_reward = self.reward_structure.raw_reward
        gamma = self.config.astuner.rollout.gamma
        step_reward_base = global_reward * (gamma ** (total_steps - index - 1))
        assert (
            gamma == 1.0
        ), "Currently only support gamma == 1.0, we'll support multi-step reward in the future"

        # --------------- compute step level reward ---------------
        step_reward = step_reward_base  # reward scalar
        if self.already_mad_flag:
            step_reward = self.config.astuner.rollout.agent_madness_reward
            self.reward_structure.madness = -1.0

        return step_reward

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
            self.reward_structure.step_reward_arr is not None
        ), "must call `process_reward` before tokenize_steps"
        assert len(self.reward_structure.step_reward_arr) == total_steps

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

        # if [prompt_token | response_token] is splited at a place where loss_mask == 0,
        # move the split index forward
        MAX_FORWARD_STEPS = 100
        for i in range(MAX_FORWARD_STEPS):
            if loss_mask[split_prompt_reponse_index] == 0:
                split_prompt_reponse_index += 1
            else:
                break

        # no matter what, the split index should not exceed max prompt length
        # make sure that the prompt length does not exceed `config.astuner.data.max_prompt_length`
        if split_prompt_reponse_index > self.config.astuner.data.max_prompt_length:
            split_prompt_reponse_index = self.config.astuner.data.max_prompt_length

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

        tracker_tokenized = {}
        tracker_tokenized["input_ids"] = input_ids
        tracker_tokenized["prompt_ids"] = prompt_ids
        tracker_tokenized["response_ids"] = response_ids
        tracker_tokenized["attention_mask"] = attention_mask
        tracker_tokenized["logprobs"] = input_logprobs
        tracker_tokenized["prompt_attention_mask"] = prompt_attention_mask
        tracker_tokenized["response_attention_mask"] = response_attention_mask
        tracker_tokenized["loss_mask"] = loss_mask
        tracker_tokenized["prompt_loss_mask"] = prompt_loss_mask
        tracker_tokenized["response_loss_mask"] = response_loss_mask
        tracker_tokenized["position_ids"] = position_ids
        tracker_tokenized["prompt_position_ids"] = prompt_position_ids
        tracker_tokenized["response_position_ids"] = response_position_ids
        tracker_tokenized["response_logprobs"] = response_logprobs
        tracker_tokenized["prompt_logprobs"] = prompt_logprobs

        # distribute reward
        tracker_tokenized["step_reward"] = self.reward_structure.step_reward_arr[index]
        try:
            tracker_tokenized["reference_advantage"] = self.reward_structure.step_advantage[index]
        except Exception:
            tracker_tokenized["reference_advantage"] = 0

        return tracker_tokenized

    @staticmethod
    def compute_reference_advantage(tracker_array: List):
        import numpy as np

        task2tracker = defaultdict(list)
        for tracker in tracker_array:
            task2tracker[tracker.task_id] += [tracker]

        # compute group normalized step_advantage (just for logging purpose)
        for task_id, tracker_list in task2tracker.items():
            tracker_reward = []

            # compute in-group mean and std
            for tracker in tracker_list:
                tracker_reward += [np.mean(tracker.reward_structure.step_reward_arr)]

            if len(tracker_reward) == 1:
                reward_mean = 0.0
                reward_std = 1.0
            else:
                reward_mean = float(np.mean(tracker_reward))
                reward_std = float(np.std(tracker_reward, ddof=1))
                if reward_std < 0.01:
                    reward_std = 0.01

            # compute advantage
            for tracker in tracker_list:
                tracker.reward_structure.step_advantage = []
                for i in range(len(tracker.reward_structure.step_reward_arr)):
                    tracker.reward_structure.step_advantage += [
                        (tracker.reward_structure.step_reward_arr[i] - reward_mean)
                        / (reward_std + 1e-6)
                    ]

        # compute simple advantage (uneven rollout sample count) (just for logging purpose)
        for task_id, tracker_list in task2tracker.items():
            tracker_reward = []
            for tracker in tracker_list:
                tracker_reward.extend(tracker.reward_structure.step_reward_arr)
            if len(tracker_reward) == 1:
                reward_mean = 0.0
                reward_std = 1.0
            else:
                reward_mean = float(np.mean(tracker_reward))
                reward_std = float(np.std(tracker_reward, ddof=1))
            for tracker in tracker_list:
                tracker.reward_structure.step_advantage_simple = []
                for i in range(len(tracker.reward_structure.step_reward_arr)):
                    tracker.reward_structure.step_advantage_simple += [
                        (tracker.reward_structure.step_reward_arr[i] - reward_mean)
                        / (reward_std + 1e-6)
                    ]
        return

    def get_generation_prompt_token(self):
        dummy_msg = [{"role": "assistant", "content": "dummy text"}]
        self.generation_prompt_token, _ = self.get_inc(
            astune_apply_chat_template(
                tokenizer=self.tokenizer,
                conversation=dummy_msg,
                tools=[],
                add_generation_prompt=False,
                tokenize=False,
            ),
            astune_apply_chat_template(
                tokenizer=self.tokenizer,
                conversation=dummy_msg,
                tools=[],
                add_generation_prompt=True,
                tokenize=False,
            ),
        )
        self.generation_prompt = self.tokenizer.decode(self.generation_prompt_token)
        return self.generation_prompt_token
