from loguru import logger

from agentscope.model import DashScopeChatModel
from astune.schema.trajectory import Reward
from transformers.tokenization_utils import PreTrainedTokenizer
from astune.context_manager.cmt_linear import CMTLinear, ExtendedMessage, replace_token_ids
from astune.utils.color_hsl import adjust_color_hsl
from astune.utils.compute_madness import compute_string_madness
from astune.schema.extended_msg import INVALID_LOG_PROB_VALUE
from astune.context_manager.agentscope_cm.timeline_merging import can_merge_steps

from typing import Any, List, Tuple, Union
from beast_logger import print_nested, print_listofdict, NestedJsonItem, SeqItem

class ASTuneContextTracking(CMTLinear):

    def __init__(
            self,
            llm_chat_fn,
            tokenizer:PreTrainedTokenizer,
            config,
            env_step_fn,
            should_interrupt_fn,
            generated_token_callback_fn,
            **kwargs
        ):
        super().__init__(config, tokenizer)
        self.task_batch_index = kwargs.pop("task_batch_index")
        self.task_tag = kwargs.pop("task_tag")
        self.task_id = kwargs.pop("task_id")
        self.dscm_ref = DashScopeChatModel(**kwargs)
        self.full_context: List[ExtendedMessage] = []
        self.llm_chat_fn = llm_chat_fn
        self.tokenizer = tokenizer
        self.stream = False
        self.config = config
        self.env_step_fn = env_step_fn
        self.should_interrupt_fn = should_interrupt_fn
        self.generated_token_callback_fn = generated_token_callback_fn
        self.context_overflow = False
        self.model_name = kwargs['model_name']
        self.output_kwargs = {}
        self.input_kwargs = {}

    def process_reward(self, reward_structure: Reward):
        self.reward_structure = reward_structure
        ext_steps = self.full_context
        # # linear mode has only one trajectory
        # self.reward_structure.step_reward = [
        #     self.compute_step_level_reward(ext_steps=ext_steps, index=0, total_steps=1)
        # ]
        # print('warning: debugging')
        self.reward_structure.step_reward = [
            self.compute_step_level_reward(ext_steps=ext_steps, index=i, total_steps=len(self.grouped_steps)) for i in range(len(self.grouped_steps))
        ]


    def generate_log(self, task_id = None, global_step="NA"):
        task_id = self.task_id
        nested_items_print_buffer = {}
        step_reward = 0.0

        for index, ext_steps in enumerate(self.grouped_steps):
            cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps, index=index, total_steps=len(self.grouped_steps))
            text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]
            input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]
            # loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in cmt_tokenized["loss_mask"]]
            logprobs = [INVALID_LOG_PROB_VALUE] * len(cmt_tokenized["prompt_ids"]) + cmt_tokenized["response_logprobs"]
            # Create adjusted color array
            loss_mask_color_abl_arr = [
                adjust_color_hsl("#09ABCF", logprob) if mask == 1
                else adjust_color_hsl("#D98510", logprob)
                for mask, logprob in zip(cmt_tokenized["loss_mask"], logprobs)
            ]
            logprob_text_arr = [f"{logprob:.4f}" if logprob != INVALID_LOG_PROB_VALUE else "N/A" for logprob in logprobs]

            buffer = {
                "text_arr": text_arr,
                "logprob_arr": logprob_text_arr,
                "input_id_arr": input_id_arr,
                "loss_mask_color_arr": loss_mask_color_abl_arr,
            }
            raw_reward = self.reward_structure.raw_reward
            step_reward:float = self.reward_structure.step_reward[index]
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
            assert len_prompt_ids + len_response_ids == len_input_ids, "len_prompt_ids + len_response_ids should equal to len_input_ids"
            nested_items_print_buffer[f".".join(selectors)] = NestedJsonItem(
                item_id=f"item",    # type: ignore
                outcome=task_outcome,   # type: ignore
                len_prompt_ids=len_prompt_ids,  # type: ignore
                len_response_ids=len_response_ids,  # type: ignore
                len_input_ids=len_input_ids,    # type: ignore
                raw_reward=f"{float(raw_reward):.3f}",  # type: ignore
                step_reward=f"{float(step_reward):.3f}",    # type: ignore
                step_advantage=f"{float(step_advantage):.3f}",  # type: ignore
                step_advantage_simple=f"{float(step_advantage_simple):.3f}",    # type: ignore
                content=SeqItem(
                    text = buffer['text_arr'],  # 文本
                    title = buffer['logprob_arr'], # 鼠标悬浮文本
                    count = buffer['input_id_arr'], # 高亮文本 # type: ignore
                    color = buffer['loss_mask_color_arr']   # 颜色
                )
            )

        print_nested(nested_items_print_buffer,
            main_content="This is the main content of the nested JSON",
            header=f"[{global_step}] Task {task_id} (Reward {float(step_reward):.3f})", # type: ignore
            mod="rollout",
            narrow=False,
            attach="copy this"  # type: ignore
        )

    def group_merge(self):

        def toggle_author(source_step: List[ExtendedMessage], target_step: List[ExtendedMessage]) -> List[ExtendedMessage]:
            # if any message in `target_step` is author == 'llm', but same-index message in `source_step` is author != 'llm'
            # change source_step's message author to 'llm'
            for i in range(len(target_step)):
                if target_step[i].author == 'llm' and source_step[i].author != 'llm':
                    source_step[i].author = target_step[i].author
                    source_step[i].token_arr = target_step[i].token_arr
                    source_step[i].token_logprob_arr = target_step[i].token_logprob_arr
                    assert source_step[i].need_training
            return source_step

        absorbed_step_indices = []
        reversed_grouped_steps = list(reversed(self.grouped_steps))
        for i in range(len(reversed_grouped_steps)):
            if i in absorbed_step_indices:
                continue
            # check whether [i, len(reversed_grouped_steps)-1] can be merged
            for j in range(i+1, len(reversed_grouped_steps)):
                if j in absorbed_step_indices:
                    continue
                source_step = reversed_grouped_steps[i]
                target_step = reversed_grouped_steps[j]
                if can_merge_steps(source_step, target_step):
                    source_step = toggle_author(source_step, target_step)
                    reversed_grouped_steps[i] = source_step
                    absorbed_step_indices += [j]

        # reverse back and exclude absorbed steps
        reversed_grouped_steps_clean = []
        for i in range(len(reversed_grouped_steps)):
            if i not in absorbed_step_indices:
                reversed_grouped_steps_clean.append(reversed_grouped_steps[i])
        self.grouped_steps = list(reversed(reversed_grouped_steps_clean))

        return self.grouped_steps

    def group_tokenize(self):
        return self.group_tokenize_multi_group()

    def get_inc(self, text_frag_from, text_frag_to):
        """
        Get the incremental token array from text_frag_from to text_frag_to.
        """
        tokenizer_output = self.tokenizer(text_frag_from, return_tensors="pt", padding=False)
        tokenizer_input_ids = tokenizer_output["input_ids"][0].tolist() # type: ignore
        token_ids_acc = tokenizer_input_ids

        tokenizer_output = self.tokenizer(text_frag_to, return_tensors="pt", padding=False)
        input_ids = tokenizer_output["input_ids"][0].tolist() # type: ignore
        input_id_increment = input_ids[len(token_ids_acc):]  # get the new tokens added in this step
        overlap_length = 0
        for i in range(len(token_ids_acc)):
            if i < len(token_ids_acc) and input_ids[i] == token_ids_acc[i]: overlap_length += 1
            else: break
        msg = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
        # print(msg)
        return input_id_increment, msg

    def get_context_token_num_and_safety(self, ext_messages: List[ExtendedMessage], tools: List = []) -> Tuple[bool, int]:   # type: ignore
        dict_messages = self.to_role_content(ext_messages)

        prompt_text = self.tokenizer.apply_chat_template(dict_messages, tokenize=False, add_generation_prompt=True, tools=tools)
        length = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0]) # type: ignore
        max_response_length = self.config.astune.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.astune.rollout.max_model_len
        max_seq_length: int = max_model_len - max_response_length

        if length < max_seq_length:
            ret = [True, length]
        else:
            ret = [False, length]
        return tuple(ret)

    def check_context_token_num_safe(self, messages: List, tools: List = []) -> Tuple[bool, str]:
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
        length = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0]) # type: ignore
        max_response_length = self.config.astune.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.astune.rollout.max_model_len
        max_seq_length: int = max_model_len - max_response_length
        if self.should_interrupt_fn():
            ret = [False, "externally_interrupted"]
        if self.already_mad_flag and self.config.astune.rollout.agent_madness_termination:
            ret = [False, "already_mad"]
        if length < max_seq_length:
            ret = [True, f"safe[{length} < {max_model_len} - {max_response_length}]"]
        else:
            ret = [False, "token_overflow"]
        return tuple(ret)

    def to_role_content(self, ext_msg_array: List[ExtendedMessage]) -> List:
        result = []
        for ext_msg in ext_msg_array:
            d = {
                "role": ext_msg.role,
                "content": ext_msg.content_for_future,
            }
            if ext_msg.tool_calls:
                d.update({
                    "tool_calls": ext_msg.tool_calls
                })
            result.append(d)
        return result

    def apply_chat_template_for_ext_messages(self, ext_messages: List[ExtendedMessage], tools: List = []) -> str:
        dict_messages = self.to_role_content(ext_messages)
        prompt_text = self.tokenizer.apply_chat_template(dict_messages, tokenize=False, add_generation_prompt=True, tools=tools)
        return prompt_text  # type: ignore