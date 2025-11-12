import copy
import importlib
from loguru import logger
from datetime import datetime
from astune.schema.trajectory import Reward, Trajectory
from astune.context_manager.agentflow_cm.cmt_linear import CMTLinear, ExtendedMessage
from agentscope.model import DashScopeChatModel, ChatResponse
from agentscope.message import TextBlock, ToolUseBlock, ThinkingBlock
from typing import Dict, Tuple
from typing import Any, AsyncGenerator, Generator, Union, TYPE_CHECKING, List, Literal, Type
from pydantic import BaseModel
from astune.context_manager.agentflow_cm.cmt_linear import replace_token_ids, CMTLinear
from astune.schema.trajectory import Sample, Reward
from beast_logger import register_logger, print_dict, print_nested, NestedJsonItem, SeqItem
from astune.utils.compute_madness import compute_string_madness
from agentscope._utils._common import _json_loads_with_repair, _create_tool_from_base_model
from astune.context_manager.agentflow_cm.cmt_base_attr import INVALID_LOG_PROB_VALUE
from astune.utils.color_hsl import adjust_color_hsl


class ASTuneContextTemplate(CMTLinear):

    def __init__(self, llm_chat_fn, tokenizer, config, env_step_fn, should_interrupt_fn, generated_token_callback_fn, **kwargs):
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
        # # lienar 模式只有一条轨迹
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
            # 创建调整后的颜色数组
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
                    count = buffer['input_id_arr'], # 高亮文本
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
        def can_merge_steps(source_step: List[ExtendedMessage], target_step: List[ExtendedMessage]) -> bool:
            # if `source_step` has more messages than `target_step`
            # and if `source_step` and `target_step` share same token_arr in [0:len(target_step)]
            # even if the authors are different, we can still merge them
            can_merge = False
            # compare_level = 'token' # 严格按照token对比
            compare_level = 'text' # 对比文本，这样子会导致有些token不一样但是文本一样的情况也能merge，更宽松一些，收益很大，代价未知
            if len(source_step) >= len(target_step):
                all_msg_match = True
                for i in range(len(target_step)):
                    if compare_level == 'text':
                        same = source_step[i].content_for_future == target_step[i].content_for_future
                    elif compare_level == 'token':
                        same = source_step[i].token_arr == target_step[i].token_arr
                    else:
                        raise NotImplementedError
                    if not same:
                        all_msg_match = False
                        break
                if all_msg_match:
                    can_merge = True
            return can_merge

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
        tokenizer_input_ids = tokenizer_output["input_ids"][0].tolist()
        token_ids_acc = tokenizer_input_ids

        tokenizer_output = self.tokenizer(text_frag_to, return_tensors="pt", padding=False)
        input_ids = tokenizer_output["input_ids"][0].tolist()
        input_id_increment = input_ids[len(token_ids_acc):]  # get the new tokens added in this step
        overlap_length = 0
        for i in range(len(token_ids_acc)):
            if i < len(token_ids_acc) and input_ids[i] == token_ids_acc[i]: overlap_length += 1
            else: break
        msg = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
        # print(msg)
        return input_id_increment, msg

    def check_context_token_num_safe(self, messages: List[dict]) -> Tuple[bool, str]:
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        length = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])
        max_response_length = self.config.astune.rollout.max_response_length_in_one_turn
        max_model_len: int = self.config.astune.rollout.max_model_len
        self.max_seq_length: int = max_model_len - max_response_length
        if self.should_interrupt_fn():
            return False, "externally_interrupted"
        if self.already_mad_flag and self.config.astune.rollout.agent_madness_termination:
            return False, "already_mad"
        if length < self.max_seq_length:
            return True, f"safe[{length} < {max_model_len} - {max_response_length}]"
        else:
            return False, "token_overflow"


class ASTuneLmProxy(ASTuneContextTemplate):

    async def execute_model_proxy(self, messages: List[dict], tools: List[dict]=[], tool_choice: str = "auto", structured_model=None, **kwargs) -> dict:
        # load messages into `self.full_context`
        self.full_context = []

        consider_roles = ['user', 'assistant', 'system', 'tool']
        disable_toolcalls = self.config.astune.rollout.agentscope_disable_toolcalls
        if disable_toolcalls:
            consider_roles.remove('tool')

        for i, msg in enumerate(messages):
            if (disable_toolcalls) and (not isinstance(msg['content'], str)):
                continue
            if msg['role'] not in consider_roles:
                continue
            if not isinstance(msg['content'], str):
                author = 'env'
                msg['content'] = str(msg['content'])    # TODO: better handling for non-str content
            if msg['role'] == 'system':
                author = 'initialization'
            if msg['role'] == 'tool':
                author = 'env'
            else:
                author = 'env'

            is_last_message = (len(messages) == i+1)
            if is_last_message and (not disable_toolcalls):
                _tools = tools
            else:
                _tools = []

            self.full_context += [
                ExtendedMessage(
                    author=author,
                    role=msg['role'],
                    content=msg['content'],
                    tokenizer=self.tokenizer,
                    tools=_tools,
                    token_generator="auto",
                )
            ]

        # execute llm policy
        messages = self.to_role_content(self.full_context)
        # 4. ⚠️ check token overflow
        is_safe, info = self.check_context_token_num_safe(messages)
        custom_sampling_params = {}
        if not is_safe:
            logger.warning(f"[{info}] detected. Current token count exceeds the limit.")
            self.context_overflow = True
            return ChatResponse(
                content = [{'type': 'text', 'text': 'astune_proxy:[context_overflow]'}]
            )

        llm_output = self.llm_chat_fn(messages, custom_sampling_params)

        # compute_string_madness
        if not self.already_mad_flag:
            if compute_string_madness(completion=llm_output['content'], checklist=self.config.astune.rollout.compute_madness_checklist) < 0.0:
                self.already_mad_flag = True

        # dummy response for now
        token_generator = "manual"
        if llm_output.get("tool_calls", None) is not None:
            tool_calls = llm_output["tool_calls"]
        else:
            tool_calls = []

        llm_ext_msg = ExtendedMessage(
            author="llm",
            role="assistant",
            content=llm_output['content'],
            token_generator=token_generator,
            tool_calls=tool_calls,
            tokenizer=self.tokenizer,
        )

        if token_generator == "manual":
            input_msg_ref = copy.deepcopy(messages)
            token_arr_method2, token_logprob_arr = self.get_token_inc_from_vllm_response(input_msg_ref, llm_output)
            assert len(token_arr_method2) <= self.config.astune.rollout.max_response_length_in_one_turn, f"Generated token length {len(token_arr_method2)} exceeds max_response_len {self.config.astune.rollout.max_response_length_in_one_turn}"
            llm_ext_msg.token_arr = token_arr_method2
            llm_ext_msg.token_logprob_arr = token_logprob_arr
            self.generated_token_callback_fn(llm_ext_msg.token_arr)

        # take snapshot of current timeline
        if is_safe:
            self.full_context += [
                llm_ext_msg
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                self.to_role_content(self.full_context),    # todo
                tokenize=False,
                add_generation_prompt=True
            )
            length = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])
            if length >= self.config.astune.rollout.max_model_len:
                raise RuntimeError(f"Unexpected token overflow after adding LLM response. Full context length {length}, before gen info {info}, generated token length {len(llm_ext_msg.token_arr)}")
            self.grouped_steps += [copy.deepcopy(self.full_context)]
        # return response
        return await self._parse_dashscope_generation_response(llm_output, structured_model=structured_model)

    async def _parse_dashscope_generation_response(
        self,
        message,
        structured_model: Type[BaseModel] | None = None,
    ) -> ChatResponse:


        content_blocks: List[TextBlock | ToolUseBlock] = []
        content = message.get("content")
        metadata: dict | None = None

        if content not in [
            None,
            "",
            [],
        ]:
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        content_blocks.append(
                            TextBlock(
                                type="text",
                                text=item["text"],
                            ),
                        )
            else:
                content_blocks.append(
                    TextBlock(
                        type="text",
                        text=content,
                    ),
                )

        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                input_ = _json_loads_with_repair(
                    tool_call["function"].get(
                        "arguments",
                        "{}",
                    )
                    or "{}",
                )
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use",
                        name=tool_call["function"]["name"],
                        input=input_,
                        id=tool_call["id"],
                    ),
                )

                if structured_model:
                    metadata = input_


        parsed_response = ChatResponse(
            content=content_blocks,
            metadata=metadata,
        )

        return parsed_response


class ASTuneProxy(ASTuneLmProxy):
    """
    A proxy class that bridge:
    - environment
    - reward
    - policy llm model
    """


    async def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        tool_choice = None,
        structured_model = None,
        **kwargs: Any,
    ):

        # For qvq and qwen-vl models, the content field cannot be `None` or
        # `[{"text": None}]`, so we need to convert it to an empty list.
        if self.model_name.startswith("qvq") or "-vl" in self.model_name:
            raise NotImplementedError("Not implemented for qvq and qwen-vl models yet.")

        kwargs = {
            "messages": messages,
            "model": self.model_name,
            "stream": self.stream,
            **self.dscm_ref.generate_kwargs,
            **kwargs,
            "result_format": "message",
            # In agentscope, the `incremental_output` must be `True` when
            # `self.stream` is True
            "incremental_output": self.stream,
        }

        if tools:
            kwargs["tools"] = self.dscm_ref._format_tools_json_schemas(tools)

        if tool_choice:
            self.dscm_ref._validate_tool_choice(tool_choice, tools)
            kwargs["tool_choice"] = self.dscm_ref._format_tool_choice(tool_choice)

        if (
            self.dscm_ref.enable_thinking is not None
            and "enable_thinking" not in kwargs
        ):
            kwargs["enable_thinking"] = self.dscm_ref.enable_thinking

        if structured_model:
            if tools or tool_choice:
                logger.warning(
                    "structured_model is provided. Both 'tools' and "
                    "'tool_choice' parameters will be overridden and "
                    "ignored. The model will only perform structured output "
                    "generation without calling any other tools.",
                )
            format_tool = _create_tool_from_base_model(structured_model)
            kwargs["tools"] = self.dscm_ref._format_tools_json_schemas(
                [format_tool],
            )
            kwargs["tool_choice"] = self.dscm_ref._format_tool_choice(
                format_tool["function"]["name"],
            )

        response = await self.execute_model_proxy(
            api_key=self.dscm_ref.api_key,
            structured_model=structured_model,
            **kwargs,
        )
        return response

    def update_agentscope_input_dictionary(self, **kwargs):
        self.input_kwargs.update(kwargs)

    def get_agentscope_input_dictionary(self):
        return self.input_kwargs

    def update_judge_input_dictionary(self, **kwargs):
        self.output_kwargs.update(kwargs)

    def get_judge_input_dictionary(self):
        return self.output_kwargs

    def get_judge(self):
        judge_protocol = self.config.astune.task_judge.judge_protocol
        module_, class_ = judge_protocol.split('->')
        protocol_cls = getattr(importlib.import_module(module_), class_)
        return protocol_cls(self.config)  # type: ignore
