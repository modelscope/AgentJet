import copy
import time
from loguru import logger
from pydantic import BaseModel
from beast_logger import print_dict, print_listofdict

from transformers.tokenization_utils import PreTrainedTokenizer
from agentscope.model import ChatResponse
from agentscope.message import TextBlock, ToolUseBlock
from agentscope._utils._common import _json_loads_with_repair, _create_tool_from_base_model
from astune.context_manager.cmt_linear import CMTLinear, ExtendedMessage
from astune.utils.compute_madness import compute_string_madness
from astune.context_manager.agentscope_cm.cmt_multi_sample import ASTuneContextTemplate

from typing import Any, List, Type, Dict

def remove_fields(d: Dict, fields: List[str]) -> Dict:
    d = copy.deepcopy(d)
    for field in fields:
        d.pop(field.strip(), None)
    return d

class ASTuneLmProxy(ASTuneContextTemplate):

    async def execute_model_proxy(self, messages: List[dict], tools: List=[], tool_choice: str = "auto", structured_model=None, **kwargs) -> dict:
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
                ignore = False
                str_content = ""
                for item in msg['content']:
                    if 'text' not in item:
                        logger.warning(f"Non-text content in message content detected: {item}. Ignoring.")
                        ignore = True
                        break
                    str_content += str(item['text'])
                    msg['content'] = str_content
                if ignore:
                    continue
                msg['content'] = str(msg['content'])    # TODO: better handling mm data
            if msg['role'] == 'system':
                author = 'initialization'
            if msg['role'] == 'tool':
                author = 'env'
            else:
                author = 'env'

            self.full_context += [
                ExtendedMessage(
                    author=author,
                    role=msg['role'],
                    content=msg['content'],
                    tokenizer=self.tokenizer,
                    tools=tools,
                    tool_calls=msg['tool_calls'] if 'tool_calls' in msg else [],
                    token_generator="auto",
                )
            ]

        # check token overflow
        converted_message = self.to_role_content(self.full_context)
        is_safe, info = self.check_context_token_num_safe(converted_message, tools)
        custom_sampling_params = {}
        if not is_safe:
            logger.warning(f"[{info}] detected. Current token count exceeds the limit.")
            self.context_overflow = True
            return ChatResponse(
                content = [{'type': 'text', 'text': 'astune_proxy:[context_overflow]'}]
            )
        # print_listofdict(converted_message, header='converted proxy messages')
        # print_listofdict(messages, header='proxy messages')
        assert self.tokenizer.apply_chat_template(converted_message, tokenize=False, add_generation_prompt=True, tools=tools) == self.tokenizer.apply_chat_template(messages,tokenize=False, add_generation_prompt=True,tools=tools)
        llm_output = self.llm_chat_fn(messages, custom_sampling_params, tools)
        print_dict(remove_fields(llm_output, fields=['request_id', 'tokens', 'tool_call_id']), header='response')

        print('-----------------------------------------')
        print('-----------------------------------------')
        print('-----------------------------------------')
        print('-----------------------------------------')
        time.sleep(10)
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
            token_arr_method2, token_logprob_arr = self.get_token_inc_from_vllm_response(input_msg_ref, llm_output, tools=tools)
            assert len(token_arr_method2) <= self.config.astune.rollout.max_response_length_in_one_turn, f"Generated token length {len(token_arr_method2)} exceeds max_response_len {self.config.astune.rollout.max_response_length_in_one_turn}"
            llm_ext_msg.token_arr = token_arr_method2
            llm_ext_msg.token_logprob_arr = token_logprob_arr
            self.generated_token_callback_fn(llm_ext_msg.token_arr)

        # take snapshot of current timeline
        if is_safe:
            self.full_context += [ llm_ext_msg ]
            is_safe, length = self.get_context_token_num_and_safety(self.full_context, tools)
            if not is_safe:
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
                        input=input_,   # type: ignore
                        id=tool_call["id"],
                    ),
                )

                if structured_model:
                    metadata = input_   # type: ignore


        parsed_response = ChatResponse(
            content=content_blocks,
            metadata=metadata,
        )

        return parsed_response
