import copy
import json
import time
import numpy as np
import uuid
import asyncio
from pydantic import BaseModel
from typing import Dict, List, Literal, Callable, Any, Type
from loguru import logger
from omegaconf import DictConfig
from astune.utils.utils import run_async_coro__no_matter_what, remove_fields
from astune.utils.testing_utils import _mock_if_test_mode, _test_if_test_mode
from astune.utils.tokenizer import astune_apply_chat_template
from astune.schema.logprob import TokenAndProb
from agentscope.model import ChatResponse
from agentscope.message import TextBlock, ToolUseBlock
from agentscope._utils._common import _json_loads_with_repair
from transformers.tokenization_utils import PreTrainedTokenizer
from astune.context_tracker.agentscope_tracker.multiagent_tracking import (
    MultiAgentContextTracking,
)
from vllm.entrypoints.openai.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


class AsyncLlmBridge(object):

    def __init__(
        self,
        config: DictConfig,
        async_rollout_manager: Any,
        tokenizer: Any,
        llm_mode: Literal["local", "remote", "trinity"] = "local",
        max_llm_retries: int = 3,
    ):
        self.config = config
        self.async_rollout_manager = async_rollout_manager
        self.tokenizer = tokenizer
        self.llm_mode = llm_mode
        self.max_llm_retries = max_llm_retries

    def get_llm_chat_fn(self, sampling_params: dict = {}) -> Callable:

        def llm_chat(
            messages: List[Dict[str, str]],
            custom_sampling_params: dict = {},
            tools=[],
            request_id: str = "",
        ) -> dict:

            request_id = uuid.uuid4().hex

            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            input_messages = copy.deepcopy(messages)
            prompt_text = astune_apply_chat_template(
                tokenizer=self.tokenizer,
                conversation=input_messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False
            )
            prompt_ids = self.tokenizer(prompt_text)["input_ids"]

            _test_if_test_mode('prompt_text', prompt_text, self.config)

            final_res = run_async_coro__no_matter_what(
                self.async_rollout_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=updated_sampling_params,
                ), timeout=1800
            )

            if self.config.astune.rollout.name == "vllm":
                token_array = final_res.outputs[0].token_ids
            elif self.config.astune.rollout.name == "sglang":
                token_array = final_res

            decoded_text = self.tokenizer.decode(token_array)  # type: ignore
            if self.config.astune.execute_test:
                decoded_text = _mock_if_test_mode('mock_decoded_text', decoded_text, self.config)

            if decoded_text.endswith("<|im_end|>"):
                decoded_text = decoded_text[: -len("<|im_end|>")]

            # if tool call
            tool_calls = None
            if ('<tool_call>' in decoded_text) and ('</tool_call>' in decoded_text) and (not self.config.astune.rollout.agentscope_disable_toolcalls):
                tool_parser = Hermes2ProToolParser(self.tokenizer)
                parsed_tool_calls = tool_parser.extract_tool_calls(decoded_text, None)  # type: ignore
                parsed_tool_calls = parsed_tool_calls.model_dump()
                _test_if_test_mode('parsed_tool_calls', parsed_tool_calls['tool_calls'], self.config)
                model_called = parsed_tool_calls['tools_called']
                if model_called:
                    tool_calls = parsed_tool_calls['tool_calls']
                    is_bad_toolcall = False
                    for i in range(len(tool_calls)):
                        if 'function' in tool_calls[i] and 'arguments' in tool_calls[i]['function']:
                            expect_dict = json.loads(tool_calls[i]['function']['arguments'])
                            if not isinstance(expect_dict, dict):
                                is_bad_toolcall = True
                    if is_bad_toolcall:
                        tool_calls = None
                        decoded_text = decoded_text
                    else:
                        decoded_text = parsed_tool_calls['content']
                        if decoded_text is None:
                            decoded_text = ""

            return {
                "role": "assistant",
                "request_id": request_id,
                "content": decoded_text,
                "tool_calls": tool_calls,
                "tokens": [
                    TokenAndProb(
                        token_id=token,
                        logprob=-1,
                        decoded_string=self.tokenizer.decode(token),
                    )
                    for token in token_array  # type: ignore
                ],
            }

        def llm_chat_remote(
            messages: List[Dict[str, str]],
            custom_sampling_params: dict = {},
            tools=[],
            request_id: str = "",
        ) -> dict:

            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)
            updated_sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})
            input_messages = copy.deepcopy(messages)
            for i in range(self.max_llm_retries):
                try:
                    # this function is defined in `astune/main_vllm.py`
                    output_message = self.async_rollout_manager.submit_chat_completions(
                        messages=input_messages,
                        sampling_params=updated_sampling_params,
                        tools=tools,
                        request_id=request_id,
                    )
                    break
                except Exception as e:
                    logger.bind(exception=True).exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)
            return output_message[-1]  # type: ignore

        def llm_chat_trinity(
            messages: List[Dict[str, str]],
            custom_sampling_params: dict = {},
            tools=[],
            request_id: str = "",
        ) -> dict:

            async def main():
                updated_sampling_params = {}
                if sampling_params:
                    updated_sampling_params.update(sampling_params)
                if custom_sampling_params:
                    updated_sampling_params.update(custom_sampling_params)
                updated_sampling_params.pop("min_tokens")

                if tools:
                    response = await self.async_rollout_manager.chat.completions.create(
                        model=self.async_rollout_manager.model_path,
                        messages=messages,
                        logprobs=True,
                        tools=tools,
                        top_logprobs=0,
                        **updated_sampling_params,
                    )
                else:
                    response = await self.async_rollout_manager.chat.completions.create(
                        model=self.async_rollout_manager.model_path,
                        messages=messages,
                        logprobs=True,
                        top_logprobs=0,
                        **updated_sampling_params,
                    )
                return response

            response = run_async_coro__no_matter_what(main(), timeout=1800)  # type: ignore
            prompt_text = self.tokenizer.decode(response.model_extra['prompt_token_ids'])
            prompt_token_ids = response.model_extra['prompt_token_ids']
            content = response.choices[0].message.content
            message = response.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)

            if content is None:
                content = ""

            if ('<tool_call>' in content) and (not message.get("tool_calls", None)):
                # logger.bind(exception=True).exception(f"Bad toolcall discovered \n\nprompt_text:\n{prompt_text}\n\nrepsonse:\n{content}")
                logger.warning(f"Bad toolcall discovered: {content}")

            return {
                "role": "assistant",
                "request_id": response.id,
                "content": content,
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt_token_ids,
                "tool_calls": message.get("tool_calls", []),
                "tokens": [
                    TokenAndProb(
                        token_id=token,
                        logprob=tokenlogprob.logprob,
                        decoded_string=tokenlogprob.token,
                    )
                    for tokenlogprob, token in zip(
                        response.choices[0].logprobs.content,
                        response.choices[0].token_ids,
                    )
                ],
            }

        if self.llm_mode == "remote":
            return llm_chat_remote
        if self.llm_mode == "trinity":
            return llm_chat_trinity
        else:
            return llm_chat


class LlmProxyForAgentScope(object):
    """
    An essential wrapper to connect AsyncLlmBridge with AgentScope

    User_Agentscope_Workflow <-> AsyncLlmBridge <-> Context Tracker.
    """

    def __init__(
        self,
        llm_chat_fn,
        tokenizer: PreTrainedTokenizer,
        context_tracker: MultiAgentContextTracking,
        config,
    ) -> None:
        self.context_tracker = context_tracker
        self.llm_chat_fn = llm_chat_fn
        self.tokenizer = tokenizer
        self.config = config

    async def __call__(
        self,
        messages: List[dict],
        tools: List = [],
        tool_choice: str = "auto",
        structured_model=None,
        **kwargs,
    ) -> ChatResponse:

        # prepare context tracker, check context safety
        context_safe, token_overflow, info, converted_message, custom_sampling_params, tools = (
            self.context_tracker.step_prepare(messages, tools)
        )
        if not context_safe:
            logger.warning(f"[{info}] detected.")
            self.context_tracker.context_overflow = True
            if token_overflow:
                # astune_action_when_overflow = self.config.astune.rollout.astune_action_when_overflow
                # cannot proceed due to context overflow
                return ChatResponse(
                    content=[{"type": "text", "text": "astune_proxy: Exceeded max model context length."}],
                )
            # else: # otherwise, for abnormal output, can still proceed, but we do not track output anymore

        # run llm inference ✨
        llm_output = await asyncio.wait_for(
            asyncio.to_thread(self.llm_chat_fn, converted_message, custom_sampling_params, tools),
            timeout=1800,
        )

        # begin context tracking
        self.context_tracker.step_track(llm_output, context_safe, converted_message, tools)

        # parse response
        response = await self._parse_dashscope_generation_response(
            llm_output, structured_model=structured_model
        )

        return response

    # copied from AgentScope's DashScopeChatModule
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
                        input=input_,  # type: ignore
                        id=tool_call["id"],
                    ),
                )

                if structured_model:
                    metadata = input_  # type: ignore

        parsed_response = ChatResponse(
            content=content_blocks,
            metadata=metadata,
        )

        return parsed_response
