import copy
import time
import numpy as np
import uuid
from pydantic import BaseModel
from typing import Dict, List, Literal, Callable, Any, Type
from loguru import logger
from omegaconf import DictConfig
from astune.utils.utils import run_async_coro__no_matter_what, remove_fields
from astune.schema.logprob import TokenAndProb
from agentscope.model import ChatResponse
from agentscope.message import TextBlock, ToolUseBlock
from agentscope._utils._common import _json_loads_with_repair
from transformers.tokenization_utils import PreTrainedTokenizer
from astune.context_tracker.agentscope_tracker.multiagent_tracking import (
    MultiAgentContextTracking,
)


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

            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            tools = messages[-1].get("tools", None)
            for msg in messages:
                msg.pop("tools", None)

            input_messages = copy.deepcopy(messages)
            request_id = uuid.uuid4().hex
            if tools is not None:
                prompt_ids = self.tokenizer.apply_chat_template(
                    input_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    tools=tools,
                )
            else:
                prompt_ids = self.tokenizer.apply_chat_template(
                    input_messages, add_generation_prompt=True, tokenize=True
                )

            final_res = run_async_coro__no_matter_what(
                self.async_rollout_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=updated_sampling_params,
                )
            )

            if self.config.astune.rollout.name == "vllm":
                token_array = final_res.outputs[0].token_ids
            elif self.config.astune.rollout.name == "sglang":
                token_array = final_res

            decoded_text = self.tokenizer.decode(token_array)  # type: ignore

            if decoded_text.endswith("<|im_end|>"):
                decoded_text = decoded_text[: -len("<|im_end|>")]

            return {
                "role": "assistant",
                "request_id": request_id,
                "content": decoded_text,
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

            response = run_async_coro__no_matter_what(main())  # type: ignore

            content = response.choices[0].message.content
            message = response.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)

            if content is None:
                content = ""

            return {
                "role": "assistant",
                "request_id": response.id,
                "content": content,
                "tool_calls": message.get("tool_calls", None),
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
        context_safe, info, converted_message, custom_sampling_params, tools = (
            self.context_tracker.step_prepare(messages, tools)
        )
        if not context_safe:
            logger.warning(f"[{info}] detected. Current token count exceeds the limit.")
            self.context_tracker.context_overflow = True
            return ChatResponse(
                content=[{"type": "text", "text": "astune_proxy:[context_overflow]"}]
            )

        # run llm inference ✨
        llm_output = self.llm_chat_fn(converted_message, custom_sampling_params, tools)

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
