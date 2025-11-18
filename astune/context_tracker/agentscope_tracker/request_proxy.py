import copy
import time
from loguru import logger
from pydantic import BaseModel

from typing import Any, AsyncGenerator, List, Type, Dict
from agentscope.model import ChatResponse
from agentscope.message import TextBlock, ToolUseBlock
from astune.utils.compute_madness import compute_string_madness
from transformers.tokenization_utils import PreTrainedTokenizer
from agentscope._utils._common import _json_loads_with_repair, _create_tool_from_base_model
from astune.context_tracker.basic_tracker import BasicContextTracker, ExtendedMessage
from astune.context_tracker.agentscope_tracker.multiagent_tracking import MultiAgentContextTracking
from astune.context_tracker.agentscope_tracker.timeline_merging import can_merge_steps
from agentscope.model import ChatResponse

def remove_fields(d: Dict, fields: List[str]) -> Dict:
    d = copy.deepcopy(d)
    for field in fields:
        d.pop(field.strip(), None)
    return d

class ASTuneLlmProxy(object):

    def __init__(
        self,
        llm_chat_fn,
        tokenizer:PreTrainedTokenizer,
        context_tracker:MultiAgentContextTracking,
        config,
    ) -> None:
        self.context_tracker = context_tracker
        self.llm_chat_fn = llm_chat_fn
        self.tokenizer = tokenizer
        self.config = config


    async def execute_model_proxy(
            self,
            messages: List[dict],
            tools: List=[],
            tool_choice: str = "auto",
            structured_model=None,
            **kwargs
        ) -> ChatResponse:


        # prepare context tracker, check context safety
        context_safe, info, converted_message, custom_sampling_params = \
            self.context_tracker.step_prepare(messages, tools)
        if not context_safe:
            logger.warning(f"[{info}] detected. Current token count exceeds the limit.")
            self.context_overflow = True
            return ChatResponse(
                content = [{'type': 'text', 'text': 'astune_proxy:[context_overflow]'}]
            )

        # run llm inference
        llm_output = self.llm_chat_fn(converted_message, custom_sampling_params, tools)

        # begin context tracking
        self.context_tracker.step_track(llm_output, context_safe, converted_message, tools)

        # parse response
        response = await self._parse_dashscope_generation_response(llm_output, structured_model=structured_model)
        return response


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
