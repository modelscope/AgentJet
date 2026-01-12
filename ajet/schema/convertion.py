
import time
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from agentscope.model import ChatResponse as AgentScopeChatResponse
from openai.types.completion_usage import CompletionUsage
from typing import Any, Callable, Dict, List, Literal, Type, Union
from agentscope.message import TextBlock, ToolUseBlock
from agentscope._utils._common import _json_loads_with_repair
from pydantic import BaseModel
from agentscope.model import ChatResponse


def convert_llm_proxy_response_to_oai_response(llm_proxy_response):

    # Create the chat completion message
    message = ChatCompletionMessage(
        role=llm_proxy_response.get("role", "assistant"),
        content=llm_proxy_response.get("content", ""),
        tool_calls=llm_proxy_response.get("tool_calls", []),
    )

    # Create a choice object
    choice = Choice(
        index=0,
        message=message,
        finish_reason="stop",
    )

    # Calculate token usage if tokens are available
    usage = None
    if "tokens" in llm_proxy_response and llm_proxy_response["tokens"]:
        completion_tokens = len(llm_proxy_response["tokens"])
        usage = CompletionUsage(
            prompt_tokens=0,  # Not available in llm_proxy_response
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
        )

    return ChatCompletion(
        id=llm_proxy_response.get("request_id", "chatcmpl-default"),
        choices=[choice],
        created=int(time.time()),
        model="unknown",  # Model name not provided in llm_proxy_response
        object="chat.completion",
        usage=usage,
    )



# modified from AgentScope's DashScopeChatModule
def convert_llm_proxy_response_to_agentscope_response(
    message,
    structured_model: Type[BaseModel] | None = None,
) -> AgentScopeChatResponse:    # type: ignore
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

    parsed_response = AgentScopeChatResponse(
        content=content_blocks,
        metadata=metadata,
    )

    return parsed_response

