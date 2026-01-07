
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
import time


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


def test_convert_llm_proxy_response_to_oai_response():
    """Test the conversion from llm_proxy_response to OpenAI ChatCompletion format."""

    from ajet.schema.logprob import TokenAndProb
    # Test case 1: Basic response with content only
    llm_proxy_response_basic = {
        "role": "assistant",
        "request_id": "req-123456",
        "content": "Hello, how can I help you today?",
        "tool_calls": None,
        "tokens": [
            TokenAndProb(
                token_id=123,
                logprob=-0.5,
                decoded_string="Hello",
            ),
            TokenAndProb(
                token_id=456,
                logprob=-0.3,
                decoded_string=",",
            ),
        ],
    }

    result = convert_llm_proxy_response_to_oai_response(llm_proxy_response_basic)

    assert result.id == "req-123456"
    assert result.object == "chat.completion"
    assert len(result.choices) == 1
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].message.content == "Hello, how can I help you today?"
    assert result.choices[0].message.tool_calls is None
    assert result.choices[0].finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.completion_tokens == 2
    assert result.usage.total_tokens == 2

    print("✓ Test case 1 passed: Basic response with content")

    # Test case 2: Response with tool calls
    llm_proxy_response_with_tools = {
        "role": "assistant",
        "request_id": "req-789012",
        "content": "",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco"}'
                }
            }
        ],
        "tokens": [],
    }

    result2 = convert_llm_proxy_response_to_oai_response(llm_proxy_response_with_tools)

    assert result2.id == "req-789012"
    assert result2.choices[0].message.content == ""
    assert result2.choices[0].message.tool_calls is not None
    assert len(result2.choices[0].message.tool_calls) == 1
    assert result2.usage is None  # No tokens provided

    print("✓ Test case 2 passed: Response with tool calls")

    # Test case 3: Minimal response with defaults
    llm_proxy_response_minimal = {
        "content": "Test response"
    }

    result3 = convert_llm_proxy_response_to_oai_response(llm_proxy_response_minimal)

    assert result3.id == "chatcmpl-default"
    assert result3.choices[0].message.role == "assistant"
    assert result3.choices[0].message.content == "Test response"
    assert result3.model == "unknown"

    print("✓ Test case 3 passed: Minimal response with defaults")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_convert_llm_proxy_response_to_oai_response()
