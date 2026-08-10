"""
Anthropic Messages API ↔ Chat Completions API adapter.

This module lets the AgentJet interchange server expose a `/v1/messages` endpoint
that re-uses the existing Chat Completions ZMQ pipeline
(oai_model_client.py / OpenaiLlmProxyWithTracker) without any worker-side change.
All Messages API requests are translated to ChatCompletionRequest before being
forwarded over ZMQ, and the ChatCompletion that comes back is translated to an
Anthropic `Message` object (or to Messages-style SSE events when streaming).

The conversion is intentionally permissive on the input side: the Anthropic
Messages API represents message `content` either as a plain string or as a list
of typed content blocks (`text`, `tool_use`, `tool_result`). We walk the list
and produce an equivalent chat-completion messages list. Image / file blocks
cannot be forwarded through chat completions and are dropped with a warning, so
the most common shapes (text, assistant tool calls, tool outputs) always go
through.

This mirrors oai_responses_adapter.py in shape:
  - build_chat_completion_request(body)           -> (ChatCompletionRequest, stream)
  - chat_completion_to_message_dict(cc, ...)      -> dict
  - iter_anthropic_sse_events(message_dict)       -> Iterable[str]
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

try:
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
except ModuleNotFoundError:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from openai.types.chat.chat_completion import ChatCompletion


# ---------------------------------------------------------------------------
# IDs / small helpers
# ---------------------------------------------------------------------------


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _content_to_text(content: Any) -> str:
    """Coerce an Anthropic message `content` (string or list of blocks) to text.

    Tool-use / tool-result blocks contribute no text here; they are handled
    separately so they can become tool_calls / tool messages.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = part.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(content, dict):
        return _content_to_text([content])
    return str(content)


def _system_to_text(system: Any) -> str:
    """Anthropic `system` is either a string or a list of {type:text,text} blocks."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return " ".join(
            part.get("text", "")
            for part in system
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(system)


# ---------------------------------------------------------------------------
# Input translation: Anthropic Messages request -> ChatCompletionRequest
# ---------------------------------------------------------------------------


def _anthropic_tools_to_chat_tools(tools: Optional[List[Any]]) -> List[Any]:
    """Convert Anthropic tool defs to chat-completion `tools`.

    Anthropic: {"name", "description", "input_schema"}
    Chat:      {"type":"function", "function":{"name","description","parameters"}}
    """
    if not tools:
        return []
    chat_tools: List[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        fn: Dict[str, Any] = {
            "name": name,
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        }
        if "description" in tool:
            fn["description"] = tool["description"]
        chat_tools.append({"type": "function", "function": fn})
    return chat_tools


def _anthropic_tool_choice_to_chat(tool_choice: Any, tool_names: List[str]) -> Any:
    """Map Anthropic tool_choice to the chat-completion tool_choice.

    Anthropic: {type:"auto"|"any"|"tool"|"none", name?}
    vLLM's ChatCompletionRequest only accepts "auto", "none", or a named tool
    ({"type":"function","function":{"name":...}}) — it rejects "required". So
    Anthropic "any" (must call *some* tool) maps to forcing the first available
    tool, which is exact for the common single-tool case and falls back to
    "auto" when no tools are named.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        tc = tool_choice.lower()
    elif isinstance(tool_choice, dict):
        tc = tool_choice.get("type")
    else:
        return None

    if tc == "auto":
        return "auto"
    if tc == "any":
        if tool_names:
            return {"type": "function", "function": {"name": tool_names[0]}}
        return "auto"
    if tc == "none":
        return "none"
    if tc == "tool":
        name = tool_choice.get("name") if isinstance(tool_choice, dict) else None
        if name:
            return {"type": "function", "function": {"name": name}}
        return "auto"
    return "auto"


def anthropic_messages_to_chat_messages(
    messages: Any,
    system: Any,
) -> List[Dict[str, Any]]:
    """Build a chat-completion messages list from Anthropic `messages` + `system`.

    `system` is prepended as a leading `system` message (Anthropic keeps system
    out of the message list, but chat completions expect it as the first turn).
    """
    out: List[Dict[str, Any]] = []

    sys_text = _system_to_text(system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})

    if not isinstance(messages, list):
        return out

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "user").lower()
        content = msg.get("content")
        # Qwen/verl tokenizer 要求 system 唯一且在开头; 顶层 system 已 prepend,
        # messages 里再出现的 system (claude code 多轮可能带上) 一律跳过, 否则
        # apply_chat_template 报 "System message must be at the beginning".
        if role == "system":
            continue

        # Plain string content — the common case.
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            out.append({"role": role, "content": _content_to_text(content)})
            continue

        # Walk typed content blocks. Accumulate text; split out tool_use (on
        # assistant turns) and tool_result (on user turns) into the shapes the
        # chat-completions API expects.
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")

            if btype == "text":
                t = block.get("text", "")
                if isinstance(t, str):
                    text_parts.append(t)

            elif btype == "tool_use" and role == "assistant":
                call_id = block.get("id") or _new_id("call")
                name = block.get("name", "")
                arguments = block.get("input", "")
                if isinstance(arguments, (dict, list)):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif arguments is None:
                    arguments = ""
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": str(arguments)},
                    }
                )

            elif btype == "tool_result" and role == "user":
                # Flush any text accumulated on this user turn before the tool
                # result, then emit the tool message.
                if text_parts:
                    out.append({"role": "user", "content": "".join(text_parts)})
                    text_parts = []
                call_id = block.get("tool_use_id", "")
                result = block.get("content", "")
                if isinstance(result, list):
                    # tool_result.content can itself be a list of text blocks.
                    result = _content_to_text(result)
                elif not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False)
                out.append({"role": "tool", "tool_call_id": call_id, "content": result})

            elif btype in ("image", "document", "tool_use", "tool_result"):
                # tool_use/tool_result with a non-matching role, or image/doc
                # blocks which we cannot forward through chat completions.
                if btype in ("image", "document"):
                    logger.debug(
                        f"[messages] dropping {btype} content block; chat-completions "
                        "pipeline cannot forward non-text content."
                    )

        if role == "assistant":
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            assistant_msg["content"] = "".join(text_parts) if text_parts else None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            # Only emit if it carries text or tool calls.
            if assistant_msg["content"] or assistant_msg.get("tool_calls"):
                out.append(assistant_msg)
        else:
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})

    return out


def build_chat_completion_request(
    body: Dict[str, Any],
) -> Tuple[ChatCompletionRequest, bool]:
    """Translate an Anthropic Messages API body into a ChatCompletionRequest.

    Returns the constructed request plus the original `stream` flag. The ZMQ
    pipeline is non-streaming — we always set `stream=False` on the forwarded
    request and synthesize Messages SSE events ourselves on the way back.
    """
    original_stream = bool(body.get("stream", False))

    messages = anthropic_messages_to_chat_messages(
        messages=body.get("messages"),
        system=body.get("system"),
    )
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant, your name is AgentJet."})

    cc_body: Dict[str, Any] = {
        "model": body.get("model") or "unknown",
        "messages": messages,
        "stream": False,
    }

    # max_tokens is required by the Anthropic API; carry it through when present.
    if body.get("max_tokens") is not None:
        cc_body["max_tokens"] = body["max_tokens"]

    # Sampling parameters (names line up between the two APIs).
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if body.get(src) is not None:
            cc_body[dst] = body[src]
    if body.get("stop_sequences"):
        cc_body["stop"] = body["stop_sequences"]

    tools = _anthropic_tools_to_chat_tools(body.get("tools"))
    if tools:
        cc_body["tools"] = tools
        tool_names = [t["function"]["name"] for t in tools if t.get("function", {}).get("name")]
        tc = _anthropic_tool_choice_to_chat(body.get("tool_choice"), tool_names)
        cc_body["tool_choice"] = tc if tc is not None else "auto"

    # Strip any None values — vLLM's pydantic model rejects unknown unset fields.
    cc_body = {k: v for k, v in cc_body.items() if v is not None}

    return ChatCompletionRequest.model_validate(cc_body), original_stream


# ---------------------------------------------------------------------------
# ChatCompletion -> Anthropic Message wire dict
# ---------------------------------------------------------------------------


def _finish_to_stop_reason(finish_reason: Optional[str]) -> str:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    # "stop", "content_filter", None — no stop_sequence detection available.
    return "end_turn"


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """OpenAI tool_call.function.arguments is a JSON string; Anthropic wants an object."""
    if arguments is None:
        return {}
    if isinstance(arguments, (dict, list)):
        return arguments  # type: ignore[return-value]
    try:
        parsed = json.loads(arguments)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def chat_completion_to_message_dict(
    cc: ChatCompletion,
    *,
    model: str,
    system: Optional[str] = None,  # noqa: ARG001 — kept for API symmetry with the responses adapter
    message_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a ChatCompletion (from the ZMQ worker) into an Anthropic Message dict."""
    mid = message_id or _new_id("msg")

    choice = cc.choices[0] if cc.choices else None
    message = choice.message if choice else None
    finish_reason = choice.finish_reason if choice else None

    content_blocks: List[Dict[str, Any]] = []

    if message is not None and message.content:
        content_blocks.append({"type": "text", "text": message.content})

    if message is not None and message.tool_calls:
        for tc in message.tool_calls:
            name = tc.function.name if tc.function else ""
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id or _new_id("toolu"),
                    "name": name,
                    "input": _parse_tool_arguments(
                        tc.function.arguments if tc.function else ""
                    ),
                }
            )

    usage_obj = cc.usage.model_dump() if cc.usage else {}
    input_tokens = int(usage_obj.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage_obj.get("completion_tokens", 0) or 0)

    message_dict: Dict[str, Any] = {
        "id": mid,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": _finish_to_stop_reason(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
    return message_dict


# ---------------------------------------------------------------------------
# Streaming: emit Messages API SSE events by chunking the final message.
# Mirrors oai_responses_adapter.iter_responses_sse_events — the worker pipeline
# is non-incremental, so we wait for the full ChatCompletion and then emit the
# event sequence the Anthropic SDK expects.
# ---------------------------------------------------------------------------


def _sse(event_type: str, payload: Dict[str, Any]) -> str:
    payload_with_type = dict(payload)
    payload_with_type["type"] = event_type
    return f"event: {event_type}\ndata: {json.dumps(payload_with_type, ensure_ascii=False)}\n\n"


def iter_anthropic_sse_events(message_dict: Dict[str, Any]) -> Iterable[str]:
    """Yield SSE-formatted Messages events for the given Message dict."""
    usage = message_dict.get("usage", {}) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)

    # message_start — message snapshot with empty content and output_tokens: 0.
    start_snapshot = {
        "id": message_dict.get("id"),
        "type": "message",
        "role": "assistant",
        "model": message_dict.get("model"),
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
    yield _sse("message_start", {"message": start_snapshot})

    # Optional keepalive; harmless and matches the real API's stream shape.
    yield _sse("ping", {})

    # One content_block_start / deltas / content_block_stop per content block.
    content_blocks = message_dict.get("content", []) or []
    for index, block in enumerate(content_blocks):
        btype = block.get("type")
        if btype == "text":
            yield _sse(
                "content_block_start",
                {"index": index, "content_block": {"type": "text", "text": ""}},
            )
            full_text = block.get("text", "") or ""
            # Ship the whole text in a single delta — the worker isn't incremental.
            yield _sse(
                "content_block_delta",
                {"index": index, "delta": {"type": "text_delta", "text": full_text}},
            )
        elif btype == "tool_use":
            start_block = {
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": {},
            }
            yield _sse("content_block_start", {"index": index, "content_block": start_block})
            input_json = json.dumps(block.get("input", {}), ensure_ascii=False)
            yield _sse(
                "content_block_delta",
                {"index": index, "delta": {"type": "input_json_delta", "partial_json": input_json}},
            )
        else:
            # Unknown block type; emit a minimal start so indices stay aligned.
            yield _sse(
                "content_block_start",
                {"index": index, "content_block": {"type": btype or "text", "text": ""}},
            )
        yield _sse("content_block_stop", {"index": index})

    # message_delta carries stop_reason + final output_tokens; then message_stop.
    yield _sse(
        "message_delta",
        {
            "delta": {
                "stop_reason": message_dict.get("stop_reason"),
                "stop_sequence": message_dict.get("stop_sequence"),
            },
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _sse("message_stop", {})
