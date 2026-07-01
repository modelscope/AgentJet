"""
OpenAI Responses API ↔ Chat Completions API adapter.

This module lets the AgentJet interchange server expose a `/v1/responses` endpoint
that re-uses the existing Chat Completions ZMQ pipeline (oai_model_client.py /
OpenaiLlmProxyWithTracker) without any worker-side change. All Responses API
requests are translated to ChatCompletionRequest before being forwarded over
ZMQ, and the ChatCompletion that comes back is translated to a Response object
(or to Responses-style SSE events when streaming).

The conversion is intentionally permissive on the input side: the OpenAI
Responses API accepts many shapes of `input` (a plain string, a list of
EasyInputMessageParam dicts, a list of typed ResponseInputMessageItem /
ResponseFunctionToolCall / ResponseFunctionCallOutput items, ...). We walk the
list and produce an equivalent chat-completion messages list. Anything we do
not understand is dropped with a warning so that the most common shapes (text
messages, assistant tool calls, tool outputs) always go through.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from openai.types.chat.chat_completion import ChatCompletion


# ---------------------------------------------------------------------------
# Response model — declared locally so we do not depend on every concrete
# openai.types.responses.* subclass being importable across SDK versions.
# We emit the wire dict directly via FastAPI / JSONResponse; this mirrors the
# shapes defined by the OpenAI Responses API spec and is what the OpenAI SDK
# parses into `openai.types.responses.Response`.
# ---------------------------------------------------------------------------


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _extract_text_from_content(content: Any) -> str:
    """Best-effort coerce a Responses-style message `content` field to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type")
            if t in ("input_text", "output_text", "text"):
                text = part.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
            # Skip image / file inputs — we cannot forward them through chat completions.
        return "".join(parts)
    if isinstance(content, dict):
        return _extract_text_from_content([content])
    return str(content)


def _coerce_role(role: Any) -> str:
    """Map a Responses input message role to a chat-completion role.

    The Responses API allows `user | assistant | system | developer`. The chat
    completions API understands `system | user | assistant | tool`; `developer`
    maps to `system` (the highest-priority instruction channel in both APIs).
    """
    role = (role or "user").lower()
    if role in ("user", "assistant", "system", "tool"):
        return role
    if role == "developer":
        return "system"
    return "user"


def responses_tools_to_chat_tools(tools: Optional[List[Any]]) -> Tuple[List[Any], Optional[Any]]:
    """Convert Responses-style `tools` to chat-completion `tools`/`tool_choice`.

    The Responses API represents a function tool as
        {"type": "function", "name": ..., "parameters": ..., "strict": ...}
    while chat completions nests the function metadata under a `function` key:
        {"type": "function", "function": {"name": ..., "parameters": ...}}.

    Non-function tools (web_search, file_search, computer, ...) are not
    supported by the rollout engine, so we drop them with a warning.
    """
    if not tools:
        return [], None

    chat_tools: List[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function":
            fn = {
                "name": tool.get("name"),
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            }
            if "description" in tool:
                fn["description"] = tool["description"]
            if "strict" in tool:
                fn["strict"] = tool["strict"]
            chat_tools.append({"type": "function", "function": fn})
        else:
            logger.debug(
                f"[responses] dropping non-function tool of type {tool.get('type')!r}; "
                "rollout engine only supports function tools."
            )
    return chat_tools, None


def _convert_one_input_item(item: Any) -> List[Dict[str, Any]]:
    """Translate one Responses input item into zero or more chat-completion messages.

    Returns a list because some items (function_call_output) need to become a
    `tool` message — we keep the ordering stable by emitting one message per
    item in the common case.
    """
    if isinstance(item, str):
        return [{"role": "user", "content": item}]
    if not isinstance(item, dict):
        return []

    item_type = item.get("type")

    # Typed input: {"type": "message", "role": ..., "content": [...]}
    if item_type == "message":
        role = _coerce_role(item.get("role"))
        text = _extract_text_from_content(item.get("content"))
        if not text:
            return []
        return [{"role": role, "content": text}]

    # Typed input: {"type": "function_call", "name":..., "arguments":..., "call_id":...}
    # This represents an assistant tool call from a previous turn.
    if item_type == "function_call":
        call_id = item.get("call_id") or item.get("id") or _new_id("call")
        name = item.get("name", "")
        arguments = item.get("arguments", "")
        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments, ensure_ascii=False)
        # Emit a synthetic assistant message carrying a tool_call; the chat
        # completions API expects tool_calls to live on an assistant message
        # AND a matching `tool` message to follow with the same tool_call_id.
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        ]

    # Typed input: {"type": "function_call_output", "call_id":..., "output":...}
    # This is the tool's return value; map to a chat-completion `tool` message.
    if item_type == "function_call_output":
        call_id = item.get("call_id") or ""
        output = item.get("output", "")
        if isinstance(output, (dict, list)):
            output = json.dumps(output, ensure_ascii=False)
        return [{"role": "tool", "tool_call_id": call_id, "content": str(output)}]

    # Easy-input shape: {"role": ..., "content": ...} with no `type`.
    if "role" in item and "content" in item and item_type is None:
        role = _coerce_role(item.get("role"))
        text = _extract_text_from_content(item.get("content"))
        if not text:
            return []
        return [{"role": role, "content": text}]

    logger.debug(f"[responses] ignoring unsupported input item of type {item_type!r}: {item!r}")
    return []


def responses_input_to_chat_messages(
    input_field: Any,
    instructions: Optional[str],
) -> List[Dict[str, Any]]:
    """Build a chat-completion messages list from a Responses API `input`.

    Pre-pends `instructions` (if any) as a leading `system` message so the
    model sees the developer instruction before any user turn. This mirrors
    the existing chat-completions endpoint behaviour of injecting a system
    message when none is present.
    """
    messages: List[Dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_field, str):
        if input_field:
            messages.append({"role": "user", "content": input_field})
    elif isinstance(input_field, list):
        for item in input_field:
            messages.extend(_convert_one_input_item(item))
    elif input_field is None:
        pass
    else:
        # Unknown shape; stringify defensively rather than 400-ing.
        messages.append({"role": "user", "content": str(input_field)})

    return messages


def build_chat_completion_request(
    responses_body: Dict[str, Any],
) -> Tuple[ChatCompletionRequest, bool]:
    """Translate a Responses API request body into a ChatCompletionRequest.

    Returns the constructed request plus the original `stream` flag (so the
    caller can decide whether to wrap the response as SSE). The actual ZMQ
    pipeline is non-streaming — we always set `stream=False` on the forwarded
    request and convert the final ChatCompletion to the appropriate format.
    """
    original_stream = bool(responses_body.get("stream", False))

    messages = responses_input_to_chat_messages(
        input_field=responses_body.get("input"),
        instructions=responses_body.get("instructions"),
    )
    if messages and messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant, your name is AgentJet."})

    tools, _ = responses_tools_to_chat_tools(responses_body.get("tools"))

    cc_body: Dict[str, Any] = {
        "model": responses_body.get("model") or "unknown",
        "messages": messages,
        "stream": False,
    }

    if tools:
        cc_body["tools"] = tools
        # Default to auto; explicit tool_choice passes through unchanged.
        tc = responses_body.get("tool_choice")
        if tc is None:
            cc_body["tool_choice"] = "auto"
        elif isinstance(tc, str):
            cc_body["tool_choice"] = tc
        elif isinstance(tc, dict):
            # Responses tool_choice can be {"type": "function", "name": ...}
            if tc.get("type") == "function" and "name" in tc:
                cc_body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc["name"]},
                }
            elif tc.get("type") in ("auto", "none", "required"):
                cc_body["tool_choice"] = tc["type"]
            else:
                cc_body["tool_choice"] = "auto"

    # Sampling parameters (Responses names differ from chat completions).
    if "temperature" in responses_body and responses_body["temperature"] is not None:
        cc_body["temperature"] = responses_body["temperature"]
    if "top_p" in responses_body and responses_body["top_p"] is not None:
        cc_body["top_p"] = responses_body["top_p"]
    if "max_output_tokens" in responses_body and responses_body["max_output_tokens"] is not None:
        cc_body["max_tokens"] = responses_body["max_output_tokens"]
    if "stop" in responses_body and responses_body["stop"] is not None:
        cc_body["stop"] = responses_body["stop"]
    if "seed" in responses_body and responses_body["seed"] is not None:
        cc_body["seed"] = responses_body["seed"]
    if "presence_penalty" in responses_body and responses_body["presence_penalty"] is not None:
        cc_body["presence_penalty"] = responses_body["presence_penalty"]
    if "frequency_penalty" in responses_body and responses_body["frequency_penalty"] is not None:
        cc_body["frequency_penalty"] = responses_body["frequency_penalty"]

    # Strip any None values — vLLM's pydantic model rejects unknown unset fields.
    cc_body = {k: v for k, v in cc_body.items() if v is not None}

    return ChatCompletionRequest.model_validate(cc_body), original_stream


# ---------------------------------------------------------------------------
# ChatCompletion -> Response wire dict
# ---------------------------------------------------------------------------


def _finish_to_status(finish_reason: Optional[str]) -> str:
    if finish_reason == "length":
        return "incomplete"
    return "completed"


def _finish_to_incomplete_reason(finish_reason: Optional[str]) -> Optional[Dict[str, str]]:
    if finish_reason == "length":
        return {"reason": "max_output_tokens"}
    return None


def chat_completion_to_responses_dict(
    cc: ChatCompletion,
    *,
    model: str,
    instructions: Optional[str] = None,
    response_id: Optional[str] = None,
    created_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert a ChatCompletion (from the ZMQ worker) into a Responses API dict."""
    now = time.time()
    rid = response_id or _new_id("resp")
    created_ts = created_at if created_at is not None else float(cc.created or int(now))

    choice = cc.choices[0] if cc.choices else None
    message = choice.message if choice else None
    finish_reason = choice.finish_reason if choice else None

    output: List[Dict[str, Any]] = []

    # 1) Assistant message with text content (if any).
    content_text = ""
    if message is not None:
        content_text = message.content or ""
    if content_text:
        output.append(
            {
                "id": _new_id("msg"),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": content_text,
                        "annotations": [],
                    }
                ],
            }
        )

    # 2) Function tool calls (each becomes its own output item).
    if message is not None and message.tool_calls:
        for tc in message.tool_calls:
            arguments = tc.function.arguments if tc.function else ""
            name = tc.function.name if tc.function else ""
            output.append(
                {
                    "id": tc.id or _new_id("fc"),
                    "type": "function_call",
                    "call_id": tc.id or _new_id("call"),
                    "name": name,
                    "arguments": arguments,
                    "status": "completed",
                }
            )

    # 3) Usage mapping. ChatCompletion.usage has prompt_tokens / completion_tokens /
    #    total_tokens; Responses uses input_tokens / output_tokens / total_tokens
    #    plus nested *_tokens_details.
    usage_obj = cc.usage.model_dump() if cc.usage else {}
    input_tokens = int(usage_obj.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage_obj.get("completion_tokens", 0) or 0)
    total_tokens = int(usage_obj.get("total_tokens", input_tokens + output_tokens) or 0)

    response_dict: Dict[str, Any] = {
        "id": rid,
        "object": "response",
        "created_at": created_ts,
        "completed_at": now,
        "model": model,
        "status": _finish_to_status(finish_reason),
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": total_tokens,
        },
    }
    if instructions is not None:
        response_dict["instructions"] = instructions
    incomplete = _finish_to_incomplete_reason(finish_reason)
    if incomplete is not None:
        response_dict["incomplete_details"] = incomplete
    return response_dict


# ---------------------------------------------------------------------------
# Streaming: emit Responses API SSE events by chunking the final response.
# This mirrors the existing chat-completions endpoint's mock_as_stream_response
# — we wait for the full ChatCompletion from the worker, then synthesize the
# event sequence the OpenAI SDK expects.
# ---------------------------------------------------------------------------


def _sse(event_type: str, payload: Dict[str, Any], sequence: int) -> str:
    payload_with_type = dict(payload)
    payload_with_type["type"] = event_type
    payload_with_type["sequence_number"] = sequence
    return f"event: {event_type}\ndata: {json.dumps(payload_with_type, ensure_ascii=False)}\n\n"


def iter_responses_sse_events(
    response_dict: Dict[str, Any],
    text_deltas: Optional[List[str]] = None,
) -> Iterable[str]:
    """Yield SSE-formatted Responses events for the given Response dict.

    `text_deltas` lets the caller control how the assistant text is split into
    `response.output_text.delta` events. When None, the whole text is shipped
    in a single delta (matches our non-incremental worker).
    """
    seq = 0

    # response.created — strip the heavy output array for the snapshot.
    created_snapshot = {k: v for k, v in response_dict.items() if k != "output"}
    created_snapshot["output"] = []
    seq += 1
    yield _sse("response.created", {"response": created_snapshot}, seq)

    # response.in_progress
    seq += 1
    yield _sse("response.in_progress", {"response": created_snapshot}, seq)

    # For each output item, emit output_item.added → content_part.added →
    # output_text.delta* → output_text.done → content_part.done → output_item.done.
    # For function_call items we skip the content-part deltas (no text).
    text_message_item = None
    for item in response_dict.get("output", []):
        if item.get("type") == "message" and item.get("content"):
            text_message_item = item
            break

    if text_message_item is not None:
        # output_item.added (without content)
        item_added = dict(text_message_item)
        item_added["content"] = []
        seq += 1
        yield _sse("response.output_item.added", {"output_index": 0, "item": item_added}, seq)

        content_part = text_message_item["content"][0]
        # content_part.added (empty text)
        added_part = dict(content_part)
        added_part["text"] = ""
        seq += 1
        yield _sse("response.content_part.added", {"output_index": 0, "content_index": 0, "part": added_part}, seq)

        full_text = content_part.get("text", "") or ""
        if text_deltas:
            chunks = text_deltas
        else:
            chunks = [full_text]
        for chunk in chunks:
            seq += 1
            yield _sse(
                "response.output_text.delta",
                {"output_index": 0, "content_index": 0, "delta": chunk},
                seq,
            )
        seq += 1
        yield _sse(
            "response.output_text.done",
            {"output_index": 0, "content_index": 0, "text": full_text},
            seq,
        )
        seq += 1
        yield _sse(
            "response.content_part.done",
            {"output_index": 0, "content_index": 0, "part": content_part},
            seq,
        )
        seq += 1
        yield _sse("response.output_item.done", {"output_index": 0, "item": text_message_item}, seq)

    # response.completed — full snapshot.
    seq += 1
    yield _sse("response.completed", {"response": response_dict}, seq)
