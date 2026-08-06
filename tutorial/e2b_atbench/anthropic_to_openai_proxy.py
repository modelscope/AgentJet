# -*- coding: utf-8 -*-
"""Anthropic -> OpenAI 透传翻译代理.

claude code 只会说 Anthropic `/v1/messages` (流式 + tools + system); 而 agentjet
swarm 的策略端点 (interchange server) 是 OpenAI 兼容 `/chat/completions`. 本模块
把 claude code 的 Anthropic 请求翻成 OpenAI 请求转发到上游 (策略 interchange),
再把 OpenAI 的流式响应翻回 Anthropic SSE 喂给 claude code. 这样策略的全部 token
都经过 interchange 被 swarm 按 episode_uuid 捕获, 用于 RL.

只覆盖 claude code 实际用到的子集: system / messages(text|tool_use|tool_result) /
tools / tool_choice / max_tokens / stop_sequences / temperature / top_p / stream,
以及 `/v1/messages/count_tokens` (粗估).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator, Optional

from aiohttp import ClientSession, ClientTimeout, web

logger = logging.getLogger("anthropic_to_openai")

# 与 cc_anthropic_proxy 一致: 转发时不带的逐跳头
_HOP_REQ = {
    "host", "content-length", "transfer-encoding", "connection", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade",
    "authorization",  # 上游用我们注入的 Bearer, 不用 claude code 的 x-api-key
}
_HOP_RESP = {
    "content-length", "transfer-encoding", "connection", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade",
    "content-encoding",
}


# ───────────────────────────── 请求: Anthropic -> OpenAI ────────────────────────

def _system_to_text(system: Any) -> str:
    if not system:
        return ""
    if isinstance(system, str):
        return system
    # list of {"type":"text","text": "..."} (可能还有 cache_control 等)
    parts = []
    for blk in system:
        if isinstance(blk, dict):
            parts.append(blk.get("text", ""))
        else:
            parts.append(str(blk))
    return "\n".join(p for p in parts if p)


def _content_to_text(content: Any) -> str:
    """把任意 Anthropic content (str 或 block 列表) 压成纯文本 (用于合并 system)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for blk in content or []:
        if isinstance(blk, dict):
            if blk.get("type") == "text":
                parts.append(blk.get("text", ""))
            elif "text" in blk:
                parts.append(blk.get("text", ""))
            else:
                parts.append(json.dumps(blk, ensure_ascii=False))
        else:
            parts.append(str(blk))
    return "\n".join(p for p in parts if p)


def _content_blocks_to_openai(role: str, content: Any) -> list[dict]:
    """把一条 Anthropic message 的 content 展开成若干 OpenAI 消息.

    user 的 tool_result 必须独立成 {role:"tool"}; text 仍留在 {role:user/assistant};
    assistant 的 tool_use 映射成 tool_calls.
    返回的 list 顺序即发送顺序.
    """
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    out: list[dict] = []
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    def flush_text(target_role: str):
        if text_parts:
            out.append({"role": target_role, "content": "\n".join(text_parts)})
        text_parts.clear()

    for blk in content or []:
        if not isinstance(blk, dict):
            text_parts.append(str(blk))
            continue
        btype = blk.get("type")
        if btype == "text":
            text_parts.append(blk.get("text", ""))
        elif btype == "tool_use":
            flush_text(role)
            tool_calls.append({
                "id": blk.get("id", f"call_{len(tool_calls)}"),
                "type": "function",
                "function": {
                    "name": blk.get("name", ""),
                    "arguments": json.dumps(blk.get("input", {}), ensure_ascii=False),
                },
            })
        elif btype == "tool_result":
            # tool_result 出现在 user 消息里; 内容可能是 str 或 blocks
            tc_content = blk.get("content", "")
            if isinstance(tc_content, list):
                tc_content = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in tc_content
                )
            elif not isinstance(tc_content, str):
                tc_content = json.dumps(tc_content, ensure_ascii=False)
            flush_text(role)
            out.append({
                "role": "tool",
                "tool_call_id": blk.get("tool_use_id", ""),
                "content": tc_content,
            })
        elif btype == "thinking":
            # 思维块: 丢弃 (OpenAI 路径不回放 reasoning), 不计入 text
            continue
        else:
            text_parts.append(json.dumps(blk, ensure_ascii=False))

    flush_text(role)
    if tool_calls:
        # 把 tool_calls 挂到最近一条 assistant 消息上; 若没有则建一条
        if out and out[-1]["role"] == "assistant":
            out[-1]["tool_calls"] = tool_calls
        else:
            out.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
    return out


def _convert_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return out


def _convert_tool_choice(tc: Any) -> Any:
    if tc is None:
        return None
    if isinstance(tc, str):
        return tc  # "auto" / "none" 透传
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "tool":
        return {"type": "function", "function": {"name": tc.get("name", "")}}
    return None


def anthropic_to_openai_body(body: dict) -> dict:
    # 合并所有 system 内容 (top-level + messages 里 system-role) 成唯一一条置 index 0.
    # ajet context_tracker 的 ExtendedMessage.auto_tokenize 遇到非首位的 system 会抛错,
    # 所以必须保证最多一条 system 且在最前.
    sys_parts = [_system_to_text(body.get("system"))]
    messages: list[dict] = []
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        if role == "system":
            sys_parts.append(_content_to_text(msg.get("content")))
            continue
        messages.extend(_content_blocks_to_openai(role, msg.get("content")))
    sys_text = "\n\n".join(p for p in sys_parts if p)
    out_messages = ([{"role": "system", "content": sys_text}] if sys_text else []) + messages

    out: dict[str, Any] = {
        "model": body.get("model", "ajet-model"),
        "messages": out_messages,
        "stream": bool(body.get("stream", False)),
    }
    tools = _convert_tools(body.get("tools"))
    if tools:
        out["tools"] = tools
    tc = _convert_tool_choice(body.get("tool_choice"))
    if tc is not None:
        out["tool_choice"] = tc
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    for k_src, k_dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", None)):
        if k_src in body and body[k_src] is not None and k_dst:
            out[k_dst] = body[k_src]
    if out.get("stream"):
        out["stream_options"] = {"include_usage": True}
    return out


# ───────────────────────────── 响应: OpenAI SSE -> Anthropic SSE ─────────────────

_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def _sse(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def _iter_openai_sse(
    resp: "ClientResponse", model: str
) -> AsyncIterator[bytes]:
    """读上游 OpenAI SSE 流, 产出 Anthropic SSE 字节流."""
    msg_id = f"msg_{int(time.time() * 1000)}"
    started = False
    text_idx: Optional[int] = None        # 已开的 text 块的 anthropic index
    tool_idx_map: dict[int, int] = {}      # openai tool_call.index -> anthropic index
    next_idx = 0
    finish_reason: Optional[str] = None
    usage_in = 0
    usage_out = 0

    def message_start() -> bytes:
        return _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "model": model, "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": usage_in, "output_tokens": 0},
            },
        })

    async for raw in resp.content:
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if chunk.get("usage"):
            u = chunk["usage"]
            usage_in = u.get("prompt_tokens", usage_in)
            usage_out = u.get("completion_tokens", usage_out)

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}

        if not started:
            started = True
            yield message_start()

        # 文本块
        text_piece = delta.get("content")
        if text_piece:
            if text_idx is None:
                text_idx = next_idx
                next_idx += 1
                yield _sse("content_block_start", {
                    "type": "content_block_start", "index": text_idx,
                    "content_block": {"type": "text", "text": ""},
                })
            yield _sse("content_block_delta", {
                "type": "content_block_delta", "index": text_idx,
                "delta": {"type": "text_delta", "text": text_piece},
            })

        # 工具调用块
        for tc in delta.get("tool_calls") or []:
            oi = tc.get("index", 0)
            fn = tc.get("function") or {}
            if oi not in tool_idx_map:
                aidx = next_idx
                next_idx += 1
                tool_idx_map[oi] = aidx
                tname = fn.get("name") or ""
                tid = tc.get("id") or f"toolu_{aidx}_{int(time.time() * 1000)}"
                yield _sse("content_block_start", {
                    "type": "content_block_start", "index": aidx,
                    "content_block": {"type": "tool_use", "id": tid, "name": tname, "input": {}},
                })
            aidx = tool_idx_map[oi]
            args_piece = fn.get("arguments")
            if args_piece:
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": aidx,
                    "delta": {"type": "input_json_delta", "partial_json": args_piece},
                })

        fr = choice.get("finish_reason")
        if fr:
            finish_reason = fr

    # 上游一个 chunk 都没产 (异常) -> 兜底发一个空 text 块, 保证消息结构完整
    if not started:
        yield message_start()
        text_idx = 0
        yield _sse("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })

    # 关掉所有开着的块 (text + tools)
    open_indices = ([text_idx] if text_idx is not None else []) + list(tool_idx_map.values())
    for aidx in sorted(set(open_indices)):
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": aidx})

    stop_reason = _FINISH_TO_STOP.get(finish_reason or "", "end_turn")
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": usage_out},
    })
    yield _sse("message_stop", {"type": "message_stop"})


# ───────────────────────────── aiohttp app ─────────────────────────────────────

def build_translator_app(
    upstream_base_url: str,
    upstream_api_key: str,
    upstream_model: Optional[str] = None,
    timeout: float = 1260.0,
) -> web.Application:
    """构造翻译代理 app: /v1/messages -> upstream /v1/chat/completions."""
    upstream_base_url = upstream_base_url.rstrip("/")
    timeout_obj = ClientTimeout(total=timeout, connect=30, sock_read=timeout)

    async def handle_messages(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400, text="invalid json")

        oai_body = anthropic_to_openai_body(body)
        model_name = upstream_model or oai_body.get("model", "ajet-model")
        is_stream = oai_body.get("stream", False)
        url = f"{upstream_base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {upstream_api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        }

        try:
            sess = ClientSession(timeout=timeout_obj)
            up = await sess.post(url, json=oai_body, headers=headers, auto_decompress=False)
        except Exception as e:
            await sess.close()
            logger.exception("[translator] upstream connect failed: %s", e)
            return web.Response(status=502, text=f"upstream connect failed: {e}")

        if up.status >= 400:
            txt = await up.text()
            await sess.close()
            # 把上游错误包成 Anthropic 风格错误返回, claude code 能识别
            return web.json_response(
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"upstream {up.status}: {txt[:800]}",
                    },
                },
                status=up.status,
            )

        if not is_stream:
            # 非流式: 翻译成 Anthropic message 一次返回
            data = await up.json()
            await sess.close()
            return web.json_response(_openai_completion_to_anthropic(data, model_name))

        # 流式
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)
        try:
            async for chunk in _iter_openai_sse(up, model_name):
                await resp.write(chunk)
            await resp.write_eof()
        except Exception as e:
            logger.exception("[translator] stream error: %s", e)
        finally:
            await sess.close()
        return resp

    async def handle_count_tokens(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            body = {}
        # 粗估: system + messages 字符数 / 4
        n = 0
        n += len(_system_to_text(body.get("system")))
        for m in body.get("messages", []):
            c = m.get("content")
            if isinstance(c, str):
                n += len(c)
            elif isinstance(c, list):
                for b in c:
                    n += len(b.get("text", "")) if isinstance(b, dict) else len(str(b))
        return web.json_response({"input_tokens": max(1, n // 4)})

    app = web.Application()
    app.router.add_post("/v1/messages", handle_messages)
    app.router.add_post("/v1/messages/count_tokens", handle_count_tokens)
    # 兼容带 /beta 前缀或 query 的路径
    return app


def _openai_completion_to_anthropic(data: dict, model: str) -> dict:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict] = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content.append({
            "type": "tool_use", "id": tc.get("id", "toolu_0"),
            "name": fn.get("name", ""), "input": args,
        })
    stop = _FINISH_TO_STOP.get(choice.get("finish_reason") or "", "end_turn")
    usage = data.get("usage") or {}
    return {
        "id": data.get("id", f"msg_{int(time.time()*1000)}"),
        "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": stop, "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        },
    }
