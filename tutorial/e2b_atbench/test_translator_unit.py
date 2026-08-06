# -*- coding: utf-8 -*-
"""翻译代理快速单元测试: 起 translator -> 网关(Qwen3.6-35B-A3B 当 stand-in 策略),
发 Anthropic /v1/messages 流式请求 (纯文本 + 带工具), 验证返回的 Anthropic SSE 事件
结构正确 (message_start/content_block_*/message_delta/message_stop, tool_use 块).
不依赖 swarm, ~30s. 用法: python -m tutorial.e2b_atbench.test_translator_unit
"""
import asyncio
import os
import sys

import aiohttp

HELLO = "/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet2"
MAT = os.environ.get("E2B_ATBENCH_MATERIAL", f"{HELLO}/tmp/coding-agent-material")
for p in (HELLO, MAT):
    if p not in sys.path:
        sys.path.insert(0, p)

from slime.agent.aiohttp_threaded import run_app_in_thread  # noqa: E402
from tutorial.e2b_atbench.anthropic_to_openai_proxy import build_translator_app  # noqa: E402

GATEWAY = os.environ.get("JUDGE_MODEL_SERVER", "http://localhost:12928")
DUMMY = os.environ.get(
    "CC_ANTHROPIC_AUTH_TOKEN",
    "sk-wefjoewfewhviuwhoewjfoiwehfiuewhvbdjnasjcoqjfdow",
)
POLICY = os.environ.get("E2B_ATBENCH_POLICY_MODEL", "Qwen3.6-35B-A3B")


async def consume_stream(url, body, timeout=120):
    events = []
    deltas = []
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=body, timeout=timeout) as r:
            print(f"  http_status={r.status}")
            if r.status != 200:
                txt = await r.text()
                print(f"  ERROR_BODY: {txt[:500]}")
                return events, deltas
            async for raw in r.content:
                line = raw.decode("utf-8", errors="ignore").rstrip()
                if line.startswith("event:"):
                    events.append(line[len("event:"):].strip())
                elif line.startswith("data:"):
                    try:
                        d = line[5:].strip()
                        if d and d != "[DONE]":
                            obj = __import__("json").loads(d)
                            if obj.get("type") == "content_block_delta":
                                deltas.append(obj.get("delta", {}))
                    except Exception:
                        pass
    return events, deltas


async def main():
    app = build_translator_app(GATEWAY, DUMMY, POLICY)
    h = run_app_in_thread(app, host="0.0.0.0", port=0, thread_name="trans-unit")
    url = f"http://127.0.0.1:{h.port}/v1/messages"
    print(f"[unit] translator -> {GATEWAY} (model={POLICY}) at {url}")
    ok = True
    try:
        # ── test 1: 纯文本流式 ──
        print("[unit] test1: plain text stream")
        body1 = {
            "model": POLICY, "max_tokens": 48, "stream": True,
            "system": "You are terse.",
            "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        }
        ev1, dl1 = await consume_stream(url, body1)
        print(f"  events={ev1}")
        text_pieces = [d.get("text", "") for d in dl1 if d.get("type") == "text_delta"]
        print(f"  text={''.join(text_pieces)!r}")
        if not ("message_start" in ev1 and "message_stop" in ev1 and "content_block_delta" in ev1):
            print("  FAIL test1: missing core events"); ok = False
        else:
            print("  PASS test1")

        # ── test 2: 工具调用 (tool_use 流式) ──
        print("[unit] test2: tool_use stream")
        body2 = {
            "model": POLICY, "max_tokens": 128, "stream": True,
            "system": "Use the get_weather tool for the user's question.",
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
            "tools": [{
                "name": "get_weather", "description": "Get current weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }
        ev2, dl2 = await consume_stream(url, body2, timeout=120)
        print(f"  events={ev2}")
        input_pieces = [d.get("partial_json", "") for d in dl2 if d.get("type") == "input_json_delta"]
        print(f"  tool_input_json={''.join(input_pieces)!r}")
        if "content_block_start" not in ev2 or not input_pieces:
            print("  FAIL test2: no tool_use block / input_json_delta"); ok = False
        else:
            print("  PASS test2")

        # ── test 3: count_tokens 端点 ──
        print("[unit] test3: count_tokens")
        async with aiohttp.ClientSession() as s:
            async with s.post(f"http://127.0.0.1:{h.port}/v1/messages/count_tokens",
                              json={"messages": [{"role": "user", "content": "hello world"}]}) as r:
                cj = await r.json()
                print(f"  count_tokens={cj}")
                if "input_tokens" not in cj:
                    print("  FAIL test3"); ok = False
                else:
                    print("  PASS test3")

    finally:
        h.stop()

    print("=" * 50)
    print("UNIT_TEST_OK" if ok else "UNIT_TEST_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
