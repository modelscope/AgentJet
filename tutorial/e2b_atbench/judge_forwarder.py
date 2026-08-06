#!/usr/bin/env python3
"""judge_forwarder.py — e2b_atbench judge 独立转发进程（常驻，所有 episode 共享）。

替代旧的 _execute_agent 里"每个 episode run_app_in_thread 起一个 judge 透传代理"：
一次启动、固定端口，所有 episode 的 judge 请求都走这一个进程。

链路: 沙盒 claude code (Anthropic /v1/messages) ──▶ 本进程 ──▶ dashscope compatible-mode/v1 (OpenAI, glm-5.2)

本进程复用 anthropic_to_openai_proxy.build_translator_app（Anthropic→OpenAI 翻译），
上游指向 DashScope OpenAI 端点，用 env 里的 key 认证（不硬编码）。

用法（8851 上，先 source venv / 设好 env）:
    nohup python3 judge_forwarder.py > /tmp/judge_forwarder.log 2>&1 &
参数（env）:
    JUDGE_MODEL_SERVER    上游 dashscope base，默认 https://dashscope.aliyuncs.com/compatible-mode/v1
    JUDGE_DASHSCOPE_KEY   dashscope key（Bearer 认证），必填
    E2B_ATBENCH_JUDGE_MODEL  模型名，默认 glm-5.2
    JUDGE_FORWARDER_PORT  监听端口，默认 18005
    JUDGE_FORWARDER_HOST  监听 host，默认 0.0.0.0
"""
from __future__ import annotations

import asyncio
import os
import sys

# 保证能 import tutorial.e2b_atbench.*（与 swarm client 一致的工作目录）
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))  # tutorial 的上两级 = agentjet 仓库根
for _p in (_REPO,):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from aiohttp import web  # noqa: E402

from tutorial.e2b_atbench.anthropic_to_openai_proxy import build_translator_app  # noqa: E402

JUDGE_MODEL_SERVER = os.environ.get(
    "JUDGE_MODEL_SERVER", "https://dashscope.aliyuncs.com/compatible-mode/v1")
JUDGE_DASHSCOPE_KEY = os.environ.get("JUDGE_DASHSCOPE_KEY", "")
JUDGE_MODEL = os.environ.get("E2B_ATBENCH_JUDGE_MODEL", "glm-5.2")
PORT = int(os.environ.get("JUDGE_FORWARDER_PORT", "18005"))
HOST = os.environ.get("JUDGE_FORWARDER_HOST", "0.0.0.0")


async def _serve() -> None:
    assert JUDGE_DASHSCOPE_KEY, "JUDGE_DASHSCOPE_KEY 未设置（judge 转发进程需要 dashscope key）"
    app = build_translator_app(JUDGE_MODEL_SERVER, JUDGE_DASHSCOPE_KEY, JUDGE_MODEL)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    await site.start()
    print(f"[judge_forwarder] listening {HOST}:{PORT} -> {JUDGE_MODEL_SERVER} (model={JUDGE_MODEL})",
          flush=True)
    # 常驻；调用方靠端口探活，不靠 stdout
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(_serve())
