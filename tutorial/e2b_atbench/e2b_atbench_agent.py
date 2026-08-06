# -*- coding: utf-8 -*-
"""e2b ATB coding-agent 的单 episode 执行器 (swarm 模式).

_execute_agent(task, api_baseurl_key) 被 e2b_atbench_swarm_client 在每个 episode
调用. 它为本 episode 起两个代理:
  - solver 代理 = anthropic->openai 翻译, 指向 swarm interchange (api_baseurl_key),
    这样 claude code 以**当前策略**为大脑解题, 且全部 token 被 swarm 按 episode 捕获.
  - judge  代理 = anthropic 透传, 指向固定 glm-5.2 (经网关隧道), 防止 reward hacking.
然后复用 coding-agent-material 的 _run_claudecode_in_sandbox 跑完整 solver+judge,
_reward_from_output 解析 verdict -> reward.

每个 episode 各自用独立端口 (run_app_in_thread port=0), 线程池并行时互不干扰.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

from ajet.schema.task import Task, WorkflowOutput
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey

# coding-agent-material 在 sys.path (vendored slime/cc_proxy/generate/utils)
MAT = os.environ.get("E2B_ATBENCH_MATERIAL", "")
if MAT and MAT not in sys.path:
    sys.path.insert(0, MAT)

from slime.agent.sandbox import E2BSandbox                       # noqa: E402
from slime.agent.aiohttp_threaded import run_app_in_thread       # noqa: E402
from cc_anthropic_proxy import build_app                          # noqa: E402
from generate_claudecode import _run_claudecode_in_sandbox, _reward_from_output  # noqa: E402
from tutorial.e2b_atbench.anthropic_to_openai_proxy import build_translator_app  # noqa: E402

logger = logging.getLogger("e2b_atbench_agent")

ADAPTER_PUBLIC_HOST = os.environ.get("ADAPTER_PUBLIC_HOST", "10.56.15.66")
JUDGE_MODEL_SERVER = os.environ.get("JUDGE_MODEL_SERVER", "http://localhost:12928")
POLICY_MODEL = os.environ.get("E2B_ATBENCH_POLICY_MODEL", "Qwen3.6-35B-A3B")
JUDGE_MODEL = os.environ.get("E2B_ATBENCH_JUDGE_MODEL", "glm-5.2")
AUTH_TOKEN = os.environ.get(
    "CC_ANTHROPIC_AUTH_TOKEN",
    "sk-wefjoewfewhviuwhoewjfoiwehfiuewhvbdjnasjcoqjfdow",
)
EPISODE_TIMEOUT = int(os.environ.get("E2B_ATBENCH_EPISODE_TIMEOUT", "3600"))


async def _run_episode(
    task: Task,
    api_baseurl_key: OpenaiBaseUrlAndApiKey,
    solver_url: str,
    judge_url: str,
) -> WorkflowOutput:
    task_dir = task.metadata.get("task_dir") or task.main_query
    session_id = f"atb_{api_baseurl_key.episode_uuid[:8]}"
    async with E2BSandbox(image="qwenpaw", timeout=EPISODE_TIMEOUT) as sb:
        try:
            output = await _run_claudecode_in_sandbox(
                sb,
                task_id=task_dir,
                adapter_url="",
                session_id=session_id,
                solver_base_url=solver_url,
                judge_base_url=judge_url,
                solver_model=POLICY_MODEL,
                judge_model=JUDGE_MODEL,
                auth_token=AUTH_TOKEN,
            )
        except Exception as e:
            logger.exception("[e2b_atbench] episode %s harness failed: %s", session_id, e)
            return WorkflowOutput(
                reward=0.0, is_success=False,
                metadata={"error": str(e)[:300], "task_dir": task_dir},
            )
    reward = _reward_from_output(output) or 0.0
    return WorkflowOutput(
        reward=float(reward),
        is_success=bool(reward and reward > 0),
        metadata={
            "verdict": (output.get("verdict", "") or "")[:300],
            "total_steps": output.get("total_steps", 0),
            "task_dir": task_dir,
        },
    )


def _execute_agent(
    task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey
) -> WorkflowOutput:
    ep = api_baseurl_key.episode_uuid
    # solver 翻译代理 -> 策略 interchange (token 按 episode 捕获)
    solver_handle = run_app_in_thread(
        build_translator_app(api_baseurl_key.base_url, api_baseurl_key.api_key, POLICY_MODEL),
        host="0.0.0.0", port=0, thread_name=f"sol-{ep[:6]}",
    )
    # judge 透传代理 -> 固定 glm-5.2 (网关隧道)
    judge_log = os.path.join(
        MAT or ".", "tmp", f"judge-{int(time.time())}-{ep[:6]}.log"
    )
    try:
        os.makedirs(os.path.dirname(judge_log), exist_ok=True)
    except Exception:
        pass
    judge_handle = run_app_in_thread(
        build_app(JUDGE_MODEL_SERVER, log_path=judge_log, model=JUDGE_MODEL),
        host="0.0.0.0", port=0, thread_name=f"jud-{ep[:6]}",
    )
    solver_url = f"http://{ADAPTER_PUBLIC_HOST}:{solver_handle.port}"
    judge_url = f"http://{ADAPTER_PUBLIC_HOST}:{judge_handle.port}"
    logger.info(
        "[e2b_atbench] ep=%s task=%s solver=%s judge=%s",
        ep[:8], task.task_id, solver_url, judge_url,
    )
    try:
        return asyncio.run(_run_episode(task, api_baseurl_key, solver_url, judge_url))
    finally:
        for h in (solver_handle, judge_handle):
            try:
                h.stop()
            except Exception:
                pass
