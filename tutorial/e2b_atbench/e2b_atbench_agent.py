# -*- coding: utf-8 -*-
"""e2b ATB coding-agent 的单 episode 执行器 (swarm 模式).

_execute_agent(task, api_baseurl_key) 被 e2b_atbench_swarm_client 在每个 episode
调用. 不再在 episode 内起代理线程 (删掉了 run_app_in_thread 中转层):
  - solver: claude code 直连 swarm interchange (api_baseurl_key.base_url),
    假设 interchange 原生有 Anthropic /v1/messages 接口. 全部 token 按 episode 捕获.
  - judge:  指向独立常驻转发进程 judge_forwarder.py (固定端口) → dashscope glm-5.2,
    防止 reward hacking.
然后复用 coding-agent-material 的 _run_claudecode_in_sandbox 跑完整 solver+judge,
_reward_from_output 解析 verdict -> reward.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys

from ajet.schema.task import Task, WorkflowOutput
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey

# coding-agent-material 在 sys.path (vendored sandbox/generate/utils)
MAT = os.environ.get("E2B_ATBENCH_MATERIAL", "")
if MAT and MAT not in sys.path:
    sys.path.insert(0, MAT)

from tutorial.e2b_atbench.atbench_runtime.sandbox import E2BSandbox                       # noqa: E402
from tutorial.e2b_atbench.atbench_runtime.generate_claudecode import _run_claudecode_in_sandbox, _reward_from_output  # noqa: E402

logger = logging.getLogger("e2b_atbench_agent")

# judge 独立转发进程 (judge_forwarder.py) 的对外地址. 沙盒从 ADAPTER_PUBLIC_HOST 回连.
# 实测 master net0 = 10.29.255.115 (旧默认 10.56.15.66 是错的).
ADAPTER_PUBLIC_HOST = os.environ.get("ADAPTER_PUBLIC_HOST", "10.29.255.115")
JUDGE_FORWARDER_PORT = int(os.environ.get("JUDGE_FORWARDER_PORT", "18005"))
JUDGE_FORWARDER_URL = f"http://{ADAPTER_PUBLIC_HOST}:{JUDGE_FORWARDER_PORT}"

POLICY_MODEL = os.environ.get("E2B_ATBENCH_POLICY_MODEL", "Qwen3.6-35B-A3B")
JUDGE_MODEL = os.environ.get("E2B_ATBENCH_JUDGE_MODEL", "glm-5.2")
# solver 直连 interchange 的认证: 用 api_baseurl_key.api_key (含 episode_uuid).
# judge 转发进程用自己的 dashscope key, 不需要这里传.
EPISODE_TIMEOUT = int(os.environ.get("E2B_ATBENCH_EPISODE_TIMEOUT", "3600"))


async def _run_episode(
    task: Task,
    api_baseurl_key: OpenaiBaseUrlAndApiKey,
) -> WorkflowOutput:
    task_dir = task.metadata.get("task_dir") or task.main_query
    session_id = f"atb_{api_baseurl_key.episode_uuid[:8]}"
    # api_baseurl_key.base_url 的 host 是 swarm server 视角的 localhost, 但沙盒内
    # claude 访问的是沙盒自己的 localhost —— 必须换成 ADAPTER_PUBLIC_HOST (master 可达 IP).
    # 且 claude 的 ANTHROPIC_BASE_URL 语义是根地址 (claude 自己拼 /v1/messages),
    # 而 base_url 已含 /v1 -> 需去掉尾部 /v1, 否则 claude 请求 /v1/v1/messages -> 404.
    # 例: http://localhost:10086/v1 -> http://10.29.255.115:10086
    solver_base_url = re.sub(
        r"/v1$", "",
        re.sub(r"^https?://[^/:]+", f"http://{ADAPTER_PUBLIC_HOST}", api_baseurl_key.base_url),
    )
    assert "/" in solver_base_url, f"unexpected base_url: {api_baseurl_key.base_url!r}"
    async with E2BSandbox(image="qwenpaw", timeout=EPISODE_TIMEOUT) as sb:
        try:
            output = await _run_claudecode_in_sandbox(
                sb,
                task_id=task_dir,
                # solver: claude 直连 interchange (Anthropic /v1/messages)
                solver_base_url=solver_base_url,
                # judge: 独立转发进程 (judge_forwarder.py) → dashscope
                judge_base_url=JUDGE_FORWARDER_URL,
                solver_model=POLICY_MODEL,
                judge_model=JUDGE_MODEL,
                auth_token=api_baseurl_key.api_key,
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
    return asyncio.run(_run_episode(task, api_baseurl_key))
