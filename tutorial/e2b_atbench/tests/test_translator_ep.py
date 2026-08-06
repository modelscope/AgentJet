# -*- coding: utf-8 -*-
"""单 episode 全链路测试: solver 走 translator->网关-openai(Qwen 当 stand-in 策略),
judge 走透传->网关-anthropic(glm-5.2), 复用 _run_claudecode_in_sandbox 跑完整
solver+judge -> reward. 验证 _execute_agent 的整条路径 (含 translator 在 claude code
多轮工具循环里的正确性), 不依赖 swarm.
用法: python -m tutorial.e2b_atbench.test_translator_ep [task_dir]
"""
import asyncio
import os
import sys

HELLO = "/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet2"
MAT = os.environ.get("E2B_ATBENCH_MATERIAL", f"{HELLO}/tmp/coding-agent-material")
for p in (HELLO, MAT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tutorial.e2b_atbench.atbench_runtime.sandbox import E2BSandbox  # noqa: E402
from tutorial.e2b_atbench.atbench_runtime.aiohttp_threaded import run_app_in_thread  # noqa: E402
from tutorial.e2b_atbench.atbench_runtime.generate_claudecode import _run_claudecode_in_sandbox, _reward_from_output  # noqa: E402
from tutorial.e2b_atbench.anthropic_to_openai_proxy import build_translator_app  # noqa: E402

# judge 独立转发进程 (judge_forwarder.py) → dashscope; 无 cc_anthropic_proxy 中转层.
GATEWAY_JUDGE = os.environ.get(
    "JUDGE_MODEL_SERVER", "https://dashscope.aliyuncs.com/compatible-mode/v1")
GATEWAY_OPENAI = os.environ.get("POLICY_OPENAI_URL", GATEWAY_JUDGE)      # solver translator->openai
DUMMY = os.environ.get(
    "CC_ANTHROPIC_AUTH_TOKEN",
    "sk-wefjoewfewhviuwhoewjfoiwehfiuewhvbdjnasjcoqjfdow",
)
HOST = os.environ.get("ADAPTER_PUBLIC_HOST", "10.56.15.66")
POLICY = os.environ.get("E2B_ATBENCH_POLICY_MODEL", "Qwen3.6-35B-A3B")
JUDGE = os.environ.get("E2B_ATBENCH_JUDGE_MODEL", "glm-5.2")
TASK_DIR = sys.argv[1] if len(sys.argv) > 1 else (
    f"{MAT}/0730_ATBV3/Clean_Tasks/v3_pawharness/v3_pawharness__task_2234_result"
)


async def main():
    sol = run_app_in_thread(
        build_translator_app(GATEWAY_OPENAI, DUMMY, POLICY),
        host="0.0.0.0", port=0, thread_name="ep-sol",
    )
    jud = run_app_in_thread(
        build_translator_app(GATEWAY_JUDGE, DUMMY, JUDGE),
        host="0.0.0.0", port=0, thread_name="ep-jud",
    )
    sol_url = f"http://{HOST}:{sol.port}"
    jud_url = f"http://{HOST}:{jud.port}"
    print(f"[ep] task={TASK_DIR}")
    print(f"[ep] solver(translator)->{GATEWAY_OPENAI} {sol_url}")
    print(f"[ep] judge(translator)->{GATEWAY_JUDGE} {jud_url}")
    try:
        async with E2BSandbox(image="qwenpaw", timeout=3600) as sb:
            output = await _run_claudecode_in_sandbox(
                sb, TASK_DIR, "", "transep",
                solver_base_url=sol_url, judge_base_url=jud_url,
                solver_model=POLICY, judge_model=JUDGE, auth_token=DUMMY,
            )
        reward = _reward_from_output(output)
        print(f"[ep] === reward={reward} verdict_head={(output.get('verdict','') or '')[:200]} ===")
        print("[ep] EPISODE_TEST_OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ep] EPISODE_TEST_FAIL: {e}")
    finally:
        try: sol.stop()
        except Exception: pass
        try: jud.stop()
        except Exception: pass


if __name__ == "__main__":
    asyncio.run(main())
