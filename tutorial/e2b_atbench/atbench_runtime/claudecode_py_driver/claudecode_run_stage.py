#!/usr/bin/env python3
# THIS RUNS INSIDE SANDBOX
"""沙盒侧入口: 跑一个 claude-code 阶段 (solver 或 judge)。

由宿主机通过 slime.agent.sandbox.exec_and_wait (setsid + done-marker 轮询) 拉起。
用法:
  python3 claudecode_run_stage.py <stage> <cwd> <settings> <model> <timeout> <prompt_file> <session_id> [flag_root]

stage = solver | judge。cwd = claude 工作目录; settings = settings.json 路径;
prompt_file = 提示词文本文件; 完成判定靠 claude jsonl 的 end_turn (driver.is_working)。
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from claudecode_driver import DriverConfig, TmuxClaudeCodeDriver
from claudecode_helper import spawn_and_wait_agent


def main() -> int:
    if len(sys.argv) < 8:
        print("usage: claudecode_run_stage.py <stage> <cwd> <settings> <model> "
              "<timeout> <prompt_file> <session_id> [flag_root]", file=sys.stderr)
        return 2
    stage = sys.argv[1]
    cwd = sys.argv[2]
    settings = sys.argv[3]
    model = sys.argv[4]
    timeout = int(sys.argv[5])
    prompt_file = sys.argv[6]
    session_id = sys.argv[7]
    flag_root = sys.argv[8] if len(sys.argv) > 8 else cwd

    with open(prompt_file, encoding="utf-8") as f:
        prompt = f.read()

    os.makedirs("/root/cc_data", exist_ok=True)
    cfg = DriverConfig(home="/root", data_path="/root/cc_data")
    driver = TmuxClaudeCodeDriver(cfg)

    t0 = time.time()
    print(f"[run_stage] {stage} start cwd={cwd} model={model} timeout={timeout}s", flush=True)
    asyncio.run(spawn_and_wait_agent(
        driver, session_id, cwd, flag_root, prompt, settings, model, timeout, stage))
    print(f"[run_stage] {stage} finished in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
