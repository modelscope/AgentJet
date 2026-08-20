# THIS RUNS INSIDE SANDBOX
"""claude code stage runner helper: spawn a tmux-driven claude agent and wait.

相对原片段的修复 (不改 tmux 驱动方式):
  - 补全原片段假设但缺失的 import + update_state_json;
  - await 掉 asyncio.sleep (原片段漏写 await -> 死循环 busy-wait)。
驱动本身仍 100% 走 tmux (TmuxClaudeCodeDriver), 与 claudecode_driver.py 一致。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

# claudecode_driver.py 与本文件同目录 (一起拷进沙盒 /root/cc_driver)。
from claudecode_driver import TmuxClaudeCodeDriver  # noqa: F401  (类型注解用)

STATE_FILE = "/root/cc_data/state.json"

# capture-pane 抓的尾部行数。20~40 之间够看清 claude 此刻屏幕 (权限框 / 空转 /
# 报错 / status line), 又不会把 state.json 撑太大。SSE 转写已进 jsonl, 这里只取
# "肉眼可见的当前屏", 供 host 侧轮询/抓诊断一眼看出卡在哪。
_TMUX_TAIL_LINES = 30
# 去掉 ANSI 转义 (claude TUI 大量带色), 留纯文本。
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# PDF + Read 可能触发某些后端报错, 前置禁掉。
SYSTEM_PROMPT_PREFIX = (
    "⚠ You MUST NOT use the `Read` tool to read PDFs. "
    "Although the tool description says it can deal with PDFs, "
    "it only deals with images and text files. "
)


def _capture_tmux_tail(tmux_target: Optional[str]) -> str:
    """抓 tmux 窗口尾部纯文本 (沙盒内直接调 tmux, 失败静默返回 '')。

    tmux_target 形如 '{hub}:{session_id}' —— 就是 claude TUI 所在窗口。
    capture-pane -S -N 取末 N 行, -p 输出纯文本。每次状态机跳变都抓一次,
    这样 state.json 不光有 'msg', 还带着"那一刻 claude 屏幕长啥样"的直接证据。
    """
    if not tmux_target:
        return ""
    try:
        r = subprocess.run(
            ["tmux", "capture-pane", "-pt", tmux_target,
             "-p", "-S", f"-{_TMUX_TAIL_LINES}"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return ""
    if r.returncode != 0 or not r.stdout:
        return ""
    return _ANSI_RE.sub("", r.stdout).rstrip()


def update_state_json(msg: str, tmux_target: Optional[str] = None) -> None:
    """写 state.json: {ts, msg, tmux_tail}。

    tmux_tail: 顺带抓的 claude TUI 窗口尾部 (末 ~30 行纯文本)。host 侧
    _poll_sandbox_state / _capture_sandbox_state 读 state.json 时就能直接看到
    "卡住那一刻 claude 屏幕是什么", 不用再单独 capture-pane。tmux_target 留空
    (拿不到) 时只写 msg。全程容错: 写盘失败不拖垮状态机。
    """
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump({
                "ts": time.time(),
                "msg": msg,
                "tmux_tail": _capture_tmux_tail(tmux_target),
            }, f, ensure_ascii=False)
    except Exception:
        pass


async def spawn_and_wait_agent(
            driver: TmuxClaudeCodeDriver,
            session_id: str, cwd: Path, flag_root: Path,
            initial_prompt: str, model_config: Path,
            model: Optional[str], timeout: int,
            stage_description: str
        ) -> None:
    """Spawn a TUI claude agent. Mirrors cross/cross_main.py's
    create_new_session call but with cross/-specific flag root + hub."""
    # claude TUI 所在的 tmux 窗口 target = '{hub}:{session_id}' (driver 保证)。
    # 传给 update_state_json, 让每次状态机跳变都顺带抓下 claude 屏幕尾部。
    tmux_target = f"{driver.cfg.hub}:{session_id}"
    try:
        deadline = timeout + time.time()
        update_state_json(f"{stage_description} | create claude code", tmux_target)

        driver.create_new_session({
            "session_id": session_id,
            "cwd": str(cwd),
            "flagRoot": str(flag_root),
            "initialPrompt": SYSTEM_PROMPT_PREFIX + initial_prompt,
            "settingsPath": str(model_config),
            "model": model,
            "useProxy": False,
            "isInitialContextPrompt": False,
        })

        await asyncio.sleep(10)  # wait some 10 sec for ready

        while driver.is_working(session_id) and time.time() < deadline:
            update_state_json(
                f"{stage_description} | waiting claude code finish ({deadline - time.time():.0f}s)",
                tmux_target)
            await asyncio.sleep(30)

    finally:
        update_state_json(f"{stage_description} | terminate claude code", tmux_target)
        driver.terminate_session(session_id)
