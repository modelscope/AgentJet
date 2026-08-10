#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tmux_driver.py — 通用 tmux 驱动 (可靠地建 session/window + 投递命令 + 交付校验)

为什么需要它
------------
手写 bash 里 `tmux send-keys` + split-pane 极易错位 (典型 bug: client 和 server
都打到 :1.0, :1.1 永远空 -> attach 后"什么都不运行"). 本驱动把布局 + 投递 + 校验
封进一个可复用的 Python 对象, 并对每条 send-keys 做交付确认 (capture 回看 + tty
直写兜底), 杜绝"发了但没到".

结构 (用户指定)
--------------
    command_to_run = {
        "<session>": {
            "<window>": [ cmd_1, cmd_2, cmd_3, ... ]   # 顺序 send-keys 到该 window 的 pane 0
        }
    }
每个 window = 单 pane, 命令按序投递 (通常是若干 setup 行 + 最后一行长驻命令).
需要并发的进程 -> 放不同 window (比 split-pane 简单且不会错位).

示例
----
    from tmux_driver import TmuxDriver, TmuxSpec
    spec = {
        "e2b_train": {
            "fwd":    ["source /tmp/e2b_env_new.sh",
                       "cd /mnt/data_cpfs/qingxu.fu/agentjet/tutorial/e2b_atbench",
                       "python judge_forwarder.py"],
            "server": ["source /tmp/e2b_env_new.sh",
                       "cd /mnt/data_cpfs/qingxu.fu/agentjet",
                       "python -m ajet.swarm_cli start --swarm-port=10086"],
            "client": ["source /tmp/e2b_env_new.sh",
                       "cd /mnt/data_cpfs/qingxu.fu/agentjet",
                       "python -m tutorial.e2b_atbench.e2b_atbench_swarm_client"],
        }
    }
    drv = TmuxDriver()
    drv.run(spec)                       # 建 session + 3 个 window, 各投递命令
    drv.wait_for("e2b_train:server", "ENGINE.ROLLING", timeout=900)
    drv.show_status("e2b_train")

CLI / 自检
---------
    python tmux_driver.py --selftest                 # dummy 命令端到端验证 (落地文件)
    python tmux_driver.py --spec spec.json --run     # spec.json = 上面的 dict
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# 底层 tmux 调用
# ──────────────────────────────────────────────────────────────────────────────

class TmuxError(RuntimeError):
    pass


@dataclass
class TmuxDriver:
    """通用 tmux 驱动. 默认运行在本机 (与 tmux server 同机)."""

    tmux_bin: str = "tmux"
    # 可选: 指定 socket (-L name 或 -S path), 避免串到别的 server
    socket_name: Optional[str] = None      # -L <name>
    socket_path: Optional[str] = None      # -S <path>
    # 投递后是否校验 (capture 回看命令是否被 shell 回显)
    verify_send: bool = True
    # 校验失败时的重试次数
    send_retries: int = 3
    # tty 直写兜底开关 (verify 失败 -> 直接写 /dev/pts/N)
    tty_fallback: bool = True
    # 默认开 remain-on-exit: 进程崩了 pane 保留 (attach 能看到输出)
    remain_on_exit: bool = True
    # 建窗前的默认 cwd
    default_cwd: Optional[str] = None
    # 内部: 记录本驱动新建的 session, 以及这些 session 是否已把自带 window 0 复用掉
    _created_sessions: set = field(default_factory=set)
    _first_window_used: set = field(default_factory=set)

    # ── 执行一条 tmux 命令 ──
    def _tmux(self, args: List[str], check: bool = True, timeout: float = 30.0) -> subprocess.CompletedProcess:
        cmd = [self.tmux_bin]
        if self.socket_name:
            cmd += ["-L", self.socket_name]
        if self.socket_path:
            cmd += ["-S", self.socket_path]
        cmd += args
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            raise TmuxError(f"tmux 二进制不存在: {self.tmux_bin}")
        if check and r.returncode != 0:
            raise TmuxError(f"tmux 失败 (rc={r.returncode}): {r.stderr.strip()} | cmd={cmd}")
        return r

    # ──────────────────────────────────────────────────────────────────────────
    # session / window / pane 基元
    # ──────────────────────────────────────────────────────────────────────────

    def has_session(self, session: str) -> bool:
        r = self._tmux(["has-session", "-t", session], check=False)
        return r.returncode == 0

    def ensure_session(self, session: str, cwd: Optional[str] = None) -> None:
        """session 不存在则新建; 已存在则复用. 全程开 remain-on-exit."""
        if self.has_session(session):
            if self.remain_on_exit:
                self._tmux(["set-option", "-t", session, "remain-on-exit", "on"], check=False)
            return
        start_cwd = cwd or self.default_cwd or os.getcwd()
        # -d detached; 给个足够大的终端尺寸, 避免 progress bar 截断
        self._tmux(["new-session", "-d", "-s", session, "-x", "240", "-y", "60",
                    "-c", start_cwd], check=True)
        if self.remain_on_exit:
            self._tmux(["set-option", "-t", session, "remain-on-exit", "on"], check=False)
        self._created_sessions.add(session)  # 标记: 自带 window 0 可被首个命名 window 复用

    def list_windows(self, session: str) -> List[str]:
        r = self._tmux(["list-windows", "-t", session, "-F", "#{window_name}"], check=False)
        if r.returncode != 0:
            return []
        return [w for w in r.stdout.splitlines() if w.strip()]

    def ensure_window(self, session: str, window: str, cwd: Optional[str] = None) -> str:
        """window 不存在则新建; 返回 pane target 'session:window'.

        优化: 若 session 是本驱动新建的且自带的 window 0 还没被复用, 则把 window 0
        重命名为 <window> (避免留一个空 'bash' 窗).
        """
        if window in self.list_windows(session):
            return f"{session}:{window}"
        # 复用新建 session 自带的 window 0
        if session in self._created_sessions and session not in self._first_window_used:
            self._tmux(["rename-window", "-t", f"{session}:0", window], check=True)
            if cwd:
                self._tmux(["send-keys", "-t", f"{session}:{window}", "-l",
                            f"cd {cwd}"], check=True)
                self._tmux(["send-keys", "-t", f"{session}:{window}", "Enter"], check=True)
            self._first_window_used.add(session)
            self._wait_pane_ready(f"{session}:{window}", timeout=10.0)
            return f"{session}:{window}"
        start_cwd = cwd or self.default_cwd or os.getcwd()
        self._tmux(["new-window", "-d", "-t", session, "-n", window, "-c", start_cwd], check=True)
        # 等 pane 的 shell 就绪
        self._wait_pane_ready(f"{session}:{window}", timeout=10.0)
        return f"{session}:{window}"

    def _target(self, session: str, window: str, pane: int = 0) -> str:
        return f"{session}:{window}.{pane}"

    # ──────────────────────────────────────────────────────────────────────────
    # pane 探测
    # ──────────────────────────────────────────────────────────────────────────

    def pane_pid(self, target: str) -> Optional[int]:
        r = self._tmux(["display-message", "-p", "-t", target, "#{pane_pid}"], check=False)
        try:
            return int(r.stdout.strip())
        except (ValueError, IndexError):
            return None

    def pane_tty(self, target: str) -> Optional[str]:
        r = self._tmux(["display-message", "-p", "-t", target, "#{pane_tty}"], check=False)
        tty = r.stdout.strip()
        return tty or None

    def pane_dead(self, target: str) -> bool:
        r = self._tmux(["display-message", "-p", "-t", target, "#{pane_dead}"], check=False)
        return r.stdout.strip() == "1"

    def capture(self, target: str, lines: int = 200) -> str:
        """回看 pane 内容 (含历史). target 可为 'session' / 'session:win' / 'session:win.pane'."""
        r = self._tmux(["capture-pane", "-p", "-S", f"-{lines}", "-t", target], check=False)
        return r.stdout

    def _wait_pane_ready(self, target: str, timeout: float = 10.0) -> bool:
        """等 pane 的 shell 起来 (pane_pid 可用且 capture 有 prompt 迹象)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.pane_pid(target) is not None:
                txt = self.capture(target, lines=20)
                # bash/zsh prompt 迹象
                if any(m in txt for m in ("$", "#", "%", "~")):
                    return True
            time.sleep(0.2)
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # send-keys (核心: 可靠投递 + 校验 + 兜底)
    # ──────────────────────────────────────────────────────────────────────────

    def send_keys(self, target: str, cmd: str, press_enter: bool = True, verify: Optional[bool] = None) -> bool:
        """向 target pane 投递一条命令.

        - 用 `-l` 发字面文本 (避免 tmux 把 ';' '|' 等误当 key 名), 再单独发 Enter.
        - verify=True 时, 投递后 capture 回看, 确认该行被 shell 回显 (输入到达).
        - 校验失败 -> 重试; 仍失败 -> (若 tty_fallback) 直接写 /dev/pts/N 兜底.
        返回是否最终投递成功.
        """
        do_verify = self.verify_send if verify is None else verify
        for attempt in range(1, self.send_retries + 1):
            self._tmux(["send-keys", "-t", target, "-l", cmd], check=True)
            if press_enter:
                self._tmux(["send-keys", "-t", target, "Enter"], check=True)
            if not do_verify:
                return True
            time.sleep(0.35)
            if self._verify_echo(target, cmd):
                return True
            # 重试前清掉可能的半行 (Ctrl-C)
            self._tmux(["send-keys", "-t", target, "C-c"], check=False)
            time.sleep(0.2)
        # 兜底: tty 直写
        if self.tty_fallback:
            if self._tty_write(target, cmd):
                return True
        raise TmuxError(f"send-keys 投递失败 (重试 {self.send_retries} 次仍校验不过): target={target} cmd={cmd!r}")

    def _verify_echo(self, target: str, cmd: str) -> bool:
        """检查 capture 里能否看到刚发的命令文本 (shell 会回显键入的字符)."""
        txt = self.capture(target, lines=40)
        if not txt:
            return False
        # 取命令的一个稳定子串做匹配 (去掉前导空格/换行影响)
        token = cmd.strip()
        if not token:
            return True
        # 用末尾 ~30 字符做指纹, 避免超长命令整行匹配受换行干扰
        needle = token[-30:] if len(token) > 30 else token
        return needle in txt

    def _tty_write(self, target: str, cmd: str) -> bool:
        """兜底: 直接向 pane 的 /dev/pts/N 写入 (绕过 tmux send-keys)."""
        tty = self.pane_tty(target)
        if not tty or not os.path.exists(tty):
            return False
        try:
            with open(tty, "w") as f:
                f.write(cmd + "\n")
                f.flush()
            time.sleep(0.35)
            return self._verify_echo(target, cmd)
        except OSError:
            return False

    def send_lines(self, target: str, cmds: List[str]) -> None:
        """按序投递多条命令到同一 pane (setup 行 + 末行长驻命令)."""
        for c in cmds:
            self.send_keys(target, c)

    # ──────────────────────────────────────────────────────────────────────────
    # 等待 / 状态
    # ──────────────────────────────────────────────────────────────────────────

    def wait_for(self, target: str, needle: str, timeout: float = 300.0, interval: float = 1.0) -> bool:
        """轮询 capture, 直到出现 needle 或超时. needle 可为正则 (用 re.search)."""
        import re
        pat = None
        try:
            pat = re.compile(needle)
        except re.error:
            pat = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            txt = self.capture(target, lines=500)
            if pat is not None:
                if pat.search(txt):
                    return True
            elif needle in txt:
                return True
            time.sleep(interval)
        return False

    def wait_port(self, port: int, timeout: float = 120.0, host: str = "127.0.0.1") -> bool:
        """等本机某 TCP 端口 listen (用于等 server ready 再起 client)."""
        import socket
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return True
            except OSError:
                time.sleep(0.5)
        return False

    def show_status(self, session: str, file=None) -> None:
        """打印 session 下每个 window/pane 的存活 + 最近输出尾部 (排查用)."""
        out = file or sys.stdout
        wins = self.list_windows(session)
        print(f"\n=== [{session}] windows={len(wins)} ===", file=out)
        for w in wins:
            target = f"{session}:{w}"
            dead = self.pane_dead(f"{target}.0")
            pid = self.pane_pid(f"{target}.0")
            tail = "\n".join(l.rstrip() for l in self.capture(target, lines=6).splitlines() if l.strip())[-300:]
            print(f"\n─ window '{w}'  pane_pid={pid}  dead={dead}", file=out)
            print(f"  tail: {tail!r}", file=out)

    # ──────────────────────────────────────────────────────────────────────────
    # 顶层: 跑一个 spec
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, spec: "TmuxSpec", ensure_only: bool = False) -> None:
        """spec = {session: {window: [cmd, ...]}}.

        - 建 session (若缺), 建 window (若缺).
        - 每个 window 的 pane 0 按序投递命令.
        - ensure_only=True 时只建布局不投递命令.
        """
        for session, windows in spec.items():
            self.ensure_session(session)
            for window, cmds in windows.items():
                self.ensure_window(session, window)
                if ensure_only:
                    continue
                if cmds:
                    target = self._target(session, window, 0)
                    print(f"[tmux_driver] -> {target}: {len(cmds)} cmds", flush=True)
                    self.send_lines(target, cmds)

    # ──────────────────────────────────────────────────────────────────────────
    # 自检 (dummy 命令端到端验证)
    # ──────────────────────────────────────────────────────────────────────────

    def selftest(self) -> bool:
        """用 dummy 命令验证 send-keys 真能落地 (touch 文件, 绕过 capture 的迷惑)."""
        session = "drv_selftest"
        proof = f"/tmp/drv_proof_{os.getpid()}"
        try:
            self._tmux(["kill-session", "-t", session], check=False)
        except TmuxError:
            pass
        self.ensure_session(session)
        self.ensure_window(session, "w0")
        target = self._target(session, "w0", 0)
        ok_send = True
        try:
            self.send_keys(target, f"rm -f {proof}")
            self.send_keys(target, f"touch {proof}")
        except TmuxError as e:
            ok_send = False
            print(f"[selftest] send-keys 投递异常: {e}")
        # 给文件落地一点时间
        time.sleep(0.8)
        file_ok = os.path.exists(proof)
        echo_seen = self._verify_echo(target, f"touch {proof}")
        tty = self.pane_tty(target)
        cap_tail = self.capture(target, lines=10)
        print(f"[selftest] send_keys_ok={ok_send}  file_created={file_ok}  echo_seen={echo_seen}  tty={tty}")
        print(f"[selftest] capture_tail={cap_tail.splitlines()[-3:]!r}")
        # 清理
        self._tmux(["kill-session", "-t", session], check=False)
        try:
            os.remove(proof)
        except OSError:
            pass
        verdict = ok_send and file_ok
        print(f"[selftest] VERDICT: {'PASS ✅' if verdict else 'FAIL ❌'}")
        return verdict


TmuxSpec = Dict[str, Dict[str, List[str]]]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_spec(path: str) -> TmuxSpec:
    with open(path) as f:
        return json.load(f)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="通用 tmux 驱动")
    p.add_argument("--tmux", default="tmux", help="tmux 二进制路径")
    p.add_argument("-L", "--socket-name", default=None, help="tmux -L socket 名")
    p.add_argument("--no-verify", action="store_true", help="关闭 send 校验")
    p.add_argument("--no-tty-fallback", action="store_true", help="关闭 tty 直写兜底")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--selftest", action="store_true", help="dummy 命令端到端自检")
    g.add_argument("--spec", help="JSON spec 文件: {session: {window: [cmds]}}")
    p.add_argument("--run", action="store_true", help="配合 --spec: 投递命令")
    p.add_argument("--ensure-only", action="store_true", help="配合 --spec: 只建布局")
    p.add_argument("--status", help="打印某 session 状态")
    args = p.parse_args(argv)

    drv = TmuxDriver(
        tmux_bin=args.tmux,
        socket_name=args.socket_name,
        verify_send=not args.no_verify,
        tty_fallback=not args.no_tty_fallback,
    )

    if args.selftest:
        return 0 if drv.selftest() else 1

    if args.status:
        drv.show_status(args.status)
        return 0

    if args.spec:
        spec = _load_spec(args.spec)
        drv.run(spec, ensure_only=args.ensure_only)
        if args.run:
            print("[tmux_driver] 已投递. attach 查看: tmux attach -t <session>")
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
