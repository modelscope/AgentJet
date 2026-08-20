# THIS RUN INSIDE SANDBOX


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
claude_code.py - independent, tmux-based Claude Code driver (Python port).

A faithful, behaviorally-equivalent Python port of the Mobius Node backend's
`/app/mobius/backend/agents/tmux-claude-code.js` (and its dependency chain:
base.js, tmux-operation-log.js, tmux_utils.js, services/jsonl-watcher.ts,
services/mobius-jsonl.ts, services/agent-prompt-events.ts, utils/session-flags.ts).

It is INDEPENDENT: it does not import or require the Mobius Node backend, does
not share Mobius's `hub-runtime.json` (it keeps its own runtime/archive files),
and does not touch the Mobius SQLite DB. Multiple processes can each run an
instance without corrupting each other's state (give each its own hub name /
data_path, or just rely on the driver-specific defaults).

Design notes (JS -> Python):
  * JS is async (Promises + event loop); this port is a synchronous,
    threading-based core. per-session mutual exclusion -> threading.RLock;
    background jsonl tailing -> daemon threads; fire-and-forget danger-permission
    self-heal -> daemon thread; non-blocking waits -> time.sleep. The public API
    is synchronous, which is the natural fit for the threaded orchestrator
    scripts that consume this driver (e.g. solve_tasks.py's ThreadPoolExecutor).
  * tmux is driven via `subprocess.run(['tmux', ...])`, mirroring JS's
    spawnSync('tmux'). shellQuote -> shell_quote (simple single-quote form used
    by the JS driver for --model/--settings). prompt text is fed via
    `tmux load-buffer - <stdin>` + `paste-buffer -p` (bracketed paste), never on
    the command line.
  * Live agent output is tailed from ~/.claude/projects/<enc-cwd>/<uuid>.jsonl
    (the claude TUI's own transcript) plus a .mobius.jsonl sidecar that records
    the prompts WE pasted (claude's TUI jsonl does not always record pasted user
    text as a user message). History = merge of both, ordered by timestamp.
  * Task completion / failure is signaled by flag files under
    <flag_root>/<HIDDEN_FOLDER>/flags/<session_id>/{running,failed}.flag, exactly
    like the JS driver / session-flags.ts. The agent is instructed (out of band,
    by the caller's prompt) to delete running.flag when it finishes.

Public API (TmuxClaudeCodeDriver):
  create_new_session(opts) -> {session_id, agent_session_id, jsonl_path, started_at}
  no_pause_current_and_queue_query_at_session(opts) -> None   # "send": queue a prompt, (re)spawn if dead
  pause_current_and_resume_from_session(opts) -> None         # "stop"/"pause": C-c x3, optional new prompt
  terminate_session(session_id) -> {session_id, killed, was_working}
  is_alive(session_id) -> bool
  is_working(session_id) -> bool
  is_job_goal_accomplished(session_id) -> bool
  is_failed(session_id) -> bool
  list_sessions() -> list[dict]
  real_time_info(session_id) -> str
  get_history(session_id, opts=None) -> dict
  subscribe_raw(session_id, listener) -> unsubscribe           # live tail (shared watcher)
  subscribe_raw_from(session_id, sentinel, listener) -> unsubscribe  # live tail from a sentinel
  get_session_title(session_id) -> str | None
  get_recent_error(session_id) -> None
  start() / shutdown()

A CLI mirroring the JS `test_tmux_claude_code` wrapper is exposed under
`python -m claude_code` / `./claude_code.py` for manual integration testing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _env_path(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


@dataclass
class DriverConfig:
    """Runtime configuration for TmuxClaudeCodeDriver.

    Defaults are resolved at __post_init__ from the environment so the driver is
    independent of (but consistent with) Mobius's config.js conventions.
    """
    # tmux hub session name. Default is driver-specific (NOT the Mobius
    # 'imac_claude_code_agent_hub') so two processes don't fight over one hub.
    hub: str = "claude_code_agent_hub"
    # Persistent state root. runtime/archive/log files live under here.
    data_path: str = ""
    # Hidden folder name used for flag dirs (matches Mobius HIDDEN_FOLDER_NAME).
    hidden_folder: str = ""
    # Live / all-time session->jsonl mapping files (driver-specific names).
    runtime_file: str = ""
    archive_file: str = ""
    # Diagnostic log of every tmux command issued (best-effort).
    tmux_log_file: str = ""
    # Home dir (claude config + projects live under ~/.claude).
    home: str = ""
    # Override Claude Code's config dir (default ~/.claude). When set, agent
    # transcripts land under <claude_config_dir>/projects/<enc-cwd>/ instead
    # of ~/.claude/projects/<enc-cwd>/, and the spawned claude process is
    # given CLAUDE_CONFIG_DIR=<this>. Empty => use ~/.claude (default).
    claude_config_dir: str = ""
    # Proxy prerequisites (only used when use_proxy=True).
    proxy_envs: str = ""
    proxy_conf: str = ""
    # Amend instruction pasted into the danger-permission box on self-heal
    # (Tab -> paste -> Enter). Default tells the agent to retry an rm with mv.
    # Override via the CLAUDE_CODE_DANGER_AMEND_MSG env var; an empty string
    # skips the paste (a bare Enter then denies the highlighted '2. No').
    danger_amend_msg: str = ""

    def __post_init__(self) -> None:
        if not self.data_path:
            self.data_path = _env_path("MOBIUS_DATA_PATH", "/data")
        if not self.hidden_folder:
            self.hidden_folder = _env_path("MOBIUS_HIDDEN_FOLDER_NAME", ".mobius")
        if not self.home:
            self.home = os.path.expanduser("~")
        if not self.runtime_file:
            self.runtime_file = os.path.join(self.data_path, "claude-code-driver-runtime.json")
        if not self.archive_file:
            self.archive_file = os.path.join(self.data_path, "claude-code-driver-archive.json")
        if not self.tmux_log_file:
            self.tmux_log_file = os.path.join(self.data_path, "logs", "tmux-operation.log")
        if not self.proxy_envs:
            self.proxy_envs = os.path.join(self.home, "proxy_envs.bash")
        if not self.proxy_conf:
            self.proxy_conf = os.path.join(self.home, "proxy_claude.conf")
        if not self.danger_amend_msg:
            self.danger_amend_msg = (os.environ.get("CLAUDE_CODE_DANGER_AMEND_MSG")
                                     or "use mv instead of rm")


# --------------------------------------------------------------------------- #
# Constants (ported verbatim from tmux-claude-code.js)
# --------------------------------------------------------------------------- #
READY_POLL_MS = 250
READY_TIMEOUT_MS = 25000
READY_SENTINEL = "bypass permissions on"

# First-contact "trust this folder" dialog. --dangerously-skip-permissions does
# NOT skip it; cwd is a fresh per-task workspace, so auto-pick the default "Yes".
TRUST_PROMPT_SENTINELS = [
    "trust this folder",
    "Is this a project you created or one you trust",
    "Do you trust the files",
]
TRUST_PRESS_INTERVAL_MS = 1500

# One-shot onboarding dialogs (theme picker, welcome screen) that block ready.
ONBOARDING_PROMPT_SENTINELS = [
    "Choose the text style",
    "Let's get started",
    "Welcome to Claude Code",
]
ONBOARDING_PRESS_INTERVAL_MS = 1500

# "Detected a custom API key in your environment" -> press "1" (Yes, use it).
API_KEY_PROMPT_SENTINELS = [
    "Detected a custom API key in your environment",
    "Do you want to use this API key",
]
API_KEY_PRESS_INTERVAL_MS = 1500

# "WARNING: Claude Code running in Bypass Permissions mode" one-shot confirm.
# Default option is "1. No, exit"; must press "2" + Enter to accept.
BYPASS_WARN_SENTINELS = [
    "WARNING: Claude Code running in Bypass Permissions mode",
    "Yes, I accept",
]
BYPASS_WARN_INTERVAL_MS = 1500

# paste landing probe + fallback
PASTE_PROBE_TIMEOUT_MS = 8000
PASTE_PROBE_INTERVAL_MS = 200
PASTE_SLEEP_BASE_MS = 800
PASTE_SLEEP_MAX_MS = 5000
# bracketed-paste (-p) is atomic; resubmit Enter N times because the TUI
# occasionally swallows the first Enter when switching input modes.
SUBMIT_ENTER_ATTEMPTS = 3
SUBMIT_ENTER_INTERVAL_MS = 500
INITIAL_CONTEXT_DELAY_MS = 5000
INITIAL_CONTEXT_GREETING_CHOICES = ["hello", "greeting", "are you there", "good day"]

# list-windows result cache (status-query path only). /status polls isAlive +
# isWorking + listSessions repeatedly; caching drops per-poll spawnSync count.
LIST_WINDOWS_TTL_MS = 3 * 1000
# isWorking reverse-scans a tail window of the transcript. Must be much larger
# than a single record: context-injection user entries and long assistant
# outputs can be 30KB+ on one line; a 16KB window would get pushed past the only
# user marker and misjudge "working" as false for minutes.
CLAUDE_WORKING_TAIL_BYTES = 256 * 1024
# capture-pane tail plain-text cache (5s TTL). Shared by isWorking fallback and
# realTimeInfo so they don't both spawn capture-pane.
PANE_TAIL_TTL_MS = 5 * 1000

# danger-permission self-heal throttle: one heal at a time per session, and the
# same warning text won't re-fire within the cooldown.
DANGER_HEAL_COOLDOWN_MS = 30 * 1000

# claude TUI status line anchor: a parenthesized group starting with an elapsed
# time, e.g. "(6m 36s · ↓ 20.0k tokens · thinking more)" or "(29s · thinking)".
CLAUDE_STATUS_LINE_RE = re.compile(r"\(\d+(?:s|m\s+\d+s|h\s+\d+m\s+\d+s)[^()]*\)")
# "Waiting for N background agents to finish" (N>=1). The JSONL often ends with
# end_turn when sub-agents are fire-and-forget, so this TUI state is invisible to
# the jsonl scan; isWorking falls back to capture-pane when jsonl looks idle.
CLAUDE_BG_AGENTS_WAITING_RE = re.compile(
    r"Waiting\s+for\s+[1-9]\d*\s+background\s+agents?\s+to\s+finish", re.IGNORECASE
)
# A background subagent still mid-run in the TUI: a "◯ general-purpose <desc> Xm"
# (or "Xm Ys") row — the live elapsed timer distinguishes a running subagent from
# a finished one (finished rows drop the timer / show "Done"). Like the
# "Waiting for N background agents" line, this state is invisible to the jsonl
# scan (the main transcript ends in end_turn while the subagent keeps running),
# so is_working's capture-pane fallback must detect it to avoid killing the
# session mid-subagent.
CLAUDE_BG_SUBAGENT_RUNNING_RE = re.compile(
    r"◯\s+(?:general-purpose|[A-Za-z][\w-]*)\s+\S[^\n]*?\s+\d+m(?:\s+\d+s)?\s*$",
    re.IGNORECASE,
)
# Dangerous-operation permission box. Even under --dangerously-skip-permissions,
# claude still confirms destructive ops, e.g.:
#   "Dangerous rm operation on possibly-empty variable path: $EWS/*"
#   "Dangerous rm -rf operation on working directory or its ancestor: ..."
# Broad match "Dangerous <X> operation on <Y>" catches both shapes; the strong
# co-presence check in detect_danger_permission ("Do you want to proceed?" +
# "Esc to cancel") keeps historical screen residue from false-firing.
CLAUDE_DANGER_OPERATION_RE = re.compile(
    r"Dangerous\s+\S[^\n]*?operation\s+on\s+[^\n]+",
    re.IGNORECASE,
)
# Self-heal action = AMEND the command in place (Tab -> paste feedback -> Enter),
# telling the agent to retry with a safer approach (default: "use mv instead of
# rm") rather than just cancelling. The feedback text is configurable via
# DriverConfig.danger_amend_msg / CLAUDE_CODE_DANGER_AMEND_MSG.
DANGER_AMEND_TAB_SETTLE_MS = 500
DANGER_AMEND_PASTE_SETTLE_MS = 300

DEFAULT_MAX_LINES = 10000
DEFAULT_HISTORY_TAIL = 200
MAX_HISTORY_FETCH = 5000
MOBIUS_JSONL_VERSION = 1
# jsonl-watcher size thresholds.
EXACT_TOTAL_MAX_BYTES = 16 * 1024 * 1024
TAIL_CHUNK_BYTES = 256 * 1024
TAIL_MAX_BYTES = 16 * 1024 * 1024

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sleep_ms(ms: float) -> None:
    time.sleep(ms / 1000.0)


# --------------------------------------------------------------------------- #
# tmux operation log + shell quoting (port of tmux-operation-log.js)
# --------------------------------------------------------------------------- #
@dataclass
class TmuxResult:
    status: int
    stdout: str
    stderr: str


_LOG_LOCK = threading.Lock()
_LOG_WARNED = {"tmux": False, "log": False}


def _single_quote(value: Any) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def _bash_ansi_quote(value: Any) -> str:
    s = str(value)
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    s = s.replace("\f", "\\f").replace("\v", "\\v").replace("\x1b", "\\e")
    s = re.sub(r"[\x00-\x08\x0e-\x1a\x1c-\x1f\x7f]",
               lambda ch: f"\\x{ord(ch.group(0)):02x}", s)
    return f"$'{s}'"


def shell_quote(value: Any) -> str:
    """Port of tmux-claude-code.js's shellQuote (simple single-quote form used
    for --model / --settings args). Everything gets single-quoted with embedded
    quotes escaped."""
    return _single_quote(value)


def _log_shell_quote(value: Any) -> str:
    """Port of tmux-operation-log.js shellQuote (fast path for safe tokens, ANSI
    fallback for control chars) - used only to render a log line for tmux ops."""
    s = str(value)
    if s and re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", s):
        return s
    if re.search(r"[\x00-\x1f\x7f]", s):
        return _bash_ansi_quote(s)
    return _single_quote(s)


def _tmux_command_string(args: List[str], input_text: Optional[str] = None) -> str:
    command = " ".join(["tmux", *[_log_shell_quote(a) for a in args]])
    if input_text is None:
        return command
    return f"printf %s {_bash_ansi_quote(input_text)} | {command}"


def _should_record(args: List[str]) -> bool:
    """Audit-log filter. Only NON-IDEMPOTENT tmux operations are recorded
    (capture-pane, list-windows, has-session, display-message, etc. are
    excluded — they don't mutate state and would flood the log under
    polling). The allowlist covers every state-mutating op used by the
    driver; ops not in the list are conservatively dropped so we never
    accidentally log a probe."""
    if not args:
        return False
    return args[0] in _NON_IDEMPOTENT_TMUX_OPS


# State-mutating tmux subcommands. Anything not in this set is treated as
# read-only and excluded from the audit log.
_NON_IDEMPOTENT_TMUX_OPS = frozenset({
    # session / window lifecycle
    "new-session", "kill-session", "rename-session",
    "new-window", "kill-window", "rename-window",
    "move-window", "link-window", "unlink-window",
    # pane manipulation
    "kill-pane", "respawn-pane", "swap-pane", "swap-window",
    "clear-history", "clear-pane",
    # input / keys
    "send-keys",
    # paste-buffer flow
    "load-buffer", "paste-buffer", "delete-buffer",
    # options / environment (driver doesn't currently use these, but
    # include for forward-compat with any future config writes)
    "set-option", "set-window-option", "set-environment",
    "source-file",
})


class TmuxOps:
    """tmux subprocess wrapper + operation logging, scoped to a log file.

    Port of tmux-operation-log.js's { tmux, log, recordTmuxCommand }. Logging is
    best-effort: a failed append warns once and then stays silent.
    """

    def __init__(self, log_file: str) -> None:
        self.log_file = log_file

    def _record(self, args: List[str], input_text: Optional[str] = None) -> None:
        if not _should_record(args):
            return
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(_tmux_command_string(args, input_text) + "\n")
        except OSError as e:
            if not _LOG_WARNED["tmux"]:
                _LOG_WARNED["tmux"] = True
                print(f"[tmux-operation-log] append failed ({self.log_file}): {e}",
                      file=sys.stderr)

    def tmux(self, args: List[str], input_text: Optional[str] = None,
             timeout: Optional[float] = None) -> TmuxResult:
        self._record(args, input_text)
        try:
            r = subprocess.run(
                ["tmux", *args],
                input=input_text,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return TmuxResult(r.returncode, r.stdout or "", r.stderr or "")
        except FileNotFoundError:
            return TmuxResult(127, "", "tmux binary not found")
        except subprocess.TimeoutExpired:
            return TmuxResult(124, "", "tmux command timed out")

    def log(self, *args: Any) -> None:
        msg = " ".join(str(a) for a in args)
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError as e:
            if not _LOG_WARNED["log"]:
                _LOG_WARNED["log"] = True
                print(f"[tmux-operation-log] append failed ({self.log_file}): {e}",
                      file=sys.stderr)
        print(msg, flush=True)


def take_tmux_window_text(ops: TmuxOps, target: str,
                          capture_head_and_tail_line: int = 100) -> str:
    """Port of tmux_utils.take_tmux_window_text: head + tail capture of a pane."""
    tail = ops.tmux(["capture-pane", "-pt", target, "-p",
                     "-S", f"-{capture_head_and_tail_line}"])
    hs = ops.tmux(["display-message", "-p", "-t", target, "#{history_size}"])
    try:
        hs_n = int((hs.stdout or "").strip())
    except ValueError:
        hs_n = 0
    head = TmuxResult(1, "", "")
    if hs_n > capture_head_and_tail_line * 2:
        head = ops.tmux(["capture-pane", "-pt", target, "-p",
                         "-S", f"-{hs_n}",
                         "-E", str(-hs_n + capture_head_and_tail_line - 1)])
    parts = []
    if head.status == 0 and head.stdout:
        parts.append(head.stdout)
    if tail.status == 0 and tail.stdout:
        parts.append(tail.stdout)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# session-flags (port of utils/session-flags.ts)
# --------------------------------------------------------------------------- #
def flag_dir_of(root: str, session_id: str, hidden_folder: str) -> str:
    return os.path.join(os.path.abspath(root), hidden_folder, "flags", session_id)


def running_flag_path_of(root: str, session_id: str, hidden_folder: str) -> str:
    return os.path.join(flag_dir_of(root, session_id, hidden_folder), "running.flag")


def failed_flag_path_of(root: str, session_id: str, hidden_folder: str) -> str:
    return os.path.join(flag_dir_of(root, session_id, hidden_folder), "failed.flag")


def _encode_flag_value(value: Any) -> str:
    return re.sub(r"\r?\n", "\\n", str(value if value is not None else ""))[:2000]


def _decode_flag_value(value: Any) -> str:
    return str(value if value is not None else "").replace("\\n", "\n")


def _parse_flag_body(body: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in str(body or "").split("\n"):
        i = line.find("=")
        if i < 0:
            continue
        out[line[:i]] = _decode_flag_value(line[i + 1:])
    return out


def read_running_flag(root: Optional[str], session_id: Optional[str],
                      hidden_folder: str) -> Optional[Dict[str, str]]:
    if not root or not session_id:
        return None
    try:
        with open(running_flag_path_of(root, session_id, hidden_folder),
                  "r", encoding="utf-8") as f:
            return _parse_flag_body(f.read())
    except OSError:
        return None


def write_running_flag(root: Optional[str], session_id: Optional[str],
                       hidden_folder: str, fields: Optional[Dict[str, Any]] = None,
                       pid: Optional[int] = None) -> bool:
    if not root or not session_id:
        return False
    fields = fields or {}
    os.makedirs(flag_dir_of(root, session_id, hidden_folder), exist_ok=True)
    existing = read_running_flag(root, session_id, hidden_folder)
    started_at = (existing or {}).get("startedAt") or datetime.utcnow().isoformat() + "Z"
    run_id = (existing or {}).get("runId") or f"{session_id}:{started_at}"
    body: Dict[str, str] = {
        "session": session_id,
        "runId": run_id,
        "pid": str(pid if pid is not None else os.getpid()),
        "startedAt": started_at,
    }
    for k, v in fields.items():
        body[k] = "" if v is None else str(v)
    lines = [f"{k}={_encode_flag_value(v)}" for k, v in body.items()
             if v is not None and v != ""]
    with open(running_flag_path_of(root, session_id, hidden_folder),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        remove_failed_flag(root, session_id, hidden_folder)
    except OSError as e:
        print(f"[session-flags] remove stale failed.flag failed ({session_id}): {e}",
              file=sys.stderr)
    return True


def safe_write_running_flag(root: Optional[str], session_id: Optional[str],
                            hidden_folder: str, fields: Optional[Dict[str, Any]] = None,
                            label: str = "claude-code-driver") -> bool:
    try:
        return write_running_flag(root, session_id, hidden_folder, fields)
    except OSError as e:
        print(f"[{label}] write running.flag failed ({session_id}): {e}",
              file=sys.stderr)
        return False


def remove_running_flag(root: Optional[str], session_id: Optional[str],
                        hidden_folder: str) -> bool:
    if not root or not session_id:
        return False
    try:
        os.remove(running_flag_path_of(root, session_id, hidden_folder))
    except FileNotFoundError:
        pass
    return True


def safe_remove_running_flag(root: Optional[str], session_id: Optional[str],
                             hidden_folder: str,
                             label: str = "claude-code-driver") -> bool:
    try:
        return remove_running_flag(root, session_id, hidden_folder)
    except OSError as e:
        print(f"[{label}] remove running.flag failed ({session_id}): {e}",
              file=sys.stderr)
        return False


def remove_failed_flag(root: Optional[str], session_id: Optional[str],
                       hidden_folder: str) -> bool:
    if not root or not session_id:
        return False
    try:
        os.remove(failed_flag_path_of(root, session_id, hidden_folder))
    except FileNotFoundError:
        pass
    return True


def safe_remove_flag_dir(root: Optional[str], session_id: Optional[str],
                         hidden_folder: str,
                         label: str = "claude-code-driver") -> bool:
    if not root or not session_id:
        return False
    try:
        shutil.rmtree(flag_dir_of(root, session_id, hidden_folder),
                      ignore_errors=True)
        return True
    except OSError as e:
        print(f"[{label}] remove flag dir failed ({session_id}): {e}",
              file=sys.stderr)
        return False


# --------------------------------------------------------------------------- #
# jsonl-watcher (port of services/jsonl-watcher.ts: read_all + watch)
# --------------------------------------------------------------------------- #
def _count_newlines(buf: bytes) -> int:
    return buf.count(b"\n")


def _parse_lines(lines: List[str]) -> List[Any]:
    out = []
    for line in lines:
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _read_full(path: str, max_lines: int, tail_count: int = 0) -> Dict[str, Any]:
    with open(path, "rb") as f:
        buf = f.read()
    size = len(buf)
    lines = [ln for ln in buf.decode("utf-8", errors="replace").split("\n") if ln]
    total = len(lines)
    if tail_count > 0:
        effective = min(max_lines, tail_count) if max_lines > 0 else tail_count
    else:
        effective = max_lines
    slc = lines[-effective:] if effective > 0 else []
    return {
        "entries": _parse_lines(slc),
        "total": total,
        "totalApproximate": False,
        "truncated": total > len(slc),
        "size": size,
    }


def _read_tail_window(path: str, max_lines: int, size: int, chunk_size: int,
                      max_tail_bytes: int, tail_count: int = 0) -> Dict[str, Any]:
    if tail_count > 0:
        effective = min(max_lines, tail_count) if max_lines > 0 else tail_count
    else:
        effective = max_lines
    if effective <= 0:
        return {"entries": [], "total": 0, "totalApproximate": size > 0,
                "truncated": size > 0, "size": size, "scannedBytes": 0}

    chunks: List[bytes] = []
    position = size
    scanned = 0
    newline_count = 0
    try:
        with open(path, "rb") as f:
            while position > 0 and scanned < max_tail_bytes and newline_count <= effective:
                length = min(chunk_size, position, max_tail_bytes - scanned)
                position -= length
                f.seek(position)
                read_buf = f.read(length)
                if not read_buf:
                    break
                chunks.insert(0, read_buf)
                scanned += len(read_buf)
                newline_count += _count_newlines(read_buf)
    except OSError:
        return {"entries": [], "total": 0, "totalApproximate": False,
                "truncated": False, "size": 0, "scannedBytes": 0}

    text = b"".join(chunks).decode("utf-8", errors="replace")
    if position > 0:
        first_nl = text.find("\n")
        text = text[first_nl + 1:] if first_nl >= 0 else ""
    parsed = [ln for ln in text.split("\n") if ln]
    lines = parsed[-effective:] if effective > 0 else []
    truncated = position > 0 or len(parsed) > len(lines)
    return {
        "entries": _parse_lines(lines),
        "total": max(len(lines) + 1, effective + 1) if position > 0 else len(parsed),
        "totalApproximate": position > 0,
        "truncated": truncated,
        "size": size,
        "scannedBytes": scanned,
    }


def read_all(path: Optional[str], opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read jsonl history tail. Small files read precisely; large files reverse
    scan the tail to avoid loading a giant transcript into memory."""
    opts = opts or {}
    empty = {"entries": [], "total": 0, "totalApproximate": False,
             "truncated": False, "size": 0}
    if not path or not os.path.exists(path):
        return empty
    try:
        size = os.path.getsize(path)
    except OSError:
        return empty
    max_lines = max(0, int(opts.get("maxLines", DEFAULT_MAX_LINES))
                    if _is_finite(opts.get("maxLines")) else DEFAULT_MAX_LINES)
    tail_count = max(0, int(opts.get("tailCount", 0))
                     if _is_finite(opts.get("tailCount")) else 0)
    if size <= EXACT_TOTAL_MAX_BYTES:
        return _read_full(path, max_lines, tail_count)
    return _read_tail_window(path, max_lines, size, TAIL_CHUNK_BYTES,
                             TAIL_MAX_BYTES, tail_count)


def _is_finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class _FileTailer(threading.Thread):
    """Background thread that tails one jsonl file, emitting parsed entries.

    Polling-based (no watchdog dependency). Handles file not-yet-existing,
    truncation (size shrink -> reset offset), and partial last lines.
    """

    def __init__(self, path: str, start_offset: int,
                 on_entry: Callable[[Any, int], None],
                 on_error: Callable[[Exception], None],
                 poll: float = 0.3) -> None:
        super().__init__(daemon=True, name=f"jsonl-tail:{os.path.basename(path)}")
        self.path = path
        self.offset = max(0, int(start_offset or 0))
        self.on_entry = on_entry
        self.on_error = on_error
        self.poll = poll
        self.line_no = 0
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        buf = ""
        while not self._stop.is_set():
            try:
                if os.path.exists(self.path):
                    size = os.path.getsize(self.path)
                    if size < self.offset:
                        # truncated / rebuilt
                        self.offset = 0
                        self.line_no = 0
                        buf = ""
                    if size > self.offset:
                        with open(self.path, "rb") as f:
                            f.seek(self.offset)
                            chunk = f.read(size - self.offset)
                        self.offset = size
                        buf += chunk.decode("utf-8", errors="replace")
                        lines = buf.split("\n")
                        buf = lines.pop()  # keep last (possibly partial) line
                        for line in lines:
                            if not line:
                                continue
                            self.line_no += 1
                            try:
                                entry = json.loads(line)
                            except Exception as e:
                                self.on_error(Exception(
                                    f"JSON.parse line {self.line_no}: {e}; "
                                    f"raw={line[:200]}"))
                                continue
                            try:
                                self.on_entry(entry, self.line_no)
                            except Exception as e:
                                self.on_error(e)
            except Exception as e:  # noqa: BLE001 - tailer must never die
                self.on_error(e)
            self._stop.wait(self.poll)


class MergedWatcher:
    """Tails primary jsonl + .mobius.jsonl sidecar, merging into one on_entry."""

    def __init__(self, primary_path: str, mobius_path: Optional[str],
                 start_sentinel: Any, on_entry: Callable[[Any, int, str], None],
                 on_error: Callable[[Exception], None]) -> None:
        offsets = normalize_sentinel(start_sentinel, primary_path)
        self.primary = _FileTailer(
            primary_path, offsets["primary"],
            lambda raw, ln: on_entry(raw, ln, "primary"), on_error)
        self.mobius: Optional[_FileTailer] = None
        if mobius_path:
            self.mobius = _FileTailer(
                mobius_path, offsets["mobius"],
                lambda raw, ln: on_entry(raw, ln, "mobius"), on_error)
        self.primary.start()
        if self.mobius:
            self.mobius.start()

    def stop(self) -> None:
        self.primary.stop()
        if self.mobius:
            self.mobius.stop()

    def state(self) -> Dict[str, Any]:
        return {
            "primary": {"byteOffset": self.primary.offset, "lineNo": self.primary.line_no},
            "mobius": {"byteOffset": self.mobius.offset, "lineNo": self.mobius.line_no}
            if self.mobius else None,
        }


# --------------------------------------------------------------------------- #
# mobius-jsonl (port of services/mobius-jsonl.ts)
# --------------------------------------------------------------------------- #
def mobius_path_of(jsonl_path: Optional[str]) -> Optional[str]:
    if not jsonl_path or not isinstance(jsonl_path, str):
        return None
    if jsonl_path.endswith(".jsonl"):
        return jsonl_path[:-len(".jsonl")] + ".mobius.jsonl"
    return jsonl_path + ".mobius.jsonl"


def _file_size(path: Optional[str]) -> int:
    if not path:
        return 0
    try:
        return os.path.getsize(path) if os.path.exists(path) else 0
    except OSError:
        return 0


def _parse_timestamp_ms(entry: Any) -> Optional[int]:
    candidates = []
    if isinstance(entry, dict):
        candidates = [
            entry.get("timestamp"),
            entry.get("created_at"),
            (entry.get("payload") or {}).get("timestamp") if isinstance(entry.get("payload"), dict) else None,
            (entry.get("message") or {}).get("created_at") if isinstance(entry.get("message"), dict) else None,
        ]
    for raw in candidates:
        if not raw:
            continue
        ms = _to_epoch_ms(raw)
        if ms is not None:
            return ms
    return None


def _to_epoch_ms(raw: Any) -> Optional[int]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return int(dt.timestamp() * 1000)
    except ValueError:
        return None


def _source_order(source: str) -> int:
    return 0 if source == "primary" else 1


def _compare_records(a: Tuple[Any, int, str], b: Tuple[Any, int, str]) -> int:
    at = _parse_timestamp_ms(a[0])
    bt = _parse_timestamp_ms(b[0])
    if at is not None and bt is not None and at != bt:
        return -1 if at < bt else 1
    if at is None and bt is not None:
        return -1
    if at is not None and bt is None:
        return 1
    so = _source_order(a[2]) - _source_order(b[2])
    if so != 0:
        return so
    return -1 if a[1] < b[1] else (1 if a[1] > b[1] else 0)


def read_merged_history(jsonl_path: Optional[str],
                        opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    opts = opts or {}
    max_lines = max(0, int(opts.get("maxLines", DEFAULT_MAX_LINES))
                    if _is_finite(opts.get("maxLines")) else DEFAULT_MAX_LINES)
    tail_count = max(0, int(opts.get("tailCount", 0))
                     if _is_finite(opts.get("tailCount")) else 0)
    mobius_path = mobius_path_of(jsonl_path)
    side_opts = {**opts, "maxLines": max_lines, "tailCount": tail_count}
    primary = read_all(jsonl_path, side_opts)
    mobius = read_all(mobius_path, side_opts) if mobius_path else {
        "entries": [], "total": 0, "totalApproximate": False, "truncated": False, "size": 0}
    records: List[Tuple[Any, int, str]] = []
    for i, e in enumerate(primary["entries"]):
        records.append((e, i, "primary"))
    for i, e in enumerate(mobius["entries"]):
        records.append((e, i, "mobius"))
    records.sort(key=_cmp_key)
    total = (primary["total"] or 0) + (mobius["total"] or 0)
    if tail_count > 0:
        effective = min(max_lines, tail_count) if max_lines > 0 else tail_count
    else:
        effective = max_lines
    entries = [r[0] for r in (records[-effective:] if effective > 0 else [])]
    return {
        "entries": entries,
        "total": total,
        "totalApproximate": bool(primary["totalApproximate"]) or bool(mobius["totalApproximate"]),
        "truncated": total > len(entries) or bool(primary["truncated"]) or bool(mobius["truncated"]),
        "sentinel": {"primary": primary["size"] or 0, "mobius": mobius["size"] or 0},
        "paths": {"primary": jsonl_path or None, "mobius": mobius_path or None},
    }


def _cmp_key(rec: Tuple[Any, int, str]):
    """Sort key mirroring _compare_records (timestamp, source order, index).

    None-timestamp entries sort FIRST (JS: at==null && bt!=null -> -1, i.e. a
    sorts before b). Then ascending timestamp; then primary before mobius; then
    original index for stable ordering within a source."""
    entry, idx, source = rec
    ts = _parse_timestamp_ms(entry)
    has_ts = ts is not None
    # group 0 = no timestamp (sorts first), group 1 = has timestamp
    return (0 if not has_ts else 1, ts if has_ts else 0, _source_order(source), idx)


def current_merged_sentinel(jsonl_path: Optional[str]) -> Dict[str, int]:
    return {"primary": _file_size(jsonl_path), "mobius": _file_size(mobius_path_of(jsonl_path))}


def normalize_sentinel(sentinel: Any, jsonl_path: Optional[str]) -> Dict[str, int]:
    current = current_merged_sentinel(jsonl_path)
    if isinstance(sentinel, (int, float)) and not isinstance(sentinel, bool):
        s = int(sentinel)
        return {"primary": max(0, s), "mobius": 0 if s == 0 else current["mobius"]}
    if not isinstance(sentinel, dict):
        return current
    primary = sentinel.get("primary", sentinel.get("primarySize", sentinel.get("size")))
    mobius = sentinel.get("mobius", sentinel.get("mobiusSize"))
    return {
        "primary": int(primary) if _is_finite(primary) and primary >= 0 else current["primary"],
        "mobius": int(mobius) if _is_finite(mobius) and mobius >= 0 else current["mobius"],
    }


def _prompt_kind(content: Any, explicit_kind: Optional[str] = None) -> str:
    if explicit_kind:
        return explicit_kind
    text = str(content or "").strip()
    return "compact" if text.startswith("/compact") else "user_input"


def build_mobius_user_entry(*, session_id=None, agent_session_id=None, cwd=None,
                            backend_name=None, content=None, input_text=None,
                            request_id=None, turn_number=None, source=None,
                            user_id=None, kind=None, timestamp=None) -> Dict[str, Any]:
    ts = timestamp or (datetime.utcnow().isoformat() + "Z")
    body = str(content or "")
    typed = None if input_text is None else str(input_text)
    resolved_kind = _prompt_kind(body, kind)
    return {
        "parentUuid": None,
        "isSidechain": False,
        "promptId": str(uuid.uuid4()),
        "type": "user",
        "message": {"role": "user", "content": body},
        "uuid": str(uuid.uuid4()),
        "timestamp": ts,
        "permissionMode": "bypassPermissions",
        "userType": "external",
        "entrypoint": "mobius",
        "cwd": cwd or None,
        "sessionId": agent_session_id or session_id,
        "version": f"mobius-jsonl/{MOBIUS_JSONL_VERSION}",
        "mobius": {
            "schema_version": MOBIUS_JSONL_VERSION,
            "source": source or "session.send",
            "kind": resolved_kind,
            "backend": backend_name or None,
            "session_id": session_id or None,
            "agent_session_id": agent_session_id or None,
            "user_id": user_id or None,
            "request_id": request_id or None,
            "turn_number": int(turn_number) if _is_finite(turn_number) else None,
            "input_text": typed,
            "content_length": len(body),
            "captured_at": ts,
        },
    }


def append_mobius_prompt_entry(jsonl_path: str, **entry_opts) -> Tuple[str, Dict[str, Any]]:
    file_path = mobius_path_of(jsonl_path)
    if not file_path:
        raise ValueError("缺少原始 JSONL 路径, 无法写入 mobius JSONL")
    entry = build_mobius_user_entry(**entry_opts)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return file_path, entry


# --------------------------------------------------------------------------- #
# Global neutral event bus (port of agents/events.js). Lets out-of-band
# observers subscribe to raw agent entries across all sessions/backends.
# --------------------------------------------------------------------------- #
_BUS_LOCK = threading.Lock()
_RAW_LISTENERS: List[Callable[[Dict[str, Any]], None]] = []


def emit_agent_raw_entry(payload: Dict[str, Any]) -> None:
    with _BUS_LOCK:
        listeners = list(_RAW_LISTENERS)
    for fn in listeners:
        try:
            fn(payload)
        except Exception as e:  # noqa: BLE001 - bus must not break on listener
            print(f"[events] raw_entry listener error: {e}", file=sys.stderr)


def on_agent_raw_entry(listener: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
    with _BUS_LOCK:
        _RAW_LISTENERS.append(listener)

    def off() -> None:
        with _BUS_LOCK:
            try:
                _RAW_LISTENERS.remove(listener)
            except ValueError:
                pass
    return off


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _encode_cwd(cwd: str) -> str:
    """cwd -> ~/.claude/projects/ subdir name. All non-alphanumerics -> '-'."""
    return re.sub(r"[^a-zA-Z0-9]", "-", cwd)


def _jsonl_path_of(home: str, cwd: str, claude_session_id: str) -> str:
    # Honour CLAUDE_CONFIG_DIR (Claude Code's official override for ~/.claude)
    # so the driver reads transcripts from wherever the spawned claude wrote
    # them. Falls back to ~/.claude when unset (default Claude behaviour).
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(home, ".claude")
    return os.path.join(config_dir, "projects", _encode_cwd(cwd),
                        f"{claude_session_id}.jsonl")


def _normalize_use_proxy(value: Any, fallback: bool = True) -> bool:
    if value in (False, 0, "0", "false"):
        return False
    if value in (True, 1, "1", "true"):
        return True
    return bool(fallback)


def _find_ascii_tail_marker(text: str) -> Optional[str]:
    """Find the trailing ASCII run (5..15 chars) of text, used as a capture-pane
    marker to confirm a paste actually landed in the TUI input box."""
    i = len(text) - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    tail = ""
    while i >= 0 and len(tail) < 15:
        ch = text[i]
        if not ("\x20" <= ch <= "\x7E"):
            break
        tail = ch + tail
        i -= 1
    return tail if len(tail) >= 5 else None


def _pick_initial_context_plan() -> str:
    import random
    roll = random.random()
    if roll < 1 / 3:
        return "greeting_then_context"
    if roll < 2 / 3:
        return "direct_context"
    return "delay_then_context"


def _pick_initial_context_greeting() -> str:
    import random
    return random.choice(INITIAL_CONTEXT_GREETING_CHOICES)


def ensure_project_trusted(cwd: str, home: str, log_fn: Callable[..., None]) -> bool:
    """Pre-mark cwd as trusted in ~/.claude.json so the TUI doesn't pop the
    'trust this folder' dialog. Idempotent + atomic (tmp+rename). Any failure is
    non-fatal: the ready-poll screen-scraper auto-confirms the dialog as backup."""
    try:
        abs_cwd = os.path.abspath(cwd)
        cfg = os.path.join(home, ".claude.json")
        if not os.path.exists(cfg):
            return False
        with open(cfg, "r", encoding="utf-8") as f:
            j = json.load(f)
        if not isinstance(j, dict):
            j = {}
        if not isinstance(j.get("projects"), dict):
            j["projects"] = {}
        cur = j["projects"].get(abs_cwd)
        if isinstance(cur, dict) and cur.get("hasTrustDialogAccepted") is True:
            return True
        merged = dict(cur) if isinstance(cur, dict) else {}
        merged["hasTrustDialogAccepted"] = True
        j["projects"][abs_cwd] = merged
        tmp = f"{cfg}.imac-tmp-{os.getpid()}-{_now_ms()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        os.replace(tmp, cfg)
        log_fn(f"[claude-code-driver] 预置目录信任: {abs_cwd} -> ~/.claude.json")
        return True
    except Exception as e:  # noqa: BLE001 - non-fatal, screen-scrape backs it up
        print(f"[claude-code-driver] 预置目录信任失败 (走截屏兜底): {e}", file=sys.stderr)
        return False


def detect_danger_permission(text: str) -> Tuple[bool, Optional[str]]:
    """Detect the dangerous-operation permission box. Returns (pending, warning).
    Requires the danger line + 'Do you want to proceed?' + 'Esc to cancel'
    co-present (strong signal) to avoid historical log residue false positives."""
    if not text:
        return False, None
    m = CLAUDE_DANGER_OPERATION_RE.search(text)
    if not m:
        return False, None
    if "Do you want to proceed?" not in text or "Esc to cancel" not in text:
        return False, None
    return True, m.group(0).strip()


class TmuxClaudeCodeDriver:
    """Independent tmux-based Claude Code driver.

    One tmux hub session; one window per session_id (window name = session_id).
    Each window runs an interactive `claude --dangerously-skip-permissions` TUI
    (optionally via proxychains). Prompts are pasted with bracketed paste; agent
    output is tailed from claude's jsonl transcript. Completion is signaled by
    the agent deleting running.flag.
    """

    def __init__(self, config: Optional[DriverConfig] = None, **overrides: Any) -> None:
        cfg = config or DriverConfig()
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        self.cfg = cfg
        # If a config-dir override is set, expose it via CLAUDE_CONFIG_DIR so the
        # module-level _jsonl_path_of (which has no cfg handle) reads transcripts
        # from the same place the spawned claude writes them.
        if cfg.claude_config_dir:
            os.environ.setdefault("CLAUDE_CONFIG_DIR", cfg.claude_config_dir)
        self.ops = TmuxOps(cfg.tmux_log_file)
        self._log = self.ops.log

        # Preflight: tmux + claude are mandatory; proxy deps are optional.
        missing = []
        for b in ("tmux", "claude"):
            if not shutil.which(b):
                missing.append(f"bin (PATH): {b}")
        if missing:
            raise RuntimeError(
                "[claude-code-driver] preflight 失败, 拒绝启动: " + ", ".join(missing))
        proxy_missing = self._proxy_prereq_missing()
        if proxy_missing:
            print(f"[claude-code-driver] ⚠️  proxychains 依赖不完整; "
                  f"use_proxy=false 的会话仍可直连启动: {', '.join(proxy_missing)}",
                  file=sys.stderr)
        self._log(f"[claude-code-driver] ✅ preflight pass (HUB={cfg.hub})")

        # runtime: live in-memory session->entry map. persisted: disk mirror
        # (live, forgotten on terminate). archive: all-time (never forgotten,
        # so get_history can find jsonl paths after a window is closed).
        self.runtime: Dict[str, Dict[str, Any]] = {}
        self.persisted: Dict[str, Dict[str, Any]] = self._load_json(cfg.runtime_file)
        self.archive: Dict[str, Dict[str, Any]] = self._load_json(cfg.archive_file)
        # One-time catch-up: copy live entries not yet in archive.
        dirty = False
        for sid, p in self.persisted.items():
            if sid not in self.archive:
                self.archive[sid] = dict(p)
                dirty = True
        if dirty:
            self._save_archive()

        # per-session locks + watchers + caches + listener registry + heal state
        self._locks: Dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._watchers: Dict[str, MergedWatcher] = {}
        self._cache_lock = threading.Lock()
        self._list_windows_cache: Optional[Dict[str, Any]] = None  # {ts, rows}
        self._pane_tail_cache: Dict[str, Dict[str, Any]] = {}
        self._listeners: Dict[str, List[Callable[[Any], None]]] = {}
        self._listeners_lock = threading.Lock()
        self._heal_state: Dict[str, Dict[str, Any]] = {}
        self._heal_lock = threading.Lock()

        self._restore_from_persisted()

    # ── persistence ────────────────────────────────────────
    def _load_json(self, file: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(file):
                return {}
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"[claude-code-driver] load {os.path.basename(file)} failed: {e}",
                  file=sys.stderr)
            return {}

    def _save_json(self, file: str, obj: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[claude-code-driver] save {os.path.basename(file)} failed: {e}",
                  file=sys.stderr)

    def _save_persisted(self) -> None:
        self._save_json(self.cfg.runtime_file, self.persisted)

    def _save_archive(self) -> None:
        self._save_json(self.cfg.archive_file, self.archive)

    def _reload_persisted(self) -> None:
        self.persisted = self._load_json(self.cfg.runtime_file)
        self.archive = self._load_json(self.cfg.archive_file)

    def _lookup_persisted_entry(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id not in self.persisted:
            self._reload_persisted()
        return self.persisted.get(session_id)

    def _lookup_persisted_jsonl_path(self, session_id: str) -> Optional[str]:
        e = self._lookup_persisted_entry(session_id)
        return (e or {}).get("jsonlPath")

    def _lookup_archived_jsonl_path(self, session_id: str) -> Optional[str]:
        if session_id not in self.archive:
            self.archive = self._load_json(self.cfg.archive_file)
        return self.archive.get(session_id, {}).get("jsonlPath")

    def _persist_entry(self, session_id: str, partial: Dict[str, Any]) -> None:
        cur = self.persisted.get(session_id, {})
        cur.update(partial)
        self.persisted[session_id] = cur
        self._save_persisted()
        acur = self.archive.get(session_id, {})
        acur.update(partial)
        self.archive[session_id] = acur
        self._save_archive()

    def _forget_persisted(self, session_id: str) -> None:
        self.persisted.pop(session_id, None)
        self._save_persisted()
        # archive is intentionally NOT touched (kept for history lookups).

    def _restore_from_persisted(self) -> None:
        """On startup, pull sessionId->agentSessionId/jsonlPath mappings back
        from disk. The claude TUI in tmux may still be alive; we don't restart
        it. Watchers are started lazily (threads can't be created before the
        driver is in use); call start() to start them for restored entries."""
        total = 0
        for sid, p in self.persisted.items():
            total += 1
            p = p or {}
            jp = p.get("jsonlPath")
            if not jp or not os.path.exists(jp):
                self._log(f"[claude-code-driver] runtime 条目 {sid} 被丢弃 "
                          f"(jsonl 缺失: {jp})")
                continue
            self.runtime[sid] = {
                "agentSessionId": p.get("agentSessionId"),
                "cwd": p.get("cwd"),
                "flagRoot": p.get("flagRoot") or p.get("cwd"),
                "model": p.get("model"),
                "useProxy": _normalize_use_proxy(p.get("useProxy"), True),
                "settingsPath": p.get("settingsPath"),
                "forceNoProxy": bool(p.get("forceNoProxy")),
                "displayName": p.get("displayName"),
                "jsonlPath": jp,
                "startedAt": p.get("startedAt") or 0,
                "watch": None,
            }
        self._log(f"[claude-code-driver] runtime 加载 {len(self.runtime)}/{total} 条")

    def start(self) -> None:
        """Start jsonl watchers for all restored runtime entries (background
        daemon threads). Safe to call once at startup; idempotent."""
        for sid in list(self.runtime.keys()):
            self._ensure_watcher(sid)

    def shutdown(self) -> None:
        """Stop all watchers. Does NOT kill tmux windows (they survive a driver
        restart by design). Call terminate_session to kill a window."""
        for sid, w in list(self._watchers.items()):
            try:
                w.stop()
            except Exception:
                pass
        self._watchers.clear()

    # ── watchers / events ─────────────────────────────────
    def _ensure_watcher(self, session_id: str) -> None:
        entry = self.runtime.get(session_id)
        if not entry or not entry.get("jsonlPath"):
            return
        if session_id in self._watchers:
            return
        jp = entry["jsonlPath"]
        watcher = MergedWatcher(
            jp, mobius_path_of(jp), None,
            on_entry=lambda raw, ln, src: self._emit_raw(session_id, raw),
            on_error=lambda e: print(f"[claude-code-driver/watch {session_id}] {e}",
                                     file=sys.stderr),
        )
        self._watchers[session_id] = watcher

    def _emit_raw(self, session_id: str, raw: Any) -> None:
        with self._listeners_lock:
            listeners = list(self._listeners.get(session_id, []))
        for fn in listeners:
            try:
                fn(raw)
            except Exception as e:  # noqa: BLE001
                print(f"[claude-code-driver] raw listener error ({session_id}): {e}",
                      file=sys.stderr)
        emit_agent_raw_entry({
            "backend": "claude-code-driver",
            "sessionId": session_id,
            "entry": raw,
        })

    def subscribe_raw(self, session_id: str,
                      listener: Callable[[Any], None]) -> Callable[[], None]:
        """Subscribe to live raw entries for a session (fed by the shared watcher
        started in start()/create). Returns an unsubscribe callable."""
        with self._listeners_lock:
            self._listeners.setdefault(session_id, []).append(listener)

        def off() -> None:
            with self._listeners_lock:
                lst = self._listeners.get(session_id, [])
                try:
                    lst.remove(listener)
                except ValueError:
                    pass
        return off

    def subscribe_raw_from(self, session_id: str, sentinel: Any,
                           listener: Callable[[Any], None]) -> Callable[[], None]:
        """Subscribe to live raw entries starting from a sentinel (byte offset),
        using an INDEPENDENT tailer (does not conflict with the shared watcher).
        Falls back to the shared emitter if no jsonl path is known."""
        jsonl_path = self._resolve_jsonl_path(session_id)
        if not jsonl_path:
            return self.subscribe_raw(session_id, listener)
        holder: Dict[str, Optional[MergedWatcher]] = {"w": None}
        watcher = MergedWatcher(
            jsonl_path, mobius_path_of(jsonl_path), sentinel,
            on_entry=lambda raw, ln, src: listener(raw),
            on_error=lambda e: print(f"[claude-code-driver/sub {session_id}] {e}",
                                     file=sys.stderr),
        )
        holder["w"] = watcher

        def off() -> None:
            w = holder.get("w")
            if w:
                try:
                    w.stop()
                except Exception:
                    pass
        return off

    # ── per-session lock ──────────────────────────────────
    def _lock_for(self, session_id: str) -> threading.RLock:
        with self._locks_guard:
            lk = self._locks.get(session_id)
            if lk is None:
                lk = threading.RLock()
                self._locks[session_id] = lk
            return lk

    def _with_lock(self, session_id: str, fn: Callable[[], Any]) -> Any:
        with self._lock_for(session_id):
            return fn()

    # ── public control methods (lock-wrapped) ─────────────
    def create_new_session(self, opts: Dict[str, Any]) -> Dict[str, Any]:
        sid = str(opts.get("session_id") or opts.get("sessionId") or "")
        return self._with_lock(sid, lambda: self._create_impl(opts))

    def pause_current_and_resume_from_session(self, opts: Dict[str, Any]) -> None:
        sid = str(opts.get("session_id") or opts.get("sessionId") or "")
        self._with_lock(sid, lambda: self._pause_impl(opts))

    # alias matching JS noPauseCurrentAndQueueQueryAtSession
    def no_pause_current_and_queue_query_at_session(self, opts: Dict[str, Any]) -> None:
        sid = str(opts.get("session_id") or opts.get("sessionId") or "")
        self._with_lock(sid, lambda: self._queue_impl(opts))

    def terminate_session(self, session_id: str) -> Dict[str, Any]:
        return self._with_lock(session_id, lambda: self._terminate_impl(session_id))

    # ── status queries (no lock; cache-guarded) ───────────
    def _hub_exists(self) -> bool:
        return self.ops.tmux(["has-session", "-t", self.cfg.hub]).status == 0

    def _ensure_hub(self) -> None:
        if self._hub_exists():
            return
        r = self.ops.tmux(["new-session", "-d", "-s", self.cfg.hub, "-n", "_root"])
        if r.status != 0:
            raise RuntimeError(f"tmux new-session 失败: {r.stderr}")
        self._log(f"[claude-code-driver] created tmux session {self.cfg.hub}")

    # ── pane 捕获 (诊断 claude-code 误判 timeout 用) ─────────────────
    _PANE_STAMPER_PATH = "/root/.cc_pane_stamp.py"

    def _ensure_pane_capture_stamper(self) -> None:
        """落一个"每行加时间戳"的小工具, pipe-pane 用它捕获 claude TUI 屏幕。

        claude-code 的 "API Error (attempt n)… Retrying" 横幅只出现在 TUI 上,
        不进 jsonl; 用 tmux pipe-pane 把整屏输出按行加 %H:%M:%S.%f 前缀写到
        /root/cc_pane_<window>.log, stage 结束随 jsonl 一起拉回宿主机,
        便可与 interchange 侧 [reqmon] 日志逐秒对表。
        """
        try:
            if os.path.exists(self._PANE_STAMPER_PATH):
                return
            with open(self._PANE_STAMPER_PATH, "w", encoding="utf-8") as f:
                f.write(
                    "import sys, datetime\n"
                    "for line in sys.stdin:\n"
                    "    sys.stdout.write(datetime.datetime.now()"
                    ".strftime('%H:%M:%S.%f')[:12] + ' ' + line)\n"
                    "    sys.stdout.flush()\n"
                )
        except Exception as e:
            self._log(f"[claude-code-driver] pane stamper 写入失败(不影响运行): {e}")

    def _start_pane_capture(self, session_id: str) -> None:
        """对新窗口开 pipe-pane -> /root/cc_pane_<session_id>.log (容错, 失败仅记日志)。"""
        if os.environ.get("CC_PANE_CAPTURE", "1") != "1":
            return
        try:
            self._ensure_pane_capture_stamper()
            target = f"{self.cfg.hub}:{session_id}"
            log_path = f"/root/cc_pane_{session_id}.log"
            cmd = (f"python3 -u {self._PANE_STAMPER_PATH} >> {log_path} 2>/dev/null")
            r = self.ops.tmux(["pipe-pane", "-t", target, "-o", cmd])
            if r.status == 0:
                self._log(f"[claude-code-driver] pane capture on -> {log_path}")
            else:
                self._log(f"[claude-code-driver] pipe-pane 失败(不影响运行): {r.stderr}")
        except Exception as e:
            self._log(f"[claude-code-driver] pane capture 启动失败(不影响运行): {e}")

    def _window_exists(self, name: str) -> bool:
        """Real-time (uncached) window existence check for control flow."""
        r = self.ops.tmux(["list-windows", "-t", self.cfg.hub, "-F", "#{window_name}"])
        if r.status != 0:
            return False
        return name in (r.stdout or "").split("\n")

    def _invalidate_window_cache(self, session_id: Optional[str] = None) -> None:
        """Drop the list-windows cache (and a session's pane-tail cache) after a
        control-flow mutation (spawn / kill-window), so a subsequent is_alive /
        real_time_info reflects reality immediately instead of waiting out the TTL.
        (The JS driver tolerates the stale window for up to 3s because its UI
        polls; an independent driver's callers expect post-mutation consistency.)"""
        with self._cache_lock:
            self._list_windows_cache = None
            if session_id is not None:
                self._pane_tail_cache.pop(session_id, None)
            else:
                self._pane_tail_cache.clear()

    def _list_windows_rows_cached(self) -> List[List[str]]:
        now = _now_ms()
        with self._cache_lock:
            c = self._list_windows_cache
            if c and now - c["ts"] < LIST_WINDOWS_TTL_MS:
                return c["rows"]
        r = self.ops.tmux(["list-windows", "-t", self.cfg.hub, "-F",
                           "#{window_name}|#{pane_pid}|#{window_index}|#{window_activity}"
                           "|#{pane_dead}|#{pane_current_command}"])
        rows: List[List[str]] = []
        if r.status == 0 and (r.stdout or "").strip():
            for line in (r.stdout or "").strip().split("\n"):
                if line:
                    rows.append(line.split("|"))
        with self._cache_lock:
            self._list_windows_cache = {"ts": now, "rows": rows}
        return rows

    def is_alive(self, session_id: str) -> bool:
        return any(cols and cols[0] == session_id
                   for cols in self._list_windows_rows_cached())

    def _capture_pane_tail(self, session_id: str) -> str:
        """capture-pane tail (last 25 lines), ANSI-stripped, 5s TTL cached."""
        now = _now_ms()
        with self._cache_lock:
            c = self._pane_tail_cache.get(session_id)
            if c and now - c["ts"] < PANE_TAIL_TTL_MS:
                return c["text"]
        text = ""
        try:
            pane = self.ops.tmux(["capture-pane", "-pt",
                                  f"{self.cfg.hub}:{session_id}", "-p", "-S", "-25"])
            if pane.status == 0 and pane.stdout:
                text = _ANSI_RE.sub("", pane.stdout)
        except Exception:
            text = ""
        with self._cache_lock:
            self._pane_tail_cache[session_id] = {"ts": now, "text": text}
        return text

    def _claude_pane_still_busy(self, session_id: str) -> bool:
        """For /stop fallback: grab the freshest pane text (bypass cache) and
        decide if claude is still mid-turn. Busy anchors: the status line,
        'Waiting for N background agents', and a background subagent row that
        still shows a live elapsed timer (means a subagent is mid-run)."""
        text = ""
        try:
            pane = self.ops.tmux(["capture-pane", "-pt",
                                  f"{self.cfg.hub}:{session_id}", "-p", "-S", "-25"])
            if pane.status == 0 and pane.stdout:
                text = _ANSI_RE.sub("", pane.stdout)
        except Exception:
            return False
        if not text:
            return False
        return bool(CLAUDE_STATUS_LINE_RE.search(text)
                    or CLAUDE_BG_AGENTS_WAITING_RE.search(text)
                    or CLAUDE_BG_SUBAGENT_RUNNING_RE.search(text))

    def is_working(self, session_id: str) -> bool:
        if not self.is_alive(session_id):
            return False
        entry = self.runtime.get(session_id)
        if not entry or not entry.get("jsonlPath"):
            return False
        jp = entry["jsonlPath"]
        try:
            if not os.path.exists(jp):
                return False
            size = os.path.getsize(jp)
            if size == 0:
                return False
            n = min(size, CLAUDE_WORKING_TAIL_BYTES)
            with open(jp, "rb") as f:
                f.seek(size - n)
                data = f.read(n)
            lines = [ln for ln in data.decode("utf-8", errors="replace").split("\n") if ln]
        except OSError:
            return False

        # Reverse scan, whitelist logic: only user/assistant/system(specific
        # subtype) decide state. Every other type (attachment, ai-title,
        # permission-mode, queue-operation, future metadata) is skipped.
        for line in reversed(lines):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if not isinstance(e, dict):
                continue
            etype = e.get("type")
            if etype == "assistant":
                sr = (e.get("message") or {}).get("stop_reason") if isinstance(e.get("message"), dict) else None
                # missing / 'tool_use' -> still running.
                if not sr or sr == "tool_use":
                    return True
                # end_turn / max_tokens / stop_sequence -> the main agent's turn
                # is done. BUT when background subagents were launched fire-and-
                # forget, the main transcript ends in end_turn while a subagent
                # is still mid-run — a state the jsonl cannot express. Consult
                # the live TUI before declaring idle, otherwise the session is
                # killed (and any in-flight subagent result is lost) before the
                # agent gets to write deliverables. (This was the root cause of
                # the forcedmulti "premature kill" failures.)
                if self._pane_has_running_bg_agents(session_id):
                    return True
                return False
            if etype == "user":
                return True
            if etype == "system":
                sub = e.get("subtype")
                if sub in ("init", "hook_started", "hook_response"):
                    return True
                # turn_duration / stop_hook_summary / away_summary / unknown -> skip
            # all other types -> skip
        # jsonl looks idle (no deciding entry in the tail), but the TUI may be
        # waiting on background agents — a state the jsonl can't express.
        # Fall back to capture-pane.
        return self._pane_has_running_bg_agents(session_id)

    def _pane_has_running_bg_agents(self, session_id: str) -> bool:
        """True if the live TUI pane shows a background subagent still running:
        either the explicit "Waiting for N background agents to finish" line or
        a subagent row that still carries a live elapsed timer. Bypasses the
        5s capture-pane cache so a just-launched subagent is seen immediately."""
        text = ""
        try:
            pane = self.ops.tmux(["capture-pane", "-pt",
                                  f"{self.cfg.hub}:{session_id}", "-p", "-S", "-25"])
            if pane.status == 0 and pane.stdout:
                text = _ANSI_RE.sub("", pane.stdout)
        except Exception:
            return False
        if not text:
            return False
        return bool(CLAUDE_BG_AGENTS_WAITING_RE.search(text)
                    or CLAUDE_BG_SUBAGENT_RUNNING_RE.search(text))

    def is_job_goal_accomplished(self, session_id: str) -> bool:
        entry = self.runtime.get(session_id)
        root = (entry or {}).get("flagRoot") or (entry or {}).get("cwd")
        if not root:
            return False
        return not os.path.exists(
            running_flag_path_of(root, session_id, self.cfg.hidden_folder))

    def is_failed(self, session_id: str) -> bool:
        entry = self.runtime.get(session_id)
        root = (entry or {}).get("flagRoot") or (entry or {}).get("cwd")
        if not root:
            return False
        return os.path.exists(
            failed_flag_path_of(root, session_id, self.cfg.hidden_folder))

    def list_sessions(self) -> List[Dict[str, Any]]:
        out = []
        for cols in self._list_windows_rows_cached():
            if not cols or len(cols) < 6:
                continue
            name, pid, idx, activity, pane_dead, pane_cmd = cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]
            entry = self.runtime.get(name)
            last_activity_sec = 0
            try:
                last_activity_sec = int(activity)
            except ValueError:
                last_activity_sec = 0
            last_activity_ms = (last_activity_sec * 1000
                                if last_activity_sec > 0 else None)
            out.append({
                "sessionId": name,
                "agentSessionId": (entry or {}).get("agentSessionId"),
                "pid": _to_int(pid),
                "index": _to_int(idx),
                "lastActivityMs": last_activity_ms,
                "lastActivityAt": (datetime.utcfromtimestamp(last_activity_ms / 1000).isoformat() + "Z"
                                   if last_activity_ms else None),
                "tmuxOpen": True,
                "paneDead": pane_dead == "1",
                "paneCurrentCommand": pane_cmd or None,
            })
        return out

    def real_time_info(self, session_id: str) -> str:
        """Live status line for a working session (LIVE card hint). Best-effort:
        never raises; returns '' when not alive/working or no line is found."""
        try:
            if self.is_alive(session_id) and self.is_working(session_id):
                pane_text = self._capture_pane_tail(session_id)
                danger_pending, warning = detect_danger_permission(pane_text)
                if danger_pending:
                    self._maybe_heal_danger_permission(session_id, warning or "")
                lines = pane_text.split("\n")
                for line in reversed(lines):
                    if line and CLAUDE_STATUS_LINE_RE.search(line):
                        return line.strip()
        except Exception:
            pass
        return ""

    def get_history(self, session_id: str, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        jsonl_path = self._resolve_jsonl_path(session_id)
        if not jsonl_path:
            return {"entries": [], "total": 0, "truncated": False, "sentinel": 0}
        return read_merged_history(jsonl_path, opts)

    def get_session_title(self, session_id: str,
                          opts: Optional[Dict[str, Any]] = None) -> Optional[str]:
        hist = self.get_history(session_id, opts) or {}
        entries = hist.get("entries") or []
        for e in reversed(entries):
            if isinstance(e, dict) and e.get("type") == "ai-title":
                title = e.get("aiTitle") or e.get("ai_title") or e.get("title")
                title = re.sub(r"\s+", " ", str(title or "")).replace("\x00", "").strip()
                if title:
                    return title
        return None

    def get_recent_error(self, session_id: str) -> None:
        # Claude TUI exposes no error channel; null (matches JS tmux-claude-code).
        return None

    def _resolve_jsonl_path(self, session_id: str) -> Optional[str]:
        entry = self.runtime.get(session_id)
        if entry and entry.get("jsonlPath"):
            return entry["jsonlPath"]
        p = self._lookup_persisted_jsonl_path(session_id)
        if p:
            return p
        return self._lookup_archived_jsonl_path(session_id)

    # ── proxy helpers ─────────────────────────────────────
    def _proxy_prereq_missing(self) -> List[str]:
        missing = []
        if not os.path.exists(self.cfg.proxy_envs):
            missing.append(f"file: {self.cfg.proxy_envs}")
        if not os.path.exists(self.cfg.proxy_conf):
            missing.append(f"file: {self.cfg.proxy_conf}")
        if not shutil.which("proxychains"):
            missing.append("bin (PATH): proxychains")
        return missing

    def _assert_proxy_available(self) -> None:
        missing = self._proxy_prereq_missing()
        if missing:
            raise RuntimeError(
                f"use_proxy=true 但代理依赖缺失: {', '.join(missing)}")

    # ── flag helpers ──────────────────────────────────────
    def _mark_running(self, root: Optional[str], session_id: str) -> bool:
        return safe_write_running_flag(root, session_id, self.cfg.hidden_folder,
                                       {}, "claude-code-driver")

    def _clear_running(self, root: Optional[str], session_id: str) -> bool:
        return safe_remove_running_flag(root, session_id, self.cfg.hidden_folder,
                                        "claude-code-driver")

    def _append_mobius_prompt_entry(self, session_id: str,
                                    mobius_jsonl: Optional[Dict[str, Any]]) -> bool:
        if not mobius_jsonl:
            return False
        entry = self.runtime.get(session_id)
        if not entry or not entry.get("jsonlPath"):
            print(f"[claude-code-driver] mobius jsonl skipped ({session_id}): "
                  f"original jsonl path missing", file=sys.stderr)
            return False
        try:
            append_mobius_prompt_entry(
                entry["jsonlPath"],
                session_id=session_id,
                agent_session_id=entry.get("agentSessionId"),
                cwd=entry.get("cwd"),
                backend_name="claude-code-driver",
                **mobius_jsonl,
            )
            return True
        except Exception as e:
            print(f"[claude-code-driver] mobius jsonl append failed "
                  f"({session_id}): {e}", file=sys.stderr)
            return False

    # ── danger-permission self-heal ───────────────────────
    def _maybe_heal_danger_permission(self, session_id: str, warning: str) -> None:
        with self._heal_lock:
            st = self._heal_state.get(session_id,
                                      {"healing": False, "lastWarning": "", "lastTs": 0})
            now = _now_ms()
            if st["healing"]:
                return
            if warning == st["lastWarning"] and now - st["lastTs"] < DANGER_HEAL_COOLDOWN_MS:
                return
            self._heal_state[session_id] = {"healing": True, "lastWarning": warning, "lastTs": now}
        t = threading.Thread(target=self._heal_danger_permission_safe,
                             args=(session_id, warning), daemon=True)
        t.start()

    def _heal_danger_permission_safe(self, session_id: str, warning: str) -> None:
        try:
            self._heal_danger_permission(session_id, warning)
        except Exception as e:
            self._log(f"[claude-code-driver] danger heal 失败 session={session_id}: {e}")
        finally:
            with self._heal_lock:
                cur = self._heal_state.get(session_id)
                if cur:
                    cur["healing"] = False

    def _heal_danger_permission(self, session_id: str, warning: str) -> None:
        """Detached self-heal (does not block real_time_info / status polling).

        AMEND the dangerous command in place instead of merely cancelling it:
          1) Tab   -> enter the amend input (the dialog advertises 'Tab to amend').
          2) paste -> the configured amend instruction (default
                      'use mv instead of rm'), so claude retries with a safer
                      command instead of running the rm.
          3) Enter -> submit the amendment (resubmit xN; an Enter on an
                      already-submitted box is a no-op, mirroring prompt-paste).

        Worst case the amend field never opened: the pasted text is ignored by the
        Yes/No box and Enter confirms the highlighted default ('2. No') -> the rm
        is denied, which is still safe (claude just continues without the rm)."""
        if not self._window_exists(session_id):
            return
        target = f"{self.cfg.hub}:{session_id}"
        msg = (self.cfg.danger_amend_msg or "").strip()
        self._log(f"[claude-code-driver] danger permission 检测到, 尝试 amend "
                  f"(session={session_id}): {warning}"
                  + (f" -> 反馈={msg!r}" if msg else " -> 无 amend 文本, 仅 Enter 拒绝"))
        # 1) Tab -> amend input.
        self.ops.tmux(["send-keys", "-t", target, "Tab"])
        with self._cache_lock:
            self._pane_tail_cache.pop(session_id, None)
        _sleep_ms(DANGER_AMEND_TAB_SETTLE_MS)
        # 2) paste the amend instruction.
        if msg:
            self._paste_text_to_pane(session_id, msg)
            _sleep_ms(DANGER_AMEND_PASTE_SETTLE_MS)
        # 3) submit.
        for i in range(SUBMIT_ENTER_ATTEMPTS):
            self.ops.tmux(["send-keys", "-t", target, "Enter"])
            if i < SUBMIT_ENTER_ATTEMPTS - 1:
                _sleep_ms(SUBMIT_ENTER_INTERVAL_MS)
        with self._cache_lock:
            self._pane_tail_cache.pop(session_id, None)
        self._log(f"[claude-code-driver] danger permission amend 已提交 (session={session_id})")

    # ── internal impls ────────────────────────────────────
    def _create_impl(self, opts: Dict[str, Any]) -> Dict[str, Any]:
        session_id = opts.get("session_id") or opts.get("sessionId")
        cwd = opts.get("cwd")
        initial_prompt = opts.get("initialPrompt") or opts.get("initial_prompt")
        if not session_id or not cwd:
            raise ValueError("createNewSession 需要 session_id + cwd")
        if not initial_prompt:
            raise ValueError("createNewSession 需要 initialPrompt")
        if not os.path.exists(cwd):
            raise RuntimeError(f"cwd 不存在: {cwd}")

        flag_root = opts.get("flagRoot") or opts.get("flag_root") or cwd
        model = opts.get("model")
        use_proxy = opts.get("useProxy", opts.get("use_proxy"))
        display_name = opts.get("displayName") or opts.get("display_name")
        agent_session_id = opts.get("agentSessionId") or opts.get("agent_session_id")
        is_initial_context = bool(opts.get("isInitialContextPrompt",
                                           opts.get("is_initial_context_prompt", False)))
        settings_path = opts.get("settingsPath") or opts.get("settings_path")
        force_no_proxy = bool(opts.get("forceNoProxy", opts.get("force_no_proxy", False)))

        # tmux windows survive a driver restart. An existing live window is
        # reused (idempotent), unlike stream-json's strict-new semantics.
        if not self._window_exists(session_id):
            self._spawn_window(session_id=session_id, cwd=cwd, flag_root=flag_root,
                               model=model, use_proxy=use_proxy,
                               display_name=display_name,
                               agent_session_id=agent_session_id,
                               settings_path=settings_path,
                               force_no_proxy=force_no_proxy)
        elif agent_session_id and session_id not in self.runtime:
            # window alive but runtime entry missing (first reload): backfill one.
            jp = _jsonl_path_of(self.cfg.home, cwd, agent_session_id or "")
            final_settings = settings_path or None
            final_force_no_proxy = force_no_proxy or bool(final_settings)
            final_use_proxy = (False if final_force_no_proxy
                               else _normalize_use_proxy(use_proxy, False))
            self.runtime[session_id] = {
                "agentSessionId": agent_session_id, "cwd": cwd,
                "flagRoot": flag_root or cwd, "model": model or None,
                "useProxy": final_use_proxy, "settingsPath": final_settings,
                "forceNoProxy": final_force_no_proxy, "displayName": display_name or None,
                "jsonlPath": jp, "startedAt": _now_ms(), "watch": None,
            }
            self._persist_entry(session_id, {
                "agentSessionId": agent_session_id, "cwd": cwd,
                "flagRoot": flag_root or cwd, "model": model,
                "useProxy": final_use_proxy, "settingsPath": final_settings,
                "forceNoProxy": final_force_no_proxy, "displayName": display_name,
                "jsonlPath": jp, "startedAt": _now_ms(),
            })
            self._ensure_watcher(session_id)

        entry = self.runtime.get(session_id)
        self._send_maybe_initial_context_prompt(session_id, initial_prompt, is_initial_context)
        self._mark_running(flag_root or (entry or {}).get("flagRoot")
                           or (entry or {}).get("cwd") or cwd, session_id)
        return {
            "session_id": session_id,
            "agent_session_id": (entry or {}).get("agentSessionId"),
            "jsonl_path": (entry or {}).get("jsonlPath"),
            "started_at": (entry or {}).get("startedAt") or _now_ms(),
        }

    def _queue_impl(self, opts: Dict[str, Any]) -> None:
        """Lenient 'send': if no live window, (re)spawn from opts/persisted, then
        queue the prompt. Chat doesn't distinguish first vs subsequent."""
        session_id = opts.get("session_id") or opts.get("sessionId")
        prompt = opts.get("prompt")
        if not session_id:
            raise ValueError("需要 session_id")
        if not prompt:
            raise ValueError("需要 prompt")

        if not self._window_exists(session_id):
            persisted = self.runtime.get(session_id)
            cwd = opts.get("cwd") or (persisted or {}).get("cwd")
            agent_sid = (opts.get("agentSessionId") or opts.get("agent_session_id")
                         or (persisted or {}).get("agentSessionId"))
            settings_path = (opts.get("settingsPath") or opts.get("settings_path")
                             or (persisted or {}).get("settingsPath") or None)
            force_no_proxy = (bool(opts.get("forceNoProxy", opts.get("force_no_proxy", False)))
                              or bool((persisted or {}).get("forceNoProxy"))
                              or bool(settings_path))
            use_proxy = (False if force_no_proxy
                         else _normalize_use_proxy(opts.get("useProxy", opts.get("use_proxy")),
                                                   (persisted or {}).get("useProxy", False)))
            if not cwd:
                raise RuntimeError(f"session {session_id} 没活 window 且无 cwd, 无法 spawn")
            self._spawn_window(session_id=session_id, cwd=cwd,
                               flag_root=opts.get("flagRoot") or opts.get("flag_root")
                               or (persisted or {}).get("flagRoot") or cwd,
                               model=opts.get("model") or (persisted or {}).get("model"),
                               use_proxy=use_proxy,
                               settings_path=settings_path,
                               force_no_proxy=force_no_proxy,
                               display_name=opts.get("displayName") or opts.get("display_name")
                               or (persisted or {}).get("displayName"),
                               agent_session_id=agent_sid)
        self._append_mobius_prompt_entry(session_id, opts.get("mobiusJsonl") or opts.get("mobius_jsonl"))
        is_initial = bool(opts.get("isInitialContextPrompt",
                                   opts.get("is_initial_context_prompt", False)))
        self._send_maybe_initial_context_prompt(session_id, prompt, is_initial)
        entry = self.runtime.get(session_id)
        self._mark_running(opts.get("flagRoot") or opts.get("flag_root")
                           or (entry or {}).get("flagRoot")
                           or (entry or {}).get("cwd") or opts.get("cwd"), session_id)

    def _pause_impl(self, opts: Dict[str, Any]) -> None:
        session_id = opts.get("session_id") or opts.get("sessionId")
        prompt = opts.get("prompt")
        urgent = bool(opts.get("urgent", False))
        if not session_id:
            raise ValueError("需要 session_id")
        persisted = self.runtime.get(session_id) or {}

        if self._window_exists(session_id):
            if urgent:
                # urgent: single C-c interrupts the current turn (one is enough).
                self.ops.tmux(["send-keys", "-t", f"{self.cfg.hub}:{session_id}", "C-c"])
                _sleep_ms(250)
                # Alt+Enter newline separates leftover input from the new prompt.
                self.ops.tmux(["send-keys", "-t", f"{self.cfg.hub}:{session_id}", "M-Enter"])
                _sleep_ms(80)
            else:
                # /stop: 3x C-c (one is eaten by the TUI empirically).
                for i in range(3):
                    self.ops.tmux(["send-keys", "-t", f"{self.cfg.hub}:{session_id}", "C-c"])
                    if i < 2:
                        _sleep_ms(50)
                _sleep_ms(300)
                # Fallback: if C-cx3 didn't take (TUI swallowed/stuck on a
                # dialog) and this is a soft-stop (no new prompt), escalate to
                # kill-window so /stop always halts the agent. Double-confirm
                # (busy -> wait 700ms -> still busy) to avoid killing a window
                # that was merely between status-line frames.
                if not prompt:
                    with self._cache_lock:
                        self._pane_tail_cache.pop(session_id, None)
                    if self._claude_pane_still_busy(session_id):
                        _sleep_ms(700)
                        with self._cache_lock:
                            self._pane_tail_cache.pop(session_id, None)
                        if self._window_exists(session_id) and self._claude_pane_still_busy(session_id):
                            self.ops.tmux(["kill-window", "-t", f"{self.cfg.hub}:{session_id}"])
                            self._invalidate_window_cache(session_id)
                            self._log(f"[claude-code-driver] /stop fallback: C-c×3 未停止, "
                                      f"kill-window={session_id}")

        if not prompt:
            self._clear_running(opts.get("flagRoot") or opts.get("flag_root")
                                or persisted.get("flagRoot") or persisted.get("cwd")
                                or opts.get("cwd"), session_id)
            return  # empty prompt = interrupt only, don't send

        # queue path (includes respawn-if-dead logic)
        self._queue_impl({
            "session_id": session_id,
            "prompt": prompt,
            "cwd": persisted.get("cwd"),
            "flagRoot": persisted.get("flagRoot"),
            "model": persisted.get("model"),
            "useProxy": persisted.get("useProxy"),
            "displayName": persisted.get("displayName"),
            "agentSessionId": persisted.get("agentSessionId"),
            "isInitialContextPrompt": False,
            "mobiusJsonl": opts.get("mobiusJsonl") or opts.get("mobius_jsonl"),
        })

    def _terminate_impl(self, session_id: str) -> Dict[str, Any]:
        was_alive = self._window_exists(session_id)
        was_working = was_alive and self.is_working(session_id)
        entry = self.runtime.get(session_id)
        watcher = self._watchers.pop(session_id, None)
        if watcher:
            try:
                watcher.stop()
            except Exception:
                pass
        self.runtime.pop(session_id, None)
        self._forget_persisted(session_id)
        if was_alive:
            self.ops.tmux(["kill-window", "-t", f"{self.cfg.hub}:{session_id}"])
            self._invalidate_window_cache(session_id)
            self._log(f"[claude-code-driver] terminate: killed window={session_id} "
                      f"(wasWorking={was_working})")
        flag_root = (entry or {}).get("flagRoot") or (entry or {}).get("cwd")
        if flag_root:
            safe_remove_flag_dir(flag_root, session_id, self.cfg.hidden_folder,
                                 "claude-code-driver")
        return {"session_id": session_id, "killed": was_alive, "was_working": was_working}

    # ── tmux low-level: spawn a window ────────────────────
    def _spawn_window(self, *, session_id: str, cwd: str, flag_root: Optional[str],
                      model: Optional[str], use_proxy: Any, display_name: Optional[str],
                      agent_session_id: Optional[str], settings_path: Optional[str],
                      force_no_proxy: bool = False) -> None:
        self._ensure_hub()
        eff_flag_root = flag_root or cwd
        final_settings_path = os.path.abspath(settings_path) if settings_path else None
        if final_settings_path and not os.path.exists(final_settings_path):
            raise RuntimeError(f"Claude Code settings 文件不存在: {final_settings_path}")
        final_force_no_proxy = bool(force_no_proxy) or bool(final_settings_path)
        final_use_proxy = (False if final_force_no_proxy
                           else _normalize_use_proxy(use_proxy, False))
        if final_use_proxy:
            self._assert_proxy_available()

        # resume protection: an old session's jsonl might not be under our path.
        use_resume = bool(agent_session_id)
        if use_resume and not os.path.exists(
                _jsonl_path_of(self.cfg.home, cwd, agent_session_id or "")):
            print(f"[claude-code-driver] resume target jsonl 不存在 ({agent_session_id}), "
                  f"fallback 为新 session", file=sys.stderr)
            use_resume = False
        claude_session_id: str = agent_session_id if use_resume else str(uuid.uuid4())

        claude_args = [
            "--dangerously-skip-permissions",
            # deny AskUserQuestion (never block to ask a human) + ExitPlanMode
            # (don't get stuck waiting for plan approval).
            "--disallowedTools AskUserQuestion,ExitPlanMode",
            f"--resume {claude_session_id}" if use_resume else f"--session-id {claude_session_id}",
        ]
        if model:
            claude_args.append(f"--model {shell_quote(model)}")
        settings_arg = (f"--settings {shell_quote(final_settings_path)}"
                        if final_settings_path
                        else '--settings "$HOME/.claude/mobiusdefault.settings.json"')

        cmd_parts = []
        # Propagate the config-dir override to the spawned claude process so its
        # transcripts land under <claude_config_dir>/projects/<enc-cwd>/ too.
        _ccd = self.cfg.claude_config_dir or os.environ.get("CLAUDE_CONFIG_DIR")
        if _ccd:
            cmd_parts.append(f"export CLAUDE_CONFIG_DIR={shell_quote(_ccd)}")
        if final_use_proxy:
            cmd_parts.append('source "$HOME/proxy_envs.bash"')
        cmd_parts.append("unset VSCODE_IPC_HOOK_CLI VSCODE_GIT_IPC_HANDLE "
                         "VSCODE_GIT_ASKPASS_NODE VSCODE_GIT_ASKPASS_MAIN")
        cmd_parts.append("export IS_SANDBOX=1")
        if final_use_proxy:
            cmd_parts.append(
                f'exec proxychains -q -f "$HOME/proxy_claude.conf" claude '
                f'{settings_arg} {" ".join(claude_args)}')
        else:
            cmd_parts.append(f"exec claude {settings_arg} {' '.join(claude_args)}")
        cmd = " && ".join(cmd_parts)

        # Main path: pre-trust cwd so the TUI never pops the trust dialog.
        ensure_project_trusted(cwd, self.cfg.home, self._log)

        r = self.ops.tmux(["new-window", "-d", "-t", self.cfg.hub, "-n", session_id,
                           "-c", cwd, "bash", "-lc", cmd])
        if r.status != 0:
            raise RuntimeError(f"tmux new-window 失败: {r.stderr}")
        self._invalidate_window_cache()
        self._log(f"[claude-code-driver] started: window={session_id} cwd={cwd} "
                  f"claude_session={claude_session_id} "
                  f"use_proxy={1 if final_use_proxy else 0}"
                  + (f" settings={final_settings_path}" if final_settings_path else ""))
        self._start_pane_capture(session_id)

        # Wait for TUI ready (status bar "bypass permissions on"), auto-confirming
        # any blocking dialogs (trust / onboarding / api-key / bypass-warn) en route.
        target = f"{self.cfg.hub}:{session_id}"
        deadline = _now_ms() + READY_TIMEOUT_MS
        ready = False
        last_trust = last_onboard = last_apikey = last_bypass = 0
        while _now_ms() < deadline:
            screen = take_tmux_window_text(self.ops, target, 100)
            if READY_SENTINEL in screen:
                ready = True
                break
            now = _now_ms()
            if any(s in screen for s in TRUST_PROMPT_SENTINELS):
                if now - last_trust > TRUST_PRESS_INTERVAL_MS:
                    self.ops.tmux(["send-keys", "-t", target, "Enter"])
                    last_trust = now
                    self._log(f"[claude-code-driver] window={session_id} 检测到目录信任对话框, "
                              f"已自动确认信任 (cwd={cwd})")
            if any(s in screen for s in ONBOARDING_PROMPT_SENTINELS):
                if now - last_onboard > ONBOARDING_PRESS_INTERVAL_MS:
                    self.ops.tmux(["send-keys", "-t", target, "Enter"])
                    last_onboard = now
                    self._log(f"[claude-code-driver] window={session_id} 检测到首次启动引导对话框, 已自动确认")
            if any(s in screen for s in API_KEY_PROMPT_SENTINELS):
                if now - last_apikey > API_KEY_PRESS_INTERVAL_MS:
                    self.ops.tmux(["send-keys", "-t", target, "1"])
                    self.ops.tmux(["send-keys", "-t", target, "Enter"])
                    last_apikey = now
                    self._log(f"[claude-code-driver] window={session_id} 检测到 API Key 对话框, "
                              f"已自动选择使用环境变量 Key")
            if any(s in screen for s in BYPASS_WARN_SENTINELS):
                if now - last_bypass > BYPASS_WARN_INTERVAL_MS:
                    self.ops.tmux(["send-keys", "-t", target, "2"])
                    self.ops.tmux(["send-keys", "-t", target, "Enter"])
                    last_bypass = now
                    self._log(f"[claude-code-driver] window={session_id} 检测到 Bypass Permissions 警告, "
                              f"已自动确认接受")
            _sleep_ms(READY_POLL_MS)

        if not ready:
            self.ops.tmux(["kill-window", "-t", target])
            raise RuntimeError(f"claude TUI 未在 {READY_TIMEOUT_MS}ms 内 ready (cwd={cwd}).")
        self._log(f"[claude-code-driver] window={session_id} TUI ready")

        jp = _jsonl_path_of(self.cfg.home, cwd, claude_session_id)
        self.runtime[session_id] = {
            "agentSessionId": claude_session_id,
            "cwd": cwd, "flagRoot": eff_flag_root, "model": model or None,
            "useProxy": final_use_proxy, "settingsPath": final_settings_path,
            "forceNoProxy": final_force_no_proxy, "displayName": display_name or None,
            "jsonlPath": jp, "startedAt": _now_ms(), "watch": None,
        }
        self._persist_entry(session_id, {
            "agentSessionId": claude_session_id, "cwd": cwd, "flagRoot": eff_flag_root,
            "model": model or None, "useProxy": final_use_proxy,
            "settingsPath": final_settings_path, "forceNoProxy": final_force_no_proxy,
            "displayName": display_name or None, "jsonlPath": jp, "startedAt": _now_ms(),
        })
        self._ensure_watcher(session_id)
        # window up -> drop a running flag now; refreshed on each prompt submit.
        # agent deletes it when done (per the caller's prompt instructions).
        self._mark_running(eff_flag_root, session_id)

    # ── prompt pasting ────────────────────────────────────
    def _send_maybe_initial_context_prompt(self, session_id: str, text: str,
                                           is_initial_context: bool) -> None:
        if not is_initial_context:
            self._send_prompt_to_window(session_id, text)
            return
        plan = _pick_initial_context_plan()
        if plan == "greeting_then_context":
            greeting = _pick_initial_context_greeting()
            self._log(f"[claude-code-driver] initial context plan={plan} "
                      f"greeting={greeting!r} delay_ms={INITIAL_CONTEXT_DELAY_MS}")
            self._send_prompt_to_window(session_id, greeting)
            _sleep_ms(INITIAL_CONTEXT_DELAY_MS)
            self._send_prompt_to_window(session_id, text)
            return
        if plan == "delay_then_context":
            self._log(f"[claude-code-driver] initial context plan={plan} "
                      f"delay_ms={INITIAL_CONTEXT_DELAY_MS}")
            _sleep_ms(INITIAL_CONTEXT_DELAY_MS)
            self._send_prompt_to_window(session_id, text)
            return
        self._log(f"[claude-code-driver] initial context plan={plan}")
        self._send_prompt_to_window(session_id, text)

    def _send_prompt_to_window(self, session_id: str, text: str) -> None:
        if not self._window_exists(session_id):
            raise RuntimeError(f"window {session_id} 不存在")
        target = f"{self.cfg.hub}:{session_id}"
        marker = _find_ascii_tail_marker(text)
        marker_repr = repr(marker) if marker is not None else "(none)"
        self._log(f"[claude-code-driver] sendPrompt window={session_id} "
                  f"len={len(text)} marker={marker_repr}")

        buf_name = f"imac_{os.getpid()}_{_now_ms()}"
        r1 = self.ops.tmux(["load-buffer", "-b", buf_name, "-"], input_text=text)
        if r1.status != 0:
            raise RuntimeError(f"tmux load-buffer 失败: {r1.stderr}")
        # -p = bracketed paste: without it, embedded newlines are submitted as
        # Enter and a multi-line message gets split into many. Idempotent resubmit
        # of Enter (below) handles the TUI occasionally eating the first Enter.
        r2 = self.ops.tmux(["paste-buffer", "-p", "-d", "-b", buf_name, "-t", target])
        if r2.status != 0:
            self.ops.tmux(["delete-buffer", "-b", buf_name])
            raise RuntimeError(f"tmux paste-buffer 失败: {r2.stderr}")

        if marker:
            deadline = _now_ms() + PASTE_PROBE_TIMEOUT_MS
            saw = False
            while _now_ms() < deadline:
                _sleep_ms(PASTE_PROBE_INTERVAL_MS)
                pane = self.ops.tmux(["capture-pane", "-pt", target, "-p", "-S", "-80"])
                if pane.status == 0 and marker in (pane.stdout or ""):
                    saw = True
                    break
            if not saw:
                print(f"[claude-code-driver] paste marker 未出现 "
                      f"({PASTE_PROBE_TIMEOUT_MS}ms 内), Enter 仍发送", file=sys.stderr)
        else:
            sleep_ms = min(PASTE_SLEEP_MAX_MS, max(PASTE_SLEEP_BASE_MS, len(text) // 2))
            _sleep_ms(sleep_ms)

        # Submit: bracketed paste is atomic, so extra Enters won't split the
        # message; resubmit N times because the TUI occasionally swallows the
        # first Enter when switching input modes. An Enter on an empty submitted
        # box is a no-op.
        for i in range(SUBMIT_ENTER_ATTEMPTS):
            r = self.ops.tmux(["send-keys", "-t", target, "Enter"])
            if r.status != 0:
                raise RuntimeError(f"tmux send-keys Enter 失败: {r.stderr}")
            if i < SUBMIT_ENTER_ATTEMPTS - 1:
                _sleep_ms(SUBMIT_ENTER_INTERVAL_MS)

        # prompt-paste recording hook (no-op by default; Mobius records to SQLite,
        # this independent driver has no DB - callers may override _on_prompt_paste).
        self._on_prompt_paste("claude-code-driver", session_id, len(text))

    def _paste_text_to_pane(self, session_id: str, text: str) -> None:
        """Paste arbitrary text into a pane's current input field via a throwaway
        tmux buffer (bracketed paste, atomic). Falls back to literal `send-keys
        -l` if the buffer path fails. Used to feed the amend instruction into the
        danger-permission box's amend input."""
        if not text:
            return
        target = f"{self.cfg.hub}:{session_id}"
        buf_name = f"imac_amend_{os.getpid()}_{_now_ms()}"
        r1 = self.ops.tmux(["load-buffer", "-b", buf_name, "-"], input_text=text)
        if r1.status != 0:
            self.ops.tmux(["send-keys", "-t", target, "-l", text])
            return
        r2 = self.ops.tmux(["paste-buffer", "-p", "-d", "-b", buf_name, "-t", target])
        if r2.status != 0:
            self.ops.tmux(["delete-buffer", "-b", buf_name])
            self.ops.tmux(["send-keys", "-t", target, "-l", text])

    def _on_prompt_paste(self, backend_name: str, session_id: str,
                         content_length: int) -> None:
        """Hook mirroring Mobius agent-prompt-events.recordPromptPaste. Default
        no-op (independent driver has no SQLite DB). Override on an instance to
        record paste events wherever you like."""
        return None


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# CLI (mirrors the JS test_tmux_claude_code wrapper)
# --------------------------------------------------------------------------- #
def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="claude_code",
        description="manual CLI for the tmux-based claude code driver",
    )
    ap.add_argument("--hub", default=None, help="tmux hub session name "
                    "(default: claude_code_agent_hub)")
    ap.add_argument("--data-path", default=None, help="state root "
                    "(default: $MOBIUS_DATA_PATH or /data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--sessionid", required=True)

    p_spawn = sub.add_parser("spawn", help="start (or reuse) a tmux window running claude")
    p_spawn.add_argument("--cwd", required=True)
    p_spawn.add_argument("--prompt", default=None)
    p_spawn.add_argument("--model", default=None)
    p_spawn.add_argument("--proxy", dest="proxy", action="store_true")
    p_spawn.add_argument("--no-proxy", dest="proxy", action="store_false")
    p_spawn.set_defaults(proxy=None)
    p_spawn.add_argument("--resume", default=None, help="claude session UUID to --resume")
    p_spawn.add_argument("--display-name", default=None)
    p_spawn.add_argument("--settings", default=None)
    add_common(p_spawn)

    p_send = sub.add_parser("send", help="queue a prompt (re-spawns if window died)")
    p_send.add_argument("--prompt", required=True)
    add_common(p_send)

    p_pause = sub.add_parser("pause", help="C-c x3 the running turn; optional new prompt")
    p_pause.add_argument("--prompt", default=None)
    p_pause.add_argument("--urgent", action="store_true")
    add_common(p_pause)

    p_kill = sub.add_parser("kill", help="kill-window + drop runtime entry")
    add_common(p_kill)

    sub.add_parser("list", help="print all hub windows with backend metadata")

    p_status = sub.add_parser("status", help="print alive/working/accomplished/failed")
    add_common(p_status)

    p_hist = sub.add_parser("history", help="dump jsonl entries")
    p_hist.add_argument("--limit", type=int, default=0)
    add_common(p_hist)

    p_attach = sub.add_parser("attach", help="print the tmux attach command")
    add_common(p_attach)

    args = ap.parse_args()
    cfg = DriverConfig()
    if args.hub:
        cfg.hub = args.hub
    if args.data_path:
        cfg.data_path = args.data_path
        cfg.runtime_file = os.path.join(args.data_path, "claude-code-driver-runtime.json")
        cfg.archive_file = os.path.join(args.data_path, "claude-code-driver-archive.json")
        cfg.tmux_log_file = os.path.join(args.data_path, "logs", "tmux-operation.log")
    driver = TmuxClaudeCodeDriver(cfg)
    driver.start()

    def _print(obj: Any) -> None:
        if isinstance(obj, str):
            print(obj)
        else:
            print(json.dumps(obj, ensure_ascii=False, indent=2))

    try:
        if args.cmd == "spawn":
            opts = {
                "session_id": args.sessionid,
                "cwd": os.path.abspath(args.cwd),
                "model": args.model,
                "useProxy": args.proxy,
                "displayName": args.display_name,
                "agentSessionId": args.resume,
                "settingsPath": args.settings,
            }
            if args.prompt:
                _print(driver.create_new_session({**opts, "initialPrompt": args.prompt}))
            else:
                with driver._lock_for(args.sessionid):
                    driver._spawn_window(session_id=args.sessionid, cwd=os.path.abspath(args.cwd),
                                         flag_root=os.path.abspath(args.cwd), model=args.model,
                                         use_proxy=args.proxy, display_name=args.display_name,
                                         agent_session_id=args.resume, settings_path=args.settings)
                e = driver.runtime.get(args.sessionid)
                _print({"session_id": args.sessionid,
                        "agent_session_id": (e or {}).get("agentSessionId"),
                        "jsonl_path": (e or {}).get("jsonlPath")})
        elif args.cmd == "send":
            driver.no_pause_current_and_queue_query_at_session(
                {"session_id": args.sessionid, "prompt": args.prompt})
            _print({"ok": True, "session_id": args.sessionid})
        elif args.cmd == "pause":
            driver.pause_current_and_resume_from_session(
                {"session_id": args.sessionid, "prompt": args.prompt, "urgent": args.urgent})
            _print({"ok": True, "session_id": args.sessionid})
        elif args.cmd == "kill":
            _print(driver.terminate_session(args.sessionid))
        elif args.cmd == "list":
            _print(driver.list_sessions())
        elif args.cmd == "status":
            _print({
                "session_id": args.sessionid,
                "alive": driver.is_alive(args.sessionid),
                "working": driver.is_working(args.sessionid),
                "accomplished": driver.is_job_goal_accomplished(args.sessionid),
                "failed": driver.is_failed(args.sessionid),
            })
        elif args.cmd == "history":
            h = driver.get_history(args.sessionid)
            limit = args.limit if args.limit > 0 else len(h["entries"])
            entries = h["entries"][-limit:] if limit < len(h["entries"]) else h["entries"]
            _print({"session_id": args.sessionid, "total": h["total"],
                    "truncated": h["truncated"], "sentinel": h["sentinel"],
                    "shown": len(entries), "entries": entries})
        elif args.cmd == "attach":
            _print(f"tmux attach -t {cfg.hub} \\; select-window -t {args.sessionid}")
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        driver.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
