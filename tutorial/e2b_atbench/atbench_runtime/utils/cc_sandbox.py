"""Claude-Code 沙盒 rollout 的辅助函数, 从 generate_claudecode.py 拆出。

让 generate_claudecode.py 只剩 "prompt 模板 + _run_claudecode_in_sandbox" 的
编排逻辑; 这里放被它复用的内部工具: 沙盒探活、verdict->reward 解析、
清理路径表、文件树打印器。

本模块只依赖标准库 (与 utils/ 其它子模块一致, 不引 slime), 因此 _wait_sandbox_ready
的 sb 参数不加类型注解。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)


async def _wait_sandbox_ready(sb, max_attempts: int = 40, interval: float = 3.0) -> None:
    """命令级探活: PAI-EAS 创建后 is_running 可能假 ready, 用 commands.run 确认真正可执行。"""
    for i in range(1, max_attempts + 1):
        try:
            r = await sb._sb.commands.run("echo ready", timeout=15)
            if r.stdout.strip() == "ready":
                return
        except Exception as e:
            last = e
        await asyncio.sleep(interval)
    raise RuntimeError(f"sandbox not ready after {max_attempts} polls: {last if 'last' in dir() else 'unknown'}")


def _cc_verdict_to_reward(verdict: str) -> float:
    m = re.search(r"\*\*Verdict:\*\*\s*(PASS|FAIL)", verdict or "", re.IGNORECASE)
    return 1.0 if (m and m.group(1).upper() == "PASS") else 0.0


def _write_verdict_local(verdict_text: str) -> None:
    """judge 完成后把 verdict.md 内容落到宿主机本地文件。

    沙盒里的 /root/task_dir/verdict.md 在进程结束/沙盒回收后就丢, 此前只在
    _run_claudecode_in_sandbox 里读成字符串算 reward, 没落盘 -> 事后想看 judge
    到底写了啥 (尤其 NONE: verdict 为空/不完整) 只能翻 worker 日志里的 verdict_head。
    这里写成本地文件: 路径优先 CC_VERDICT_FILE (批跑时每 job 一个, 如 verdicts/T09_R1.md),
    否则落进 CC_STATE_DIR/verdict.md, 再否则 tmp/verdict-<pid>.md。
    全程容错: 写盘失败只 warning, 不拖垮 rollout 收尾。空 verdict 也写 (NONE 的直接证据)。
    """
    path = os.environ.get("CC_VERDICT_FILE", "").strip()
    if not path:
        state_dir = os.environ.get("CC_STATE_DIR", "").strip()
        if state_dir:
            path = os.path.join(state_dir, "verdict.md")
        else:
            path = os.path.join("tmp", f"verdict-{os.getpid()}.md")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(verdict_text or "")
        logger.info("[cc_state] verdict.md 已落本地 -> %s (%d 字节)",
                    path, len(verdict_text or ""))
    except Exception as e:
        logger.warning("[cc_state] 写 verdict.md 本地副本失败(忽略): %s", str(e)[:120])


# 沙盒里 claude-code 留下的诊断物: 抓这些回来就能定位"卡在哪一步"。
#   /root/cc_data/state.json    -> spawn_and_wait_agent 的 update_state_json 尾部
#                                 (create claude code | waiting claude code finish | terminate)
#   /root/cc_data/*.jsonl       -> claude 转写流, is_working 靠它判 end_turn; 末尾几行说明
#                                 最后一条是 assistant(tool_use=还活着) 还是 end_turn(完成了)
#   tmux pane                   -> claude TUI 当前屏幕 (是不是停在权限提示 / 空转)
_CC_STATE_PATHS = ("/root/cc_data/state.json",)
# jsonl 不用 files API 的 glob (e2b async list 只吃目录不吃 glob), 直接 find 进沙盒跑,
# 一次命令把 cc_data + ~/.claude/projects 下的 jsonl 全列出来再各自 tail。
_CC_JSONL_FIND = "/root/cc_data /root/.claude/projects 2>/dev/null"
_CC_TMUX_PANE_DUMP = (
    # 把所有 tmux pane 内容抓出来: 这一步直接显示 claude 屏幕"卡在哪"。
    "for s in $(tmux ls -F '#{session_name}' 2>/dev/null); do "
    "  for p in $(tmux list-panes -t $s -F '#{pane_id}' 2>/dev/null); do "
    "    echo '----- tmux session='$s' pane='$p' -----'; "
    "    tmux capture-pane -p -t $p -S -200 2>/dev/null; "
    "  done; "
    "done"
)


async def _capture_sandbox_state(sb, dest_dir: str, tag: str = "") -> None:
    """把沙盒里 claude-code 的诊断状态抓回宿主机 dest_dir, 便于事后定位"卡哪一步"。

    抓三样 (都是事后判定 stall 的关键证据, host 侧 proxy 日志看不到):
      1. state.json        — spawn_and_wait_agent 的状态机尾部 (create/waiting/terminate)
      2. claude 转写 jsonl  — 末尾几条决定 is_working 的判定 (end_turn=完成 / tool_use=还在跑)
      3. tmux pane          — claude TUI 当前屏幕 (权限提示 / 空转 / 报错 都在这)

    全程容错: 沙盒死透时任意一步失败都不抛, 只记 warning, 免得拖垮主流程收尾。
    """
    os.makedirs(dest_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(dest_dir, f"{ts}-{tag}" if tag else ts)

    # 1) state.json (update_state_json 写的, 是"卡住时 spawn_and_wait 在干嘛"的直接证据)。
    #    state.json 里还自带 tmux_tail (写 state.json 时顺带 capture-pane 抓的 claude 屏幕
    #    尾部 ~30 行) —— 把它单独存一份 -tmux.txt, 这样即便下面第 3 步 host 侧直接抓 tmux
    #    pane 失败 (沙盒死透/tmux 已退出), "状态机最后跳变那一刻的屏幕"仍能捞回来。
    for p in _CC_STATE_PATHS:
        try:
            txt = await sb._sb.files.read(p, format="text")
            with open(f"{base}-state.json", "w", encoding="utf-8") as f:
                f.write(txt or "")
            # 解出 tmux_tail 单独落盘 (原始 state.json 是单行 JSON, 不便肉眼看屏幕)
            try:
                o = json.loads(txt) if txt else {}
                tail = str((o or {}).get("tmux_tail", "") or "")
                if tail:
                    with open(f"{base}-tmux.txt", "w", encoding="utf-8") as f:
                        f.write(tail)
            except Exception:
                pass
        except Exception as e:
            logger.warning("[cc_state] 读 %s 失败: %s", p, str(e)[:120])

    # 2) 各 jsonl 的末尾几行 (is_working 反向扫最后一条 assistant.stop_reason 判定是否完成)
    #    find -name '*.jsonl' 列路径 -> 对每条 tail -n 15, 合到一段 stdout 里 (分块 echo 分隔)
    try:
        r = await sb._sb.commands.run(
            f"for f in $(find {_CC_JSONL_FIND} -name '*.jsonl' 2>/dev/null); do "
            f"  echo '----- jsonl '$f' -----'; tail -n 15 \"$f\" 2>/dev/null; "
            f"done",
            timeout=30)
        if r.stdout:
            with open(f"{base}-jsonl.tails", "w", encoding="utf-8") as f:
                f.write(r.stdout)
    except Exception as e:
        logger.warning("[cc_state] 抓 jsonl 末尾失败: %s", str(e)[:120])

    # 3) tmux pane 当前屏幕 (claude 屏幕"卡在哪"的唯一肉眼证据)
    try:
        r = await sb._sb.commands.run(_CC_TMUX_PANE_DUMP, timeout=30)
        with open(f"{base}-tmux.txt", "w", encoding="utf-8") as f:
            f.write((r.stdout or ""))
    except Exception as e:
        logger.warning("[cc_state] 抓 tmux pane 失败: %s", str(e)[:120])
    logger.info("[cc_state] 沙盒诊断已抓回 -> %s-*", base)


async def _download_jsonl(sb, dest_dir: str, tag: str) -> None:
    """stage 正常结束后, 把沙箱内 claude 转写 jsonl 完整拉回宿主机落盘。

    jsonl 只存在于沙箱 (episode 结束 __aexit__ kill 后永久丢失), 所以必须在
    stage done 之后、沙箱销毁之前取。实测本环境转写在 /root/.claude/projects
    (claude-code 原生 session 转写, /root/cc_data 下只有 driver runtime 无
    jsonl), 因此默认两个根都拉。文件名带时间戳前缀, 多 episode 互不覆盖。
    全程容错: 任一步失败只 warning 不抛。
    """
    roots = "/root/cc_data /root/.claude/projects"
    try:
        r = await sb._sb.commands.run(f"find {roots} -name '*.jsonl' 2>/dev/null", timeout=30)
        paths = (r.stdout or "").split()
    except Exception as e:
        logger.warning("[cc_state] jsonl 枚举失败(%s): %s", tag, str(e)[:120])
        return
    os.makedirs(dest_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    n = 0
    for p in paths:
        try:
            txt = await sb._sb.files.read(p, format="text")
            with open(os.path.join(dest_dir, f"{ts}-{tag}-{p.replace('/', '_')}.jsonl"),
                      "w", encoding="utf-8") as f:
                f.write(txt or "")
            n += 1
        except Exception as e:
            logger.warning("[cc_state] jsonl %s 拉取失败: %s", p, str(e)[:120])
    logger.info("[cc_state] %s 完整 jsonl 已落盘 %d 个 -> %s", tag, n, dest_dir)


async def _poll_sandbox_state(sb, interval_sec: float = 5.0) -> None:
    """后台每 interval_sec 秒打印一次沙盒 claude-code 的"当前状态"一行到日志。

    定位 stall 的关键: host 侧只能看到 proxy 的 LLM IO 摘要 (且只在请求返回后才更新),
    看不到 claude 此刻在沙盒里干嘛。这条轮询直接读沙盒内的:
      - state.json 的 msg 字段 (create / waiting finish / terminate) + 距上次更新的秒数
      - 最新 jsonl 末条 (assistant.end_turn=完成 / assistant.tool_use=还在跑 / user=轮到模型)
      - tmux pane 末非空行 (claude 屏幕: 等权限 / Esc 空转 / 报错 都在这)
    合成一行摘要打 logger.info。与 stage 并发跑, 任意子步失败只跳过本轮, 不抛。
    """
    import json as _json
    prev_state_msg = ""
    while True:
        try:
            # state.json: {ts, msg, tmux_tail} -> msg + 距现在的秒数 + state.json 里
            # 自带的 claude 屏幕尾部 (update_state_json 写 state.json 时顺带 capture-pane
            # 抓的, 是"状态机跳变那一刻"的屏幕, 比 host 侧再 capture-pane 更贴那一瞬)。
            state_age = "?"
            state_msg = "(no state.json)"
            state_tmux_tail = ""
            try:
                txt = await sb._sb.files.read("/root/cc_data/state.json", format="text")
                o = _json.loads(txt) if txt else {}
                state_msg = str(o.get("msg", "")) or state_msg
                ts = o.get("ts")
                if isinstance(ts, (int, float)):
                    state_age = f"{time.time() - ts:.0f}s"
                state_tmux_tail = str(o.get("tmux_tail", "") or "")
            except Exception:
                pass
            # 最新 jsonl 末条类型 (is_working 的判定依据)
            jsonl_tail = "(no jsonl)"
            try:
                r = await sb._sb.commands.run(
                    "for f in $(find /root/cc_data /root/.claude/projects "
                    "-name '*.jsonl' 2>/dev/null); do "
                    "echo \"$f\"; tail -n 1 \"$f\"; break; done", timeout=15)
                lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
                if len(lines) >= 2:
                    try:
                        e = _json.loads(lines[1])
                        t = e.get("type", "?") if isinstance(e, dict) else "?"
                        sr = ((e.get("message") or {}).get("stop_reason")
                              if isinstance(e, dict) else None)
                        jsonl_tail = f"{t}" + (f"/{sr}" if sr else "")
                    except Exception:
                        jsonl_tail = lines[1][:60]
            except Exception:
                pass
            # tmux pane 末非空行 (claude 屏幕当前状态)。优先用 state.json 自带的
            # tmux_tail (那一刻的快照), 缺了再 host 侧 capture-pane 实抓。
            pane_tail = "(no tmux)"
            if state_tmux_tail:
                nonblank = [ln for ln in state_tmux_tail.splitlines() if ln.strip()]
                pane_tail = (nonblank[-1][:80] if nonblank else "")
            if not pane_tail or pane_tail == "(no tmux)":
                try:
                    r = await sb._sb.commands.run(
                        "tmux capture-pane -p -S -50 2>/dev/null | grep -v '^$' | tail -1",
                        timeout=15)
                    pane_tail = (r.stdout or "").strip()[:80] or pane_tail
                except Exception:
                    pass
            # 只在状态 msg 变了时记 INFO, 否则 DEBUG, 免得刷屏
            changed = state_msg != prev_state_msg
            prev_state_msg = state_msg
            (logger.info if changed else logger.debug)(
                "[cc_state] state=%s (age %s) | jsonl_tail=%s | pane=%r",
                state_msg, state_age, jsonl_tail, pane_tail)
            # 状态 msg 变了时, 顺带把 state.json 自带的完整 tmux_tail 多行也吐到日志,
            # 这样 stall 时一眼看到 claude 屏幕全貌 (权限框/报错/status line), 不只末行。
            if changed and state_tmux_tail:
                tail = state_tmux_tail[-1200:]
                logger.info("[cc_state] tmux_tail(state.json):\n%s", tail)
        except Exception as e:
            logger.warning("[cc_state] 轮询失败: %s", str(e)[:120])
        await asyncio.sleep(interval_sec)


# claudecode 路径用不到的 qwenpaw 模板残留, INIT 前清掉 (免得污染 /root 文件树)
_CC_CLEANUP_PATHS = (
    "/root/agent_runner.py", "/root/bench_client.py", "/root/copaw_eval", "/root/core",
    "/root/export_training_data.py", "/root/judge.py", "/root/judge_reward.py",
    "/root/otel_init.py", "/root/patch", "/root/run.py", "/root/setup_otel.sh",
    "/root/setup_provider.py", "/root/test_judge_reward.py",
)

# 沙盒内文件树打印器 (ASCII ├─/└─, 目录优先, 隐藏 .*), 用法: python3 _cc_tree.py <root> [maxdepth]
_CC_TREE_PY = """\
import os, sys
root = sys.argv[1] if len(sys.argv) > 1 else '/root'
maxd = int(sys.argv[2]) if len(sys.argv) > 2 else 6
def build(p, d):
    t = {}
    if d >= maxd:
        return t
    try:
        entries = list(os.scandir(p))
    except Exception:
        return t
    for e in sorted(entries, key=lambda x: x.name):
        if e.name.startswith('.'):  # 隐藏 .config/.cache/.gnupg 等噪音
            continue
        t[e.name] = build(e.path, d + 1) if e.is_dir() else {}
    return t
def render(node, prefix=''):
    items = sorted(node.items(), key=lambda kv: (not kv[1], kv[0]))  # 目录优先, 再按名
    for i, (n, c) in enumerate(items):
        last = (i == len(items) - 1)
        print(prefix + ('`-- ' if last else '|-- ') + n + ('/' if c else ''))
        if c:
            render(c, prefix + ('    ' if last else '|   '))
if not os.path.isdir(root):
    print(root + '  (not exist)')
else:
    print(root)
    render(build(root, 0))
"""


# 沙盒里 claude-code 的 trajectory (训练/复盘用的完整转写流)。
# judge 阶段会复用同一个沙盒开新 session, 也往这两个目录写新 jsonl, 所以必须在
# judge staging 之前 (solver 跑完、judge 还没开) 就把 solver 的 trajectory 取走。
_CC_TRAJ_PATHS = ("/root/.claude/projects", "/root/cc_data")


async def _save_trajectory_local(sb, stage: str = "solver") -> str | None:
    """把沙盒里 claude-code 的 trajectory (jsonl 转写流) 打包取回宿主机落盘。

    在 judge staging 之前调: 此时 /root/.claude/projects 和 /root/cc_data 下只有
    solver 的转写, 取走的是干净的 solver trajectory; judge 跑完后再取会混在一起。

    落盘路径: 优先 CC_TRAJECTORY_FILE; 否则 <CC_JSONL_DIR 父目录>/trajectories/
    trajectory-<stage>-<sandbox_id 前 8 位>.tar.gz —— swarm 多线程并发 rollout 共享
    进程级 env, 默认名必须带 sandbox id 防互相覆盖; 批跑 (每 job 子进程) 用
    CC_TRAJECTORY_FILE 每 job 唯一路径。方式: 沙盒内 tar czf -> files.read(bytes)
    取回 -> 本地解包保留 root/.claude/... 结构。全程容错, 失败只 warning。返回 tar 路径。
    """
    sand_tar = f"/tmp/_cc_traj_{stage}.tar.gz"
    paths = " ".join(p.lstrip("/") for p in _CC_TRAJ_PATHS)
    try:
        r = await sb._sb.commands.run(
            f"tar czf {sand_tar} -C / {paths} 2>/dev/null; "
            f"echo size=$(wc -c < {sand_tar} 2>/dev/null || echo 0)", timeout=120)
        logger.info("[cc_traj] %s trajectory 打包: %s", stage, (r.stdout or "")[:120])
    except Exception as e:
        logger.warning("[cc_traj] %s 沙盒内打包 trajectory 失败(忽略): %s", stage, str(e)[:160])
        return None
    try:
        data = await sb._sb.files.read(sand_tar, format="bytes", request_timeout=600)
    except Exception as e:
        logger.warning("[cc_traj] %s 取回 trajectory tar 失败(忽略): %s", stage, str(e)[:160])
        return None
    if not data:
        logger.warning("[cc_traj] %s trajectory tar 为空(忽略)", stage)
        return None
    tar_path = os.environ.get("CC_TRAJECTORY_FILE", "").strip()
    if not tar_path:
        sbid = ""
        try:
            sbid = "-" + str(getattr(sb._sb, "sandbox_id", "") or "")[:8]
        except Exception:
            pass
        base_dir = os.path.dirname(os.environ.get("CC_JSONL_DIR", "").rstrip("/")) or "tmp"
        tar_path = os.path.join(base_dir, "trajectories", f"trajectory-{stage}{sbid}.tar.gz")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(tar_path)), exist_ok=True)
        with open(tar_path, "wb") as f:
            f.write(bytes(data))
        logger.info("[cc_traj] %s trajectory tar 已落本地 -> %s (%d 字节)", stage, tar_path, len(data))
    except Exception as e:
        logger.warning("[cc_traj] %s 写 trajectory tar 本地失败: %s", stage, str(e)[:160])
        return None
    import tarfile
    extract_dir = tar_path[:-len(".tar.gz")] if tar_path.endswith(".tar.gz") else tar_path + ".dir"
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")
        logger.info("[cc_traj] %s trajectory 已解包 -> %s", stage, extract_dir)
    except Exception as e:
        logger.warning("[cc_traj] %s 解包 trajectory 失败(忽略, tar.gz 仍可用): %s", stage, str(e)[:160])
    return tar_path
