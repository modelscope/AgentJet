"""Claude-Code agent RL: E2B 沙盒内跑 solver+judge 全流程的 rollout。

本模块只保留 prompt 模板 + _run_claudecode_in_sandbox 编排逻辑; 沙盒探活、
verdict->reward 解析、清理路径表、文件树打印器等辅助已拆到 utils.cc_sandbox。
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tarfile

from tutorial.e2b_atbench.atbench_runtime.sandbox import E2BSandbox

from tutorial.e2b_atbench.atbench_runtime.utils.cc_sandbox import (
    _CC_CLEANUP_PATHS,
    _CC_TREE_PY,
    _capture_sandbox_state,
    _cc_verdict_to_reward,
    _download_jsonl,
    _download_pane_logs,
    _poll_sandbox_state,
    _wait_sandbox_ready,
    _write_verdict_local,
    _save_trajectory_local,
)

logger = logging.getLogger(__name__)
logging.getLogger("e2b").setLevel(logging.WARNING)


# ── claude code 路径的 prompt 模板 (solver / judge) ──
# {subagent_guidance} 占位只在 forcedmulti 模式注入强制多子代理指令；
# freechoice / naive 均不额外干预 solver 的求解 prompt。
SOLVER_PROMPT_TEMPLATE = """\
你是一个独立求解 agent，version 0805。
题目如下（完整指令）：
---
{instruction_in_instruction_md}
---
你需要：
1. 理解题目（一个强调多智能体协作的任务）。
2. 解决问题，并把结果输出到指定位置（通常是 workspace，具体根据题目确定）。
3. 完成后，在 ./ 写一个 `summary.md` 概述你的产出与结论。如果题目要求中提出在其他位置输出总结，请把 ./summary.md 也复制到题目要求的位置。
{subagent_guidance}"""

# 子代理工作流指令: 要求创建多个子代理并把实质工作委派给它们；主 agent 根据任务复杂度
# 决定角色与分工，在适合时并行执行，最后汇总结果并完成解答。
# 仅当 CC_SOLVER_MODE=forcedmulti 时注入。
SUBAGENT_GUIDANCE = """\
## MANDATORY: Solve this task with a subagent workflow
This task MUST be solved by spawning multiple subagents and delegating the substantive work to them. This is a hard requirement that overrides any inclination to do the work yourself. Do NOT complete the task single-handedly, no matter how simple it looks.

Everything else is your decision — how many subagents to spawn, the role and subtask each one handles, and the workflow you use to coordinate them. Choose based on the task complexity and the independent lines of work you identify. Delegate focused work in parallel where useful, wait for their results, synthesize their findings, and then make or finalize the solution yourself."""


SOLVER_MODES = frozenset(("freechoice", "naive", "forcedmulti"))
# Claude Code v2.1.63 起 Task 改名为 Agent；同时 deny 两个名称兼容新旧版本。
NAIVE_DISALLOWED_TOOLS = ("Agent", "Task")


def _resolve_solver_mode() -> str:
    """Parse the three-way solver mode, retaining the old boolean switch as fallback.

    CC_SOLVER_MODE takes precedence and accepts freechoice/naive/forcedmulti
    case-insensitively. If it is absent, the legacy CC_SOLVER_USE_SUBAGENTS=1
    maps to forcedmulti; every other legacy value maps to freechoice.
    """
    configured = os.environ.get("CC_SOLVER_MODE", "").strip().lower()
    if configured:
        if configured not in SOLVER_MODES:
            supported = ", ".join(sorted(SOLVER_MODES))
            raise ValueError(
                f"invalid CC_SOLVER_MODE={configured!r}; expected one of: {supported}")
        return configured

    legacy = os.environ.get("CC_SOLVER_USE_SUBAGENTS", "").strip().lower()
    return "forcedmulti" if legacy in ("1", "true", "yes", "on") else "freechoice"


JUDGE_PROMPT_TEMPLATE = """\
请根据 ./judge 下的材料，评估 ./judge/environment/assets/local_files 中 summary md 的答案。

你是一个独立的评判 agent。你的工作目录（cwd）组装如下：
- `./judge/tests/` 判定程序与材料（test.sh / verify.py / grader / expected_behavior / grading_criteria / llm_judge_rubric / checklist.jsonl 等）。
- `./judge/eval/` pytest 检查点（A类，若有）。
- `./judge/solution/` 参考解（gold answer）。
- `./judge/task.toml` judge 模型配置。
- `./judge/instruction.md` 改造后的题目。
- `./judge/environment/assets/local_files/` solver 的完整提交（含产出 + summary.md）。

工作流：
1. 读题目（prompt 里的增强任务指令），理解要求。
2. 快速读 `./judge/tests/` 下的判定材料：A类读 verify.py/checklist.jsonl；B类读 expected_behavior/grading_criteria/llm_judge_rubric。
3. 快速读 solver 提交：`./judge/environment/assets/local_files/` 下的产出 + summary.md；对照 `./judge/solution/` 的参考解。
4. 快速运行 `bash ./judge/tests/test.sh` 判定（A类调 verify.py；B类按 rubric 判分）。注意 verify.py 期望的 workspace 路径，把 ./judge/environment/assets/local_files 的产出放到它期望的位置。
5. 判 PASS 或 FAIL。重点考察多智能体协作维度：角色分工是否落实、交接产物是否齐全、证据链是否可溯源、最终结论是否正确。
6. 把判定写到 **绝对路径** `/root/task_dir/verdict.md`（不要写 `./verdict.md` 等相对路径，也不要写到别的目录），结构严格如下：

# Judge Verdict

**Verdict:** PASS

**Reason:** <一段简短说明，引用具体标准与客观指标。>

注意：`**Verdict:**` 行后必须紧跟 `PASS` 或 `FAIL`（大写）。必须用 Write 工具把文件写到`/root/task_dir/verdict.md`，写完即结束，不要再做其它操作。
"""


async def _run_claudecode_in_sandbox(sb: E2BSandbox, task_id: str,
                                     solver_base_url: "str | None" = None,
                                     judge_base_url: "str | None" = None,
                                     solver_model: "str | None" = None,
                                     judge_model: "str | None" = None,
                                     auth_token: "str | None" = None) -> dict:
    """Claude-Code agent rollout (tmux-driven) 在 E2B 沙盒内跑完整 solver+judge。

    task_id 这里是 **任务目录的宿主机路径** (内含 enhance_cwd_edit/)。
    adapter_url/session_id 仅为与 qwenpaw 版签名兼容: claude code 直连模型服务器
    (settings.json 里的 ANTHROPIC_BASE_URL), 不走 adapter。
    返回 {"dataset":[{"reward"}], "verdict", "total_steps"} 供 _reward_from_output 解析。

    7 步: ①拷 claude/tmux 二进制(补 libevent) ②settings.json ③driver/helper/run_stage
    ④solver 任务文件 ⑤跑 solver ⑥拷 judge 文件+solver 答案 ⑦跑 judge -> verdict.md。
    约束: tmux/node/claude 一律"预先准备二进制 + 沙盒 API 复制", 不在沙盒内安装;
    node 沙盒已自带(v25), claude.exe 是原生 ELF 自带运行时, tmux 缺 libevent(从宿主补)。
    """
    from tutorial.e2b_atbench.atbench_runtime.sandbox import exec_and_wait

    solver_mode = _resolve_solver_mode()
    await _wait_sandbox_ready(sb)
    await sb._sb.files.write("/root/_cc_tree.py", _CC_TREE_PY)  # 文件树打印器 (dump_tree 用)

    here = os.path.dirname(os.path.abspath(__file__))
    CLAUDE_BIN = os.environ.get("CC_CLAUDE_BIN", os.path.join(here, "claudecode_binary", "claude"))
    TMUX_BIN = os.environ.get("CC_TMUX_BIN", os.path.join(here, "tmux_binary", "tmux"))
    TMUX_LIB = os.environ.get("CC_TMUX_LIBEVENT", os.path.join(here, "tmux_binary", "libevent_core-2.1.so.7"))
    DRIVER_DIR = os.environ.get("CC_DRIVER_DIR", os.path.join(here, "claudecode_py_driver"))
    SOLVER_BASE_URL = solver_base_url or os.environ.get("CC_SOLVER_BASE_URL") or os.environ.get("CC_ANTHROPIC_BASE_URL", "http://47.76.255.52:29928")
    JUDGE_BASE_URL = judge_base_url or os.environ.get("CC_JUDGE_BASE_URL") or SOLVER_BASE_URL
    AUTH_TOKEN = auth_token or os.environ.get("CC_ANTHROPIC_AUTH_TOKEN", "sk-wefjoewfewhviuwhoewjfoiwehfiuewhvbdjnasjcoqjfdow")
    SOLVER_MODEL = solver_model or os.environ.get("CC_SOLVER_MODEL") or os.environ.get("CC_MODEL", "Qwen3.6-35B-A3B")
    JUDGE_MODEL = judge_model or os.environ.get("CC_JUDGE_MODEL") or os.environ.get("CC_MODEL", "glm-5.2")
    SOLVER_TIMEOUT = int(os.environ.get("CC_SOLVER_TIMEOUT", "1800"))
    JUDGE_TIMEOUT = int(os.environ.get("CC_JUDGE_TIMEOUT", "1800"))
    # 类型收窄: 上述变量经 env 默认值兜底, 运行时恒为 str; assert 供 mypy 通过
    assert SOLVER_BASE_URL and JUDGE_BASE_URL and AUTH_TOKEN and SOLVER_MODEL and JUDGE_MODEL


    task_dir = task_id
    enhance = task_dir

    # 清理 qwenpaw 模板残留 (在 INIT 文件树快照前), 让 /root 只留 claudecode 需要的
    await _sh(sb, "rm -rf " + " ".join(_CC_CLEANUP_PATHS))
    await _dump_tree(sb, "INIT")

    # step 1: claude + tmux + libevent (node 沙盒已自带)
    await _upload_binaries(sb, CLAUDE_BIN, TMUX_BIN, TMUX_LIB)

    # step 2: solver/judge 各一份 settings.json (各自模型 + 各自 base_url) naive 模式仅在 solver settings 中 deny Agent/Task，judge 始终不受影响。
    await _write_settings(sb, SOLVER_BASE_URL, JUDGE_BASE_URL, AUTH_TOKEN,
                          SOLVER_MODEL, JUDGE_MODEL, solver_mode)

    # step 3: driver + helper + run_stage
    await _up_dir(sb, DRIVER_DIR, "/root/cc_driver")

    # step 4: solver 任务文件 (environment + instruction.md) -> /root/task_dir/solver
    instruction = await _stage_solver_files(sb, enhance)

    # step 5: 跑 solver (tmux-driven claude code, cwd=/root/task_dir/solver) forcedmulti 注入强制多子代理 prompt；freechoice/naive 不注入。naive 的单 agent 约束已由 settings deny 工具强制执行。
    subagent_guidance = SUBAGENT_GUIDANCE if solver_mode == "forcedmulti" else ""
    solver_prompt = SOLVER_PROMPT_TEMPLATE.format(instruction_in_instruction_md=instruction, subagent_guidance=subagent_guidance)
    logger.info("[cc_rl] solver mode=%s disallowed_tools=%s", solver_mode, ",".join(NAIVE_DISALLOWED_TOOLS) if solver_mode == "naive" else "none")
    await sb._sb.files.write("/root/task_dir/_solver_prompt.txt", solver_prompt)
    await _run_stage(sb, exec_and_wait, "solver", "/root/task_dir/solver",
                     "/root/cc_settings_solver.json", SOLVER_MODEL, SOLVER_TIMEOUT,
                     "/root/task_dir/_solver_prompt.txt", "cc_solver", "/root/task_dir", tag="cc_solver")

    # step 5.5: solver 跑完 -> 提取 trajectory (jsonl 转写流) 落本地。必须在 step 6
    # 之前: judge 会在同一沙盒开新 session 写自己的 jsonl, 不先取走就混在一起分不开。
    # 全程容错, 失败不拖垮 rollout。
    traj_path = await _save_trajectory_local(sb, stage="solver")
    logger.info("[cc_rl] solver trajectory -> %s", traj_path or "(未落盘)")
    # step 5.6: 从 trajectory 统计智能体用量 (数量/各 agent 结束上下文/耗时/输出
    # token) 落 <trajectory>.agent_stats.json + 一行摘要日志。CC_AGENT_STATS=0 可关。
    _record_agent_stats(traj_path)

    # step 6: judge 文件 + solver 答案 -> judge 树
    await _stage_judge_files(sb, enhance)

    # step 7: 跑 judge (cwd=/root/task_dir) -> verdict.md
    # 关键: judge 可能 (a) 超时/崩 (b) 把 verdict.md 写到非约定路径 (如 /root/verdict.md
    # 或 /root/task_dir/judge/verdict.md) -> fetch /root/task_dir/verdict.md 取不到 -> 空 ->
    # NONE。这里改成最多 3 次: 每次跑完 judge 都用多候选路径取 verdict, 取到含
    # `**Verdict:** PASS|FAIL` 的非空 verdict 就停; 否则重跑 judge (prompt 已要求写死
    # /root/task_dir/verdict.md, 重跑多半能写对; 取到非空但没 PASS/FAIL 也算可用, 不浪费轮次)。
    await sb._sb.files.write("/root/task_dir/_judge_prompt.txt", JUDGE_PROMPT_TEMPLATE)
    verdict = ""
    verdict_attempt = 0
    _VERDICT_PATHS = (
        "/root/task_dir/verdict.md",         # 约定路径 (prompt 要求写的)
        "/root/verdict.md",                   # cwd 漂到 /root 时的常见落点
        "/root/task_dir/judge/verdict.md",   # 模型自作主张写进 judge 子目录
        "/root/task_dir/solver/verdict.md",  # 极少见: 写回 solver 目录
    )
    for verdict_attempt in range(1, 4):  # 最多 3 次
        try:
            await _run_stage(sb, exec_and_wait, "judge", "/root/task_dir",
                             "/root/cc_settings_judge.json", JUDGE_MODEL, JUDGE_TIMEOUT,
                             "/root/task_dir/_judge_prompt.txt", "cc_judge", "/root/task_dir",
                             tag=f"cc_judge_a{verdict_attempt}")
        except Exception as e:
            # judge 本身崩 (E2B RPC 超时等): 记下, 试下一轮; 三轮都崩才放弃 -> verdict 留空
            logger.warning("[cc_rl] judge attempt %d/3 崩: %s", verdict_attempt, str(e)[:160])
            continue
        # 多候选路径取 verdict: 任一路径取到非空就用; 全 404/空 -> 视为本轮没产出, 重跑
        for cand in _VERDICT_PATHS:
            try:
                v = await sb._sb.files.read(cand, format="text")
            except Exception:
                continue
            if v and v.strip():
                verdict = v
                logger.info("[cc_rl] judge attempt %d/3: verdict 取自 %s (%d 字节)", verdict_attempt, cand, len(v))
                break
        # 取到含 **Verdict:** PASS|FAIL 的有效判定 -> 停; 取到非空但格式不全也停 (能用就不浪费)
        if verdict:
            break
        logger.warning("[cc_rl] judge attempt %d/3: 未取到 verdict (多候选路径全空), 重跑",
                       verdict_attempt)
    # judge 完成: verdict.md 在沙盒里, 一旦沙盒回收就丢。这里同步拷一份到宿主机 (CC_VERDICT_FILE, 批跑时每 job 一个 -> verdicts/T##_R#.md), NONE/空 verdict 也照拷, 直接证据留底, 不用再翻 worker 日志的 verdict_head。
    _write_verdict_local(verdict)
    reward = _cc_verdict_to_reward(verdict)
    logger.info("[cc_rl] verdict (attempt %d) reward=%.2f\n%s", verdict_attempt, reward, (verdict or "")[:600])
    return {"dataset": [{"reward": reward}], "total_steps": 2, "verdict": verdict or ""}


# ── _run_claudecode_in_sandbox 的子步骤 (按 7 步拆分, 各自可独立读/测) ──
# 约定: 这些 helper 都接收 sb (E2BSandbox) 及其所需参数, 不持有闭包状态。


async def _sh(sb, cmd: str, timeout: int = 180) -> tuple[int, str]:
    """沙盒内跑一条 shell 命令, 返回 (exit_code, stdout+stderr)。"""
    r = await sb._sb.commands.run(cmd, timeout=timeout)
    return r.exit_code, (r.stdout or "") + (r.stderr or "")


async def _up_file(sb, host_path: str, sand_path: str) -> None:
    """宿主机单文件 -> 沙盒 (走 files.write, 大文件 request_timeout=600)。"""
    with open(host_path, "rb") as fp:
        await sb._sb.files.write(sand_path, fp, request_timeout=600)


async def _up_dir(sb, host_dir: str, sand_dir: str) -> None:
    """宿主机目录 -> 沙盒: 打 tar.gz 上传后解压 (沙盒 API 复制, 不在沙盒内装)。"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(host_dir, arcname=".")
    await sb._sb.files.write("/tmp/_cc_payload.tar.gz", buf.getvalue(), request_timeout=600)
    await _sh(sb, f"mkdir -p {sand_dir} && tar xzf /tmp/_cc_payload.tar.gz -C {sand_dir} && rm -f /tmp/_cc_payload.tar.gz")


async def _dump_tree(sb, stage: str) -> None:
    """solver/judge 启动前, 把沙盒 /root 文件树追加到该阶段的 IO 日志。"""
    llm_io_log = (os.environ.get(f"CC_{stage}_IO_LOG")
                  or os.environ.get("CC_SOLVER_IO_LOG")  # INIT 等无专属日志的阶段 -> solver 日志
                  or os.environ.get("LLM_IO_LOG"))
    if not llm_io_log:
        print("[cc_rl] dump_tree skipped: LLM_IO_LOG not set")
        return
    try:
        r = await sb._sb.commands.run(
            "echo '--- /root/task_dir (deep tree) ---'; "
            "python3 /root/_cc_tree.py /root/task_dir 8 2>/dev/null; "
            "echo '--- /root (top, depth 1) ---'; "
            "python3 /root/_cc_tree.py /root 1 2>/dev/null",
            timeout=30)
        tree = (r.stdout or "")
    except Exception as e:
        tree = f"(tree dump failed: {e})"
    try:
        with open(llm_io_log, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 78
                    + f"\n=== {stage} 启动前: 沙盒 /root 文件树 ===\n"
                    + tree + "\n")
    except Exception:
        pass


def _cc_settings(base_url: str, auth_token: str, model: str,
                 disallowed_tools: tuple[str, ...] = ()) -> dict:
    """Build settings.json, optionally removing selected tools from model context."""
    settings = {
        "env": {
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "ANTHROPIC_BASE_URL": base_url,
            "API_TIMEOUT_MS": "3600000",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        },
        "includeCoAuthoredBy": False,
        "skipDangerousModePermissionPrompt": True,
        "model": model,
    }
    if disallowed_tools:
        # Bare tool-name deny rules remove the tools from Claude's context and
        # still take precedence under --dangerously-skip-permissions.
        settings["permissions"] = {"deny": list(disallowed_tools)}
    return settings


async def _upload_binaries(sb, claude_bin: str, tmux_bin: str, tmux_lib: str) -> None:
    """step 1: 拷 claude/tmux 二进制 + 补 tmux 缺的 libevent, chmod + ldconfig + 探活。"""
    await _up_file(sb, claude_bin, "/usr/local/bin/claude")
    await _up_file(sb, tmux_bin, "/usr/local/bin/tmux")
    await _up_file(sb, tmux_lib, "/usr/local/lib/libevent_core-2.1.so.7")
    _, probe = await _sh(sb,
        "chmod +x /usr/local/bin/claude /usr/local/bin/tmux && ldconfig && "
        "echo claude=$(claude --version 2>&1 | head -1) tmux=$(tmux -V 2>&1)")
    logger.info("[cc_rl] step1 binaries: %s", probe.strip().replace("\n", " | "))


async def _write_settings(sb, solver_url: str, judge_url: str, auth_token: str,
                          solver_model: str, judge_model: str,
                          solver_mode: str = "freechoice") -> None:
    """step 2: solver/judge 各一份 settings.json (各自模型 + 各自代理 -> 各自 IO 日志)。"""
    solver_disallowed = NAIVE_DISALLOWED_TOOLS if solver_mode == "naive" else ()
    await sb._sb.files.write("/root/cc_settings_solver.json",
                             json.dumps(_cc_settings(solver_url, auth_token, solver_model,
                                                     solver_disallowed), indent=2))
    await sb._sb.files.write("/root/cc_settings_judge.json",
                             json.dumps(_cc_settings(judge_url, auth_token, judge_model), indent=2))


async def _stage_solver_files(sb, enhance: str) -> str:
    """step 4: solver 任务文件 (environment + instruction.md) -> /root/task_dir/solver; 返回指令文本。"""
    if os.path.exists(os.path.join(enhance, "environment/assets/local_files")):
        await _up_dir(sb, os.path.join(enhance, "environment/assets/local_files"), "/root/task_dir/solver")
    else:
        assert os.path.exists(os.path.join(enhance, "environment/data/local_files"))
        await _up_dir(sb, os.path.join(enhance, "environment/data/local_files"), "/root/task_dir/solver")

    await _up_file(sb, os.path.join(enhance, "instruction.md"), "/root/task_dir/solver/instruction.md")
    with open(os.path.join(enhance, "instruction.md"), encoding="utf-8") as f:
        return f.read()


# claudecode_run_stage.py 的 argv 形参 (注释保留原 inline 说明, 便于对照 driver):
#   argv[1]=stage argv[2]=cwd argv[3]=settings argv[4]=model argv[5]=timeout(int)
#   argv[6]=prompt_file argv[7]=session_id argv[8]=flag_root(可省, 默认 cwd)
async def _run_stage(sb, exec_and_wait, stage: str, cwd: str, settings: str,
                     model: str, timeout: int, prompt_file: str,
                     session_id: str, flag_root: str, tag: str) -> str:
    """step 5/7: 调 driver 跑一个 stage (solver 或 judge), 返回输出尾部。"""
    cmd = (
        f"python3 /root/cc_driver/claudecode_run_stage.py "
        f"{stage} "                      # stage = sys.argv[1]
        f"{cwd} "                         # cwd = sys.argv[2]
        f"{settings} {model} {timeout} " # settings=sys.argv[3] model=sys.argv[4] timeout=int(sys.argv[5])
        f"{prompt_file} {session_id} {flag_root}"  # prompt_file/session_id/flag_root = argv[6..8]
    )
    await _dump_tree(sb, stage.upper())
    # 后台每 5s 打印沙盒内 claude 的当前状态 (state.json msg + jsonl 末条 + tmux pane 尾行),
    # 让 stall 第一时间暴露: 不再像现在要等 stage 超时才发现"proxy 日志停在 count_tokens"。
    poll_task = asyncio.create_task(_poll_sandbox_state(sb, interval_sec=5.0))
    try:
        _, out = await exec_and_wait(sb, cmd=cmd, time_budget_sec=timeout + 300,
                                     tag=tag, want_output=True)
        logger.info("[cc_rl] %s done. tail:\n%s", tag, (out or "")[-1800:])
        # done 后立刻把完整转写 jsonl 拉回宿主机 (沙箱 kill 后再也拿不到)
        jsonl_dir = os.environ.get("CC_JSONL_DIR", os.path.join("tmp", "cc_state"))
        await _download_jsonl(sb, jsonl_dir, tag=tag)
        # pane 捕获 (带秒级时间戳的 claude TUI 全量输出) 也一并拉回, 用于与 interchange 侧 [reqmon] 对表; 沙箱销毁后不可再取。
        await _download_pane_logs(sb, jsonl_dir, tag=tag)
        return out or ""
    except Exception:
        # stall/超时: 抓回沙盒内诊断 (state.json + jsonl 末尾 + tmux 屏) 供事后定位
        state_dir = os.environ.get("CC_STATE_DIR", os.path.join("tmp", "cc_state"))
        await _capture_sandbox_state(sb, state_dir, tag=tag)
        await _download_pane_logs(sb, state_dir, tag=tag)
        raise
    finally:
        poll_task.cancel()


async def _stage_judge_files(sb, enhance: str) -> None:
    """step 6: 拷 judge 文件 (task.toml/tests/solution/eval/...) + solver 答案到 judge 树。"""
    for sub in ("task.toml", "instruction.md", "tests", "solution", "eval"):
        src = os.path.join(enhance, sub)
        if not os.path.exists(src):
            continue
        if os.path.isdir(src):
            await _up_dir(sb, src, f"/root/task_dir/judge/{sub}")
        else:
            await _up_file(sb, src, f"/root/task_dir/judge/{sub}")

    # solver 全量产出 -> judge 树下 (judge prompt 读 ./judge/environment/assets/local_files)
    if os.path.exists(os.path.join(enhance, "environment/assets/local_files")):
        await _sh(sb,
            "mkdir -p /root/task_dir/judge/environment/assets/local_files &&"
            "cp   -av   /root/task_dir/solver/*   /root/task_dir/judge/environment/assets/local_files/   2>/dev/null; true")
    else:
        assert os.path.exists(os.path.join(enhance, "environment/data/local_files"))
        await _sh(sb,
            "mkdir -p /root/task_dir/judge/environment/data/local_files &&"
            "cp   -av   /root/task_dir/solver/*   /root/task_dir/judge/environment/data/local_files/   2>/dev/null; true")


def _reward_from_output(output: dict) -> float:
    # QWENPAW_DUMMY_REWARD: 绕过 DashScope judge (key 失效时), 用固定 reward 验证 token 记录链路
    dummy = os.environ.get("QWENPAW_DUMMY_REWARD")
    if dummy is not None:
        logger.info("[qwenpaw_rl] DUMMY_REWARD=%s (绕过 judge)", dummy)
        return float(dummy)
    dataset = output.get("dataset", [])
    return float(dataset[0]["reward"]) if dataset else 0.0


def _record_agent_stats(traj_path: str | None) -> dict | None:
    """step 5.6: 统计 solver 用的智能体数量与每个 agent 的上下文/耗时/输出 token。

    数据源是 step 5.5 落盘的 trajectory tar (主会话 <uuid>.jsonl + 每个子代理
    <uuid>/subagents/agent-<hash>.jsonl, assistant 条目自带 usage/timestamp)。
    统计逻辑在 utils.cc_session_stats, 可靠口径: 按 message.id 去重防 token 翻倍
    (一次调用拆多条 jsonl 且 usage 重复)、跳过全 0 usage (网关偶发不回传, 取最后
    一条非零)、meta.toolUseId 精确关联子代理 (fallback 时间戳就近)。

    产物: CC_AGENT_STATS_FILE 指定路径, 否则 <trajectory 去掉 .tar.gz>.agent_stats.json
    (与 tar 同目录好对齐)。全程容错: 失败只 warning; CC_AGENT_STATS=0 关闭。
    """
    if not traj_path:
        return None
    if os.environ.get("CC_AGENT_STATS", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    try:
        try:
            from tutorial.e2b_atbench.atbench_runtime.utils.cc_session_stats import (
                analyze_path, summarize_line)
        except Exception:
            from utils.cc_session_stats import analyze_path, summarize_line
        stats = analyze_path(traj_path)
        out_path = os.environ.get("CC_AGENT_STATS_FILE", "").strip() or (
            traj_path[:-len(".tar.gz")] + ".agent_stats.json"
            if traj_path.endswith(".tar.gz") else traj_path + ".agent_stats.json")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info("[cc_stats] solver agent 用量 -> %s | %s", out_path, summarize_line(stats))
        return stats
    except Exception as e:
        logger.warning("[cc_stats] 统计失败(忽略): %s", str(e)[:160])
        return None
