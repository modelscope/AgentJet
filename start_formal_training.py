#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
start_formal_training.py — e2b_atbench 正式训练启动 (GRPO, batch=16, repeat=4)

纯 Python 版. 只做两件事:
  1. 起 ray 集群 (head 本地 8851, worker ssh 8852)
  2. 用 tmux_driver 把 3 个长驻进程布进一个 tmux session

布局 (tmux session "e2b_train" — 每进程一个 window):
  window "fwd"    : judge_forwarder        (18005 -> dashscope)
  window "server" : swarm server           (10086)
  window "client" : 正式训练 GRPO client   (16 并行)

窗口存活保证 (用户硬性要求: 中断也不能让 window 消失):
  - 绝不用 `tmux new-window "<cmd>"` 这种弱写法 (cmd 退出即关窗).
  - new-window 只开空 shell 窗, 再 send-keys 把命令送进 shell -> shell 拥有 pane.
  - 命令末行不用 exec: python 在 shell 前台跑, 进程崩了 shell prompt 还在, window 不消失.
  - 额外开 remain-on-exit on 双保险.
  - 两段式: A.先把 3 个 window 全建出来; B.再逐窗投递, 任一窗失败不波及其余.

用法:
  python3 /mnt/data_cpfs/qingxu.fu/agentjet/start_formal_training.py
  tmux attach -t e2b_train              # Ctrl-b n/p 切 fwd/server/client
  tmux kill-session -t e2b_train        # 一键全停

注: tmux_driver 是纯标准库, 本脚本用任意 python3 即可跑; ray/GPU 查询内部显式走
    /tmp/conda_venv/bin/{python,ray}, 不依赖主解释器装了 ray.
"""

from __future__ import annotations

import os
import re
import resource
import socket
import subprocess
import sys
import time

# ── 路径 / 集群常量 ──
SESSION = "e2b_train"
VENV_PY = "/tmp/conda_venv/bin/python"
RAY_BIN = "/tmp/conda_venv/bin/ray"
REPO = "/mnt/data_cpfs/qingxu.fu/agentjet"
FWD_DIR = REPO + "/tutorial/e2b_atbench"
MASTER_IP = "10.29.255.115"
WORKER_IPS = ["10.29.255.116", "10.29.255.112", "10.29.255.114"]  # 8852, 8853, 8854
NOFILE = 102400  # 该集群 RLIMIT_NOFILE hard limit

# 编译缓存 (防 virtiofs ESTALE).
# 注: setup_environment() 已将这些写入 os.environ; 此处 CACHE_ENV 仅供 ray head/worker 启动时显式传递.
CACHE_ENV = {
    "PYTHONNOUSERSITE": "1",
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    "TRITON_CACHE_DIR": "/dev/shm/triton_e2b",
    "VLLM_CACHE_ROOT": "/dev/shm/vllm_cache_e2b",
    "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/torchinductor_e2b",
}
SHM_DIRS = ("/dev/shm/triton_e2b", "/dev/shm/vllm_cache_e2b", "/dev/shm/torchinductor_e2b")

# fork 到 venv python 查 ray 节点/GPU 数 (单行 print, 末尾取一行)
GPU_QUERY = (
    "import ray\n"
    "ray.init(address='auto', ignore_reinit_error=True)\n"
    "alive=[n for n in ray.nodes() if n['Alive']]\n"
    "print('  nodes=%d gpu=%d' % (len(alive),"
    " sum(n['Resources'].get('GPU',0) for n in alive)))\n"
)


# ──────────────────────────────────────────────────────────────────────────────
# 环境变量设置 (原 /tmp/e2b_env_new.sh 内容转为 Python)
# ──────────────────────────────────────────────────────────────────────────────

def setup_environment():
    """设置所有训练所需的环境变量 (原 /tmp/e2b_env_new.sh 的 Python 版本)."""
    env = {
        # Python / 编译缓存
        "PYTHONNOUSERSITE": "1",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "TRITON_CACHE_DIR": "/dev/shm/triton_e2b",
        "VLLM_CACHE_ROOT": "/dev/shm/vllm_cache_e2b",
        "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/torchinductor_e2b",

        # E2B 沙箱 (PAI-EAS)
        "E2B_DOMAIN": "sandbox01.vpc.cn-hongkong.pai-eas.aliyuncs.com",
        "E2B_API_KEY": os.environ.get("E2B_API_KEY", ""),
        "E2B_VALIDATE_API_KEY": "false",
        "SLIME_AGENT_E2B_TEMPLATE": "agentscope-qwenpaw-0604",

        # 网络: 沙箱回连 master; judge 走独立转发进程 -> dashscope
        "ADAPTER_PUBLIC_HOST": "10.29.255.115",
        "JUDGE_FORWARDER_PORT": "18005",
        "JUDGE_MODEL_SERVER": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "JUDGE_DASHSCOPE_KEY": os.environ.get("JUDGE_DASHSCOPE_KEY", ""),
        "E2B_ATBENCH_JUDGE_MODEL": "glm-5.2",

        # claude code 二进制 + python driver (vendored in tutorial/e2b_atbench/atbench_runtime)
        "E2B_ATBENCH_MATERIAL": "/root/slime/examples/coding_agent_rl",
        "CC_CLAUDE_BIN": REPO + "/tutorial/e2b_atbench/atbench_runtime/claudecode_binary/claude",
        "CC_TMUX_BIN": REPO + "/tutorial/e2b_atbench/atbench_runtime/tmux_binary/tmux",
        "CC_TMUX_LIBEVENT": REPO + "/tutorial/e2b_atbench/atbench_runtime/tmux_binary/libevent_core-2.1.so.7",
        "CC_DRIVER_DIR": REPO + "/tutorial/e2b_atbench/atbench_runtime/claudecode_py_driver",

        # 数据: 用 slime 源 (含 Clean_Tasks)
        "E2B_ATBENCH_CLEAN_TASKS": "/root/slime/examples/coding_agent_rl/0730_ATBV3_TRAIN/Clean_Tasks",

        # Swarm + 训练模型
        "AJET_SWARM_URL": "http://localhost:10086",
        "REMOTE_MODEL_PATH": "/dev/shm/Qwen3.6-35B-A3B",
        "FORCE_RESTART_SWARM_ENGINE": "1",
        "E2B_ATBENCH_VLLM_TOOL_PARSER": "qwen3_coder",
        "SWANLAB_API_KEY": os.environ.get("SWANLAB_API_KEY", ""),
        "SWANLAB_API_HOST": "https://cloud-20.agent-matrix.com/api",
        "SWANLAB_WEB_HOST": "https://cloud-20.agent-matrix.com",

        # Ulysses sequence parallel size
        "ULYSSES_SEQUENCE_PARALLEL_SIZE": "8",
        "PPO_MAX_TOKEN_LEN_PER_GPU": "20000",
    }
    # SECURITY: secrets must come from the environment, never be hardcoded/committed.
    _required_secrets = ("E2B_API_KEY", "JUDGE_DASHSCOPE_KEY", "SWANLAB_API_KEY")
    _missing = [k for k in _required_secrets if not env.get(k)]
    if _missing:
        raise SystemExit(
            "FATAL: required secrets not in env: " + ", ".join(_missing)
            + ". Export them in ~/.bashrc (then source it); never commit them."
        )
    os.environ.update(env)

    # 把训练 env 落盘成 shell 文件, 供 tmux pane 启动时 source。
    # 必要性: tmux pane 继承的是 tmux *server* 的环境, 而非本 launcher 进程的 os.environ;
    # 若 server 早于本次启动已存在 (有别的 session), pane 拿不到上面这些 var
    # (实测: client 因 E2B_ATBENCH_CLEAN_TASKS 未设而用了代码默认路径 -> AssertionError)。
    ENV_DUMP_PATH = "/tmp/e2b_train_env.sh"
    with open(ENV_DUMP_PATH, "w") as _f:
        for _k, _v in env.items():
            _f.write(f'export {_k}="{_v}"\n')
    print(f"  [env] 训练环境已写入 {ENV_DUMP_PATH} ({len(env)} vars)")


# ──────────────────────────────────────────────────────────────────────────────
# 小工具
# ──────────────────────────────────────────────────────────────────────────────

def _run(cmd, env=None, cwd=None, preexec_fn=None, timeout=None):
    """跑一条命令 (list 形式, 不经本地 shell), 合并 stderr->stdout."""
    return subprocess.run(
        cmd, env=env, cwd=cwd, preexec_fn=preexec_fn, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _tail(text):
    """取输出最后一个非空行 (对应原 bash 的 `2>&1 | tail -1`)."""
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _raise_nofile():
    """preexec_fn: 子进程 exec 前把 RLIMIT_NOFILE 提到 NOFILE (等价 ulimit -n)."""
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(NOFILE, hard), hard))


def _port_listening(port, host="127.0.0.1", timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 0. ray 集群 (head 本地起, worker ssh; 缓存 env + ulimit 透传给 ray 子进程)
# ──────────────────────────────────────────────────────────────────────────────

def _ray_worker_up(worker_ip):
    r = _run(["ssh", "-o", "ConnectTimeout=10", f"root@{worker_ip}",
              "pgrep -c raylet 2>/dev/null || echo 0"])
    return bool(re.match(r"^[1-9]", (r.stdout or "").strip()))


def start_ray():
    for d in SHM_DIRS:
        os.makedirs(d, exist_ok=True)
    ray_env = dict(os.environ, **CACHE_ENV)

    print("=== [0/2] ray 集群 ===")
    if _port_listening(6379):
        print("  ray head 已在 (6379)")
    else:
        print("  启动 8851 ray head...")
        r = _run([RAY_BIN, "start", "--head", "--port=6379", "--dashboard-host=0.0.0.0", "--min-worker-port=20000", "--max-worker-port=29999"],
                 env=ray_env, preexec_fn=_raise_nofile)
        line = _tail(r.stdout)
        if line:
            print(" ", line)

    exports = " ".join(f"export {k}={v};" for k, v in CACHE_ENV.items())
    mkdirs = "mkdir -p " + " ".join(SHM_DIRS)
    for wip in WORKER_IPS:
        if _ray_worker_up(wip):
            print(f"  ray worker {wip} 已在")
            continue
        print(f"  启动 ray worker {wip}...")
        remote_cmd = (f"{exports} {mkdirs}; ulimit -n {NOFILE}; "
                      f"{RAY_BIN} start --address={MASTER_IP}:6379 --min-worker-port=20000 --max-worker-port=29999")
        r = _run(["ssh", "-o", "ConnectTimeout=10", f"root@{wip}", remote_cmd])
        line = _tail(r.stdout)
        if line:
            print(" ", line)

    time.sleep(3)
    print("  ray 节点/GPU:")
    r = _run([VENV_PY, "-c", GPU_QUERY], cwd=REPO)
    line = _tail(r.stdout)
    if line:
        print(" ", line)


# ──────────────────────────────────────────────────────────────────────────────
# 1/2. tmux 布局 spec
#   - 每个 window = 单 pane (空 shell), 命令按序 send-keys 进去.
#   - 末行不用 exec: python 在 shell 前台跑; 进程崩 -> shell prompt 回来, window 不消失.
#   - 环境变量由 setup_environment() 设置, 子进程自动继承.
# ──────────────────────────────────────────────────────────────────────────────

def build_spec():
    return {
        SESSION: {
            "fwd": [
                "source /tmp/e2b_train_env.sh",
                f"cd {FWD_DIR}",
                "echo '[fwd] judge_forwarder -> dashscope'",
                f"{VENV_PY} judge_forwarder.py",
            ],
            "server": [
                "source /tmp/e2b_train_env.sh",
                f"cd {REPO}",
                "echo '[server] swarm 10086'",
                f"{VENV_PY} -m ajet.swarm_cli start --swarm-port=10086",
            ],
            "client": [
                "sleep 16s",
                "source /tmp/e2b_train_env.sh",
                f"cd {REPO}",
                "echo '[client] GRPO batch=16 repeat=4 16并行'",
                f"{VENV_PY} -m tutorial.e2b_atbench.e2b_atbench_swarm_client",
            ],
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2/2. 启动 tmux 布局 (两段式 + 逐窗隔离; window 永不因中断/崩溃消失)
# ──────────────────────────────────────────────────────────────────────────────

def launch_tmux():
    """建 3 个空 shell window, 再逐窗 send-keys 投递命令.

    关键设计:
      - ensure_window 用 `new-window -n <name> -c <cwd>` (不带命令), 得到空 shell 窗;
        绝不用 `new-window "<cmd>"` (那是弱写法: cmd 退出即关窗).
      - verify_send=False: echo 回看校验对 exec 启动的长驻进程易误判 (进程输出冲掉
        命令行 -> 判"没送到" -> 重试 -> 抛 TmuxError -> 整个 run 中断只剩 fwd).
        send-keys -l 本就可靠, 关掉这个误判源; 交付由末尾 show_status 的 pane cmd 确认.
      - 两段式: A.先把 3 个 window 全 ensure 出来 (中断也至少留 3 个活 shell 窗);
                B.再逐窗 send-keys, 任一窗 try/except 不波及其余.
      - remain_on_exit=True: 即便某 pane 的 shell 真的退了, 窗口也保留 (双保险).
    """
    from tmux_driver import TmuxDriver
    drv = TmuxDriver(default_cwd=REPO, verify_send=False, remain_on_exit=True)
    windows = build_spec()[SESSION]

    # 阶段 A: 先建齐 3 个空 shell window (fwd 复用 window0, 其余 new-window)
    drv.ensure_session(SESSION)
    for w in windows:
        drv.ensure_window(SESSION, w)
    print(f"  [layout] windows={drv.list_windows(SESSION)} (3 个空 shell 窗已就绪)")

    # 阶段 B: 逐窗投递 (互不影响)
    for w, cmds in windows.items():
        target = f"{SESSION}:{w}.0"
        try:
            drv.send_lines(target, cmds)
            print(f"  [OK]   {target}  <- {len(cmds)} cmds")
        except Exception as e:
            print(f"  [FAIL] {target}  <- {e}")
    return drv


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.chdir(REPO)
    sys.path.insert(0, REPO)

    # 设置所有环境变量 (原 /tmp/e2b_env_new.sh 内容)
    setup_environment()

    start_ray()

    # 1. 清旧 tmux session (干净重启; 除此处外全脚本绝不 kill 任何 window)
    _run(["tmux", "kill-session", "-t", SESSION])
    time.sleep(1)

    # 2. 建布局 + 投递 (两段式, window 永不因中断消失)
    print(f"=== [1/2] tmux_driver 布局 (session={SESSION}) ===")
    drv = launch_tmux()
    drv.show_status(SESSION)

    # 3. 汇报
    print()
    print("=== tmux 布局已启动 (via tmux_driver) ===")
    print(f"  session : {SESSION}")
    print("  window 'fwd'    : judge_forwarder (18005)")
    print("  window 'server' : swarm (10086)")
    print("  window 'client' : GRPO 训练 (batch=16 repeat=4)")
    print()
    print(f"  查看:    tmux attach -t {SESSION}     (脱离: Ctrl-b d)")
    print("  切窗:    Ctrl-b n / Ctrl-b p         (下一个/上一个 window)")
    print(f"  全部停:  tmux kill-session -t {SESSION}")


if __name__ == "__main__":
    main()
