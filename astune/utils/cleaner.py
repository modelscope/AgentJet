
import subprocess
import argparse
import shutil
import time
import sys
import os
import shlex

def _fast_kill_by_keyword_bash(keyword: str, exclude_substrings=["vscode"], grace_seconds: float = 1.0):
    """Use bash pipelines to kill processes matching keyword quickly.

    - Filters out processes containing any exclude_substrings
    - Excludes current launcher process
    - Sends TERM once to all PIDs, then KILL once to all PIDs after a short grace period
    - Returns list of PIDs targeted
    """
    self_pid = os.getpid()

    # Build a fast PID collector using pgrep if available; fallback to ps/grep
    # We prefer pgrep -af to filter by full command and then extract PID (column 1)
    exclude_filters = " ".join([f"| grep -v -F {shlex.quote(s)}" for s in exclude_substrings])
    pid_list_cmd = (
        f"(pgrep -af -- {shlex.quote(keyword)} 2>/dev/null || true) "
        f"{exclude_filters} | awk '{{print $1}}' | grep -v -x {self_pid} || true"
    )

    try:
        res = subprocess.run(["bash", "-lc", pid_list_cmd], capture_output=True, text=True, check=False)
        pids = [pid for pid in res.stdout.split() if pid.isdigit()]
    except Exception as e:
        print(f"Failed to list PIDs via bash: {e}")
        pids = []

    # Fallback to ps/grep if pgrep path     produced nothing (e.g., no pgrep installed)
    if not pids:
        ps_pid_cmd = (
            f"ps -eo pid,command -ww | grep -F -- {shlex.quote(keyword)} | grep -v grep "
            f"{exclude_filters} | awk '{{print $1}}' | grep -v -x {self_pid} || true"
        )
        try:
            res2 = subprocess.run(["bash", "-lc", ps_pid_cmd], capture_output=True, text=True, check=False)
            pids = [pid for pid in res2.stdout.split() if pid.isdigit()]
        except Exception as e:
            print(f"Failed to list PIDs via ps/grep: {e}")
            pids = []

    if not pids:
        return []

    pid_args = " ".join(pids)
    try:
        # Send TERM to all in one call
        subprocess.run(["bash", "-lc", f"kill -TERM -- {pid_args} 2>/dev/null || true"], check=False)
        time.sleep(grace_seconds)
        # Escalate with KILL once; ignore failures for already-exited PIDs
        subprocess.run(["bash", "-lc", f"kill -KILL -- {pid_args} 2>/dev/null || true"], check=False)
    except Exception as e:
        print(f"Error issuing kill commands: {e}")

    return [int(p) for p in pids]
