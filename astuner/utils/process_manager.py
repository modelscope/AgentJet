import os
import shlex
import subprocess
import time


def _replace_placeholder_in_config(config_obj, placeholder: str, replacement: str):
    """Recursively replace placeholder in all string values within dict/list structures.

    - Traverses dicts and lists deeply
    - Replaces all occurrences of `placeholder` inside string values
    - Leaves non-string scalars untouched
    """

    def _walk(node):
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, str):
            return node.replace(placeholder, replacement)
        return node

    return _walk(config_obj)


def kill_process_with_keyword(keyword: str, exclude_substrings=None, grace_seconds: float = 1.0):
    """Use bash pipelines to kill processes matching keyword quickly.

    - Filters out processes containing any exclude_substrings
    - Excludes current launcher process
    - Sends TERM once to all PIDs, then KILL once to all PIDs after a short grace period
    - Returns list of PIDs targeted
    """
    if exclude_substrings is None:
        exclude_substrings = ["vscode"]

    self_pid = os.getpid()

    # Build a fast PID collector using pgrep if available; fallback to ps/grep
    # We prefer pgrep -af to filter by full command and then extract PID (column 1)
    exclude_filters = " ".join([f"| grep -v -F {shlex.quote(s)}" for s in exclude_substrings])
    pid_list_cmd = (
        f"(pgrep -af -- {shlex.quote(keyword)} 2>/dev/null || true) "
        f"{exclude_filters} | awk '{{print $1}}' | grep -v -x {self_pid} || true"
    )

    try:
        res = subprocess.run(
            ["bash", "-lc", pid_list_cmd],
            capture_output=True,
            text=True,
            check=False,
        )
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
            res2 = subprocess.run(
                ["bash", "-lc", ps_pid_cmd],
                capture_output=True,
                text=True,
                check=False,
            )
            pids = [pid for pid in res2.stdout.split() if pid.isdigit()]
        except Exception as e:
            print(f"Failed to list PIDs via ps/grep: {e}")
            pids = []

    if not pids:
        return []

    pid_args = " ".join(pids)
    try:
        # Send TERM to all in one call
        subprocess.run(
            ["bash", "-lc", f"kill -TERM -- {pid_args} 2>/dev/null || true"],
            check=False,
        )
        time.sleep(grace_seconds)
        # Escalate with KILL once; ignore failures for already-exited PIDs
        subprocess.run(
            ["bash", "-lc", f"kill -KILL -- {pid_args} 2>/dev/null || true"],
            check=False,
        )
    except Exception as e:
        print(f"Error issuing kill commands: {e}")

    return [int(p) for p in pids]
