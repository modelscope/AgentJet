---
name: monitor-with-tmux
description: Monitor training progress by reading tmux content with exponential backoff intervals (30s, 1min, 2min, 4min, 8min, 16min), analyze logs when anomalies occur, and provide fix suggestions
license: See LICENSE.txt for full terms
---

# Monitor with Tmux

Monitor training progress in tmux, detect anomalies, analyze errors, provide fix suggestions.

## Step Zero

Create a sleep script for tmux monitoring:

1. Create `./tmp/wait_tmux.py`

```python
import argparse
import subprocess
import time

SHELLS = {"bash", "zsh", "sh", "fish", "csh", "tcsh", "ksh", "dash", "ash"}

def smart_sleep(session: str, seconds: float, check_every: float = 2.0) -> bool:
    """
    Alternative to time.sleep(), but returns early when commands finish.

    Returns:
        True  - Normal timeout (command still running)
        False - Early return (command finished or session gone)
    """
    end_time = time.time() + seconds
    while time.time() < end_time:
        try:
            r = subprocess.run(
                ["tmux", "list-panes", "-F", "#{pane_current_command}", "-t", session],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0:
                return False  # session gone
            cmds = [l.strip().lower() for l in r.stdout.splitlines() if l.strip()]
            if not any(c not in SHELLS for c in cmds):
                return False  # command finished, back to shell
        except Exception:
            return False

        time.sleep(min(check_every, end_time - time.time()))

    return True


def main():
    parser = argparse.ArgumentParser(description="Wait for a tmux session with smart early-exit.")
    parser.add_argument("session", help="tmux session name")
    parser.add_argument("seconds", type=float, help="total seconds to wait")
    args = parser.parse_args()

    timed_out = smart_sleep(args.session, args.seconds, 2)
    raise SystemExit(0 if timed_out else 1)


if __name__ == "__main__":
    main()
```

## Start Monitoring

When you need to monitor a tmux window, run:

```bash
python ./tmp/tmux_wait.py my_ajet_session_name 30
```

This means:
1. Monitor tmux session named `my_ajet_session_name`
2. Wait 30 seconds

- Exit code 0: Normal timeout (command still running)
- Exit code 1: Command ended early or session disappeared

## Using SSH

When using SSH, always use a local tmux window to establish the SSH connection.

## When You Want to Delay Before Reading tmux Window Again

You must have error immediate return functionality - do not use `sleep xxx`, instead use `python ./tmux_wait.py my_ajet_session_name xxx`

Don't use: `sleep 60 && tmux capture-pane -t my_ajet_session_name -p | tail -80`

You should use: `python ./tmux_wait.py my_ajet_session_name 30 && tmux capture-pane -t my_ajet_session_name -p | tail -80`

- 60 seconds is too long
- Always use `python ./tmux_wait.py` to wait



## Examples


### Examples Without SSH

0 examples available


### Examples With SSH

1 example available

```agent

--- example user instruction begin ---
login
ssh root@localhost -p 8824
goto
/mnt/data_cpfs/qingxu.fu/agentjet/good-luck-agentjet
run with venv and monitor
source .venv/bin/activate && python -m ajet.launcher --conf tests/bench/benchmark_math/benchmark_math.yaml --autokill
--- example user instruction end ---

--- example agent reaction begin ---
The user wants me to monitor a tmux session while running a training task. Let me break down what they're asking:
1. SSH to remote machine (root@localhost -p 8824)
2. Go to specific directory
3. Activate virtual environment and run Python command
4. Monitor tmux session

# Write ./tmp/tmux_wait.py
....

# Create tmux session for training (note: session name must contain keyword `ajet`)
$ tmux new-session -d -s ajet_session -c "/mnt/data_cpfs/qingxu.fu/agentjet/good-luck-agentjet"
...

# Send SSH command to tmux session
$ tmux send-keys -t ajet_session "ssh root@localhost -p 8824" Enter
...

# Wait and check SSH connection
$ python3 /tmp/tmux_wait.py ajet_session 10 && tmux capture-pane -t ajet_session -p | tail -20
SSH connection successful. Now I need to navigate to the directory and run the training command. Let me send commands to the tmux session.
...

# Navigate to working directory
$ tmux send-keys -t ajet_session "cd /mnt/data_cpfs/qingxu.fu/agentjet/good-luck-agentjet" Enter
...

# Start training with venv and launcher
$ tmux send-keys -t ajet_session "source .venv/bin/activate && python -m ajet.launcher --conf tests/bench/benchmark_math/benchmark_math.yaml --autokill" Enter
...

# Capture initial training output
$ tmux capture-pane -t ajet_session -p | tail -40
...

# Wait 30 seconds and check training progress
$ python3 /tmp/tmux_wait.py ajet_session 30 && tmux capture-pane -t ajet_session -p | tail -50
...
Training is running. Commands are executing correctly. Let me continue monitoring at 1-minute intervals.

# Wait 1 minute and check training progress
$ python3 /tmp/tmux_wait.py ajet_session 60 && tmux capture-pane -t ajet_session -p | tail -50
...

# Wait 2 minutes and check training progress
$ python3 /tmp/tmux_wait.py ajet_session 120 && tmux capture-pane -t ajet_session -p | tail -60
...
Training is progressing well. Checkpoints loaded, vLLM server started, swanlab tracking the run. No errors so far, only warnings. Let me continue monitoring.

# Wait 4 minutes and check training progress
$ python3 /tmp/tmux_wait.py ajet_session 240 && tmux capture-pane -t ajet_session -p | tail -60
...

... multiple rounds of monitoring ...

--- example agent reaction end ---

# Destroy tmux session
tmux kill-session -t ajet_session

```
