---
name: monitor-training-with-tmux
description: Monitor training progress by reading tmux content at exponential backoff intervals (30s, 1min, 2min, 4min, 8min...), analyze logs when anomalies occur, and provide fix suggestions
license: Complete terms in LICENSE.txt
---

# Training Monitor with Tmux

Monitor training in tmux, detect anomalies, analyze errors, provide fix suggestions.

## Smart Sleep Function

Use `smart_sleep` instead of `time.sleep` - returns early if command ends:

```python
"""替代 sleep，在超时或命令结束时返回"""

import subprocess
import time

SHELLS = {"bash", "zsh", "sh", "fish", "csh", "tcsh", "ksh", "dash", "ash"}


def smart_sleep(session: str, seconds: float, check_every: float = 2.0) -> bool:
    """
    替代 time.sleep()，但在命令结束时提前返回。
    
    Returns:
        True  - 正常超时（命令还在跑）
        False - 提前返回（命令结束了或session没了）
    """
    end_time = time.time() + seconds
    while time.time() < end_time:
        try:
            r = subprocess.run(
                ["tmux", "list-panes", "-F", "#{pane_current_command}", "-t", session],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0:
                return False  # session没了
            cmds = [l.strip().lower() for l in r.stdout.splitlines() if l.strip()]
            if not any(c not in SHELLS for c in cmds):
                return False  # 命令结束了，回到shell
        except Exception:
            return False
        
        time.sleep(min(check_every, end_time - time.time()))
    
    return True
```

## Work Cycle

1. Enter tmux window, run command
2. `smart_sleep(session, interval)` - 30s, 1min, 2min, 4min, 8min, 8min...
3. If `smart_sleep` returns `False` → command ended, analyze log
4. If error → analyze, suggest fix, wait for user
5. If no error → continue monitoring

## Monitoring Intervals

```
Check 1: 30s
Check 2: 1min  
Check 3: 2min
Check 4: 4min
Check 5+: 8min (max)
```

## Capture Tmux Content

```bash
tmux capture-pane -t <session> -p -S -2000
```

## Anomaly Detection

### Critical Errors
- `CUDA out of memory` / `OOM`
- `RuntimeError` / `Exception` / `Traceback`
- `NaN` / `nan` / `inf` in loss
- `Killed` / `SIGKILL`

### Warning Patterns
- Loss increasing over steps
- Gradient explosion
- `WARNING` messages

## Common Fixes

| Error | Fix |
|-------|-----|
| CUDA OOM | Reduce batch_size, enable gradient_checkpointing, use bf16 |
| Loss NaN | Reduce lr, enable gradient clipping, add warmup |
| Process Killed | Reduce workers, check system memory |
| Training Stuck | Check NCCL, restart from checkpoint |

## Gather Context on Error

```bash
nvidia-smi
free -h
df -h
dmesg | tail -50
```
