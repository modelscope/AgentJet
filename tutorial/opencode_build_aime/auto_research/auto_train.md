# Auto Research Training — tmux Launch Guide

Launch the **AgentJet swarm server** and **`auto_train.py`** side by side in a
single tmux session. The trainer runs with every argument-parser default; only
the three argparse-required flags are supplied explicitly.

```
┌──────────────────────────┬──────────────────────────┐
│  pane 0.0 (left)         │  pane 0.1 (right)        │
│  ajet-swarm start        │  auto_train.py           │
│  (swarm server :10086)   │  (waits for :10086, then │
│                          │   trains w/ defaults)    │
└──────────────────────────┴──────────────────────────┘
```

## One-shot launch command

Copy-paste this block into your shell:

```bash
SESSION=aime
ROOT=/mnt/data_cpfs/qingxu.fu/alpha_auto_research/agentjet_codebase
EXP=aime_auto_default
RESULT_DIR=$ROOT/tutorial/opencode_build_aime/auto_research/results/$EXP

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session  -d -s "$SESSION" -c "$ROOT"

# Left pane (0.0): swarm server
tmux send-keys -t "$SESSION":0.0 \
  "source .venv/bin/activate && ajet-swarm start" C-m

# Right pane (0.1): auto_train.py — wait for the swarm port, then train
tmux split-window -h -t "$SESSION":0 -c "$ROOT"
tmux send-keys -t "$SESSION":0.1 \
  "source .venv/bin/activate && sleep 10 && python -m tutorial.opencode_build_aime.auto_research.auto_train --batch-size 32 --experiment-name $EXP --result-dir $RESULT_DIR" C-m

tmux select-layout -t "$SESSION":0 even-horizontal
tmux attach -t "$SESSION"
```

What it does:

1. `tmux kill-session` — make the launch idempotent (re-runnable).
2. Left pane activates the venv and runs `ajet-swarm start` (default port `10086`).
3. Right pane activates the venv, **blocks until TCP `:10086` is open** (so the
   trainer never races the server), sources `load_research_env.sh` for the
   SwanLab credentials (`logging="swanlab"`), then starts training.
4. `even-horizontal` makes the two panes equal width; `attach` drops you in.

## tmux cheatsheet

| Action | Keys |
|---|---|
| Switch panes | `Ctrl-b` then `←` / `→` |
| Detach (leave both running) | `Ctrl-b` then `d` |
| Re-attach later | `tmux attach -t aime` |
| Kill everything | `tmux kill-session -t aime` |

## auto_train.py parameters (argparse defaults)

The launch supplies only the three argparse `required=True` flags
(`--batch-size`, `--experiment-name`, `--result-dir`). `--batch-size 32` simply
restates its own default value. Everything else below is left at its default.

| Flag | Default | Notes |
|---|---|---|
| `--batch-size` | `32` | required by argparse; value = its default |
| `--experiment-name` | *(required, no default)* | set to `aime_auto_default` |
| `--result-dir` | *(required, no default)* | `.../auto_research/results/aime_auto_default` |
| `--swarm-url` | `http://localhost:10086` | matches `ajet-swarm start` port |
| `--project-name` | `subject14_aime_baseline_group_4` | |
| `--resolved-yaml-path` | `None` | falls back to `<result-dir>/resolved_swarm_config.yaml` |
| `--prepare-only` | `False` | build config & exit when set |
| `--max-response-length-in-one-turn` | `12000` | |
| `--max-prompt-length` | `3000` | |
| `--max-response-length` | `20000` | |
| `--max-model-len` | `23000` | |
| `--total-training-steps` | `120` | hard step cap |
| `--n-gpu` | `8` | GPUs reserved for the swarm server |
| `--max-env-worker` | `128` | parallel env workers |
| `--eval-interval` | `10` | eval every N global steps |
| `--eval-k` | `4` | rollouts per eval task (pass@k) |
| `--grpo-repeat` | `4` | GRPO `num_repeat` per training task |
| `--ppo-epochs` | `1` | |
| `--mini-batch-num` | `1` | |
| `--use-kl-loss` | `True` | `--no-use-kl-loss` to disable |
| `--use-kl-in-reward` | `False` | `--no-use-kl-in-reward` is the default |
| `--kl-penalty-type` | `kl` | one of `kl/abs/mse/low_var_kl/full` |

> All `__init__` defaults were removed from `AIMEAutoResearchTrainer`; the
> argument parser above is now the single source of truth for defaults.

## Optional environment overrides

These are read at runtime and are unset by default:

| Variable | Default | Effect |
|---|---|---|
| `REMOTE_MODEL_PATH` | `/mnt/data_cpfs/xielipeng.xlp/models/Qwen3-8B` | model to train |
| `AJET_SWARM_URL` | `http://localhost:10086` | overrides `--swarm-url` only if the flag is empty |
| `AJET_SWARM_RESTART` | `0` | `1` forces the swarm engine to restart |

Export them in the relevant pane *before* the command if you need non-defaults.

## Monitoring (optional 3rd pane)

```bash
tmux split-window -v -t aime:0.0 -c "$ROOT"
tmux send-keys -t aime:0.2 \
  "source .venv/bin/activate && ajet-swarm overwatch --swarm-url=http://localhost:10086" C-m
```
