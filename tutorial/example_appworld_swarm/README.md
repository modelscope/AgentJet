## AppWorld swarm mode

Swarm-mode rewrite of `tutorial/example_appworld`.
The training engine runs remotely (server side), while task enumeration,
env_service instance lifecycle and reward evaluation all happen locally
in the rollout client.

Files:
- `appworld_swarm.py`  — workflow + lightweight `EnvClient` gym wrapper
- `agent_roll.py`      — rollout driver (calls `begin_episode` / `end_episode`)
- `appworld.yaml`      — swarm-mode training config

Required env vars (with sensible defaults):
- `AJET_SWARM_URL`               — swarm server URL (default `http://localhost:10086`)
- `APPWORLD_ENV_URL`             — appworld env_service URL (default `http://127.0.0.1:8080`)
- `APPWORLD_ENV_TYPE`            — env_type passed to env_service (default `appworld`)
- `APPWORLD_TRAINING_SPLIT`      — train split for `get_env_profile` (default `train`)
- `APPWORLD_VALIDATION_SPLIT`    — eval split for `get_env_profile` (default `dev`)
- `APPWORLD_MAX_STEPS`           — per-episode step cap (default `25`)
- `APPWORLD_EVAL_INTERVAL`       — run eval every N global steps (default `10`)
- `APPWORLD_EVAL_K`              — rollouts per eval task, pass@k (default `1`)
- `APPWORLD_TOTAL_TRAINING_STEPS`— hard cap on global steps (default `200`)
- `APPWORLD_RESULT_DIR`          — where eval logs / `val_results.md` are written (default `./appworld_swarm_results`)
- `APPWORLD_MAX_ENV_WORKER`      — max parallel env workers for both train and eval (default `64`)


## Run swarm

```
tmux new-session -d -s "SWARM_SERVER"
tmux send-keys -t "SWARM_SERVER" "cd /mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet" Enter
tmux send-keys -t "SWARM_SERVER" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER" "ajet-swarm start" Enter
ta "SWARM_SERVER"


tmux new-session -d -s "SWARM_CLIENT"
tmux send-keys -t "SWARM_CLIENT" "cd /mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet" Enter
tmux send-keys -t "SWARM_CLIENT" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT" "python -m tutorial.example_appworld_swarm.agent_roll" Enter
ta "SWARM_CLIENT"
```
