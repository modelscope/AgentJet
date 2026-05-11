# example_cocktail_rl_v2

Cocktail RL on AppWorld + AIME with configurable per-client batch ratios and an optional dynamic schedule.

```bash

rm -rf cocktail_results_v2
source .venv/bin/activate && ajet --autokill
tmux kill-session -t ajet_swarm

# 创建一个 session，名字叫 ajet_swarm，第一个 pane 跑 appworld
tmux new -d -s ajet_swarm -n main
tmux send-keys -t ajet_swarm:main.0 "bash /tmp/pack_all_in_one/EnvService/env_sandbox/appworld.sh" Enter

# 第二个 pane：server
tmux split-window -t ajet_swarm:main
tmux send-keys -t ajet_swarm:main.1 "source .venv/bin/activate && ajet-swarm start" Enter

# 第三个 pane：client 0 (appworld)
tmux split-window -t ajet_swarm:main
tmux send-keys -t ajet_swarm:main.2 "export COCKTAIL_RATIO_SCHEDULE=constant && source .venv/bin/activate && python -m tutorial.example_cocktail_rl_v2.train_appworld_as_swarm_client_0" Enter

# 第四个 pane：client 1 (aime)
tmux split-window -t ajet_swarm:main
tmux send-keys -t ajet_swarm:main.3 "export COCKTAIL_RATIO_SCHEDULE=constant && source .venv/bin/activate && python -m tutorial.example_cocktail_rl_v2.train_aime_as_swarm_client_1" Enter

# 把四个 pane 平铺成 2x2 网格
tmux select-layout -t ajet_swarm:main tiled

# 进入 session 查看
tmux attach -t ajet_swarm

```

Edit `CocktailV2Config` defaults (cocktail_v2_runner.py) for `total_batch_size`, `schedule_start`/`schedule_end`/`schedule_end_step`. Engine knobs live in `build_cocktail_ajet_job()` (train_appworld_as_swarm_client_0.py). Both clients must agree on these.
