# example_cocktail_rl_v2

Cocktail RL on AppWorld + AIME with configurable per-client batch ratios and an optional dynamic schedule.



```bash
# install appworld
rm -rf /tmp/pack_all_in_one & wget https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/astuner_archive/appworld_pack_v3.tar.gz  &&   tar   -xzf   ./appworld_pack_v3.tar.gz  -C /tmp

cd /mnt/data_cpfs/qingxu.fu/alpha_auto_research/agentjet_codebase

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

## Custom result dir + ratio 0.25 (client_0 = appworld)

`COCKTAIL_RESULT_DIR` overrides `result_dir` (default `./cocktail_results_v2`); `COCKTAIL_SCHEDULE_START` overrides `schedule_start` (client_0's ratio; under `constant` schedule it is the ratio at every step). Both env vars MUST be set to the same value in both client panes. With `total_batch_size=64`, ratio 0.25 → client_0 (appworld) = 16, client_1 (aime) = 48.

```bash

# install appworld
rm -rf /tmp/pack_all_in_one & wget https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/astuner_archive/appworld_pack_v3.tar.gz  &&   tar   -xzf   ./appworld_pack_v3.tar.gz  -C /tmp

cd /mnt/data_cpfs/qingxu.fu/alpha_auto_research/agentjet_codebase

rm -rf cocktail_results_v2_r025
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
tmux send-keys -t ajet_swarm:main.2 "export COCKTAIL_RATIO_SCHEDULE=constant && export COCKTAIL_SCHEDULE_START=0.25 && export COCKTAIL_RESULT_DIR=./cocktail_results_v2_r025 && source .venv/bin/activate && python -m tutorial.example_cocktail_rl_v2.train_appworld_as_swarm_client_0" Enter

# 第四个 pane：client 1 (aime)
tmux split-window -t ajet_swarm:main
tmux send-keys -t ajet_swarm:main.3 "export COCKTAIL_RATIO_SCHEDULE=constant && export COCKTAIL_SCHEDULE_START=0.25 && export COCKTAIL_RESULT_DIR=./cocktail_results_v2_r025 && source .venv/bin/activate && python -m tutorial.example_cocktail_rl_v2.train_aime_as_swarm_client_1" Enter

# 把四个 pane 平铺成 2x2 网格
tmux select-layout -t ajet_swarm:main tiled

# 进入 session 查看
tmux attach -t ajet_swarm

```

## Custom result dir + ratio 0.75 (client_0 = appworld)

Same setup as above, ratio flipped. With `total_batch_size=64`, ratio 0.75 → client_0 (appworld) = 48, client_1 (aime) = 16.

```bash

rm -rf /tmp/pack_all_in_one & wget https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/astuner_archive/appworld_pack_v3.tar.gz  &&   tar   -xzf   ./appworld_pack_v3.tar.gz  -C /tmp

cd /mnt/data_cpfs/qingxu.fu/alpha_auto_research/agentjet_codebase

rm -rf cocktail_results_v2_r075
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
tmux send-keys -t ajet_swarm:main.2 "export COCKTAIL_RATIO_SCHEDULE=constant && export COCKTAIL_SCHEDULE_START=0.75 && export COCKTAIL_RESULT_DIR=./cocktail_results_v2_r075 && source .venv/bin/activate && python -m tutorial.example_cocktail_rl_v2.train_appworld_as_swarm_client_0" Enter

# 第四个 pane：client 1 (aime)
tmux split-window -t ajet_swarm:main
tmux send-keys -t ajet_swarm:main.3 "export COCKTAIL_RATIO_SCHEDULE=constant && export COCKTAIL_SCHEDULE_START=0.75 && export COCKTAIL_RESULT_DIR=./cocktail_results_v2_r075 && source .venv/bin/activate && python -m tutorial.example_cocktail_rl_v2.train_aime_as_swarm_client_1" Enter

# 把四个 pane 平铺成 2x2 网格
tmux select-layout -t ajet_swarm:main tiled

# 进入 session 查看
tmux attach -t ajet_swarm

```
