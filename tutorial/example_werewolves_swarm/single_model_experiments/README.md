# Single-Model Werewolves Experiments

This directory contains bash scripts for running single-model training experiments on the Werewolves RPG game, where one model trains specific roles while opponents and collaborators use the 235B model.

## Experiment Overview

| Exp | Trainable | Size | Non-trainable | Size | Initial SR | Final SR |
|-----|-----------|------|---------------|------|------------|----------|
| 3 | seer (sr) | 14B | opponents: werewolf, collaborators: villager, witch, hunter | 235B | - | - |
| 4 | villager (vl) | 14B | opponents: werewolf, collaborators: seer, witch, hunter | 235B | - | - |
| 5 | witch (wt) | 14B | opponents: werewolf, collaborators: villager, seer, hunter | 235B | - | - |
| 6 | hunter (ht) | 14B | opponents: werewolf, collaborators: villager, seer, witch | 235B | - | - |
| 7 | seer, witch, hunter | 14B | opponents: werewolf, collaborators: villager | 235B | 0.2291 | 0.3531 |
| 8 | villager, seer, witch, hunter | 14B | opponents: werewolf | 235B | - | - |

## Game Configuration

- **Players**: 9 players with composition (n_ww, n_vl, n_sr, n_wt, n_ht) = (3, 3, 1, 1, 1)
- **Roles**: 3 werewolves, 3 villagers, 1 seer, 1 witch, 1 hunter
- **Model**: Qwen2.5-14B-Instruct with LoRA (rank=32, alpha=32)
- **Opponent/Collaborators**: Qwen3-235B-A22B-Instruct-2507
- **Algorithm**: GRPO

## Running Experiments

Each experiment script creates tmux sessions for the swarm server and client:

```bash
# Run experiment 3 (seer training)
bash tutorial/example_werewolves_swarm/single_model_experiments/exp_3.bash

# Run experiment 4 (villager training)
bash tutorial/example_werewolves_swarm/single_model_experiments/exp_4.bash

# Run experiment 5 (witch training)
bash tutorial/example_werewolves_swarm/single_model_experiments/exp_5.bash

# Run experiment 6 (hunter training)
bash tutorial/example_werewolves_swarm/single_model_experiments/exp_6.bash

# Run experiment 7 (seer + witch + hunter training)
bash tutorial/example_werewolves_swarm/single_model_experiments/exp_7.bash

# Run experiment 8 (all good guys training)
bash tutorial/example_werewolves_swarm/single_model_experiments/exp_8.bash
```

## Tmux Session Management

List all sessions:
```bash
tmux ls
```

Attach to a session:
```bash
tmux attach -t SWARM_SERVER
tmux attach -t SWARM_CLIENT
```

Kill all sessions (example for exp_3):
```bash
tmux kill-session -t SWARM_SERVER_EXP3
tmux kill-session -t SWARM_CLIENT_EXP3
```

## Configuration

All experiments use:
- **Model**: Qwen2.5-14B-Instruct with LoRA (rank=32, alpha=32)
- **Opponent**: Qwen3-235B-A22B-Instruct-2507
- **Algorithm**: GRPO
- **LoRA**: Enabled by default

To customize, edit `agent_roll_single.py` or create new experiment configs.
