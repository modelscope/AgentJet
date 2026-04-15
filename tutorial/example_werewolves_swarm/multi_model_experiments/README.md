# Multi-Model Werewolves Experiments

This directory contains bash scripts for running non-shared parameter multi-agent training experiments on the Werewolves RPG game.

## Experiment Overview

| Exp | Trainable Models | Roles Controlled | Opponent (235B) | Swarm Ports |
|-----|------------------|------------------|-----------------|-------------|
| 1 | M1, M2 (14B-LoRA) | M1: hunter, villager; M2: witch, seer | werewolf | 10086, 10087 |
| 2 | M1-M4 (14B-LoRA) | One role each (vl, sr, wt, ht) | werewolf | 10086-10089 |
| 3 | M1-M3 (14B-LoRA) | Each controls one werewolf | vl, sr, wt, ht | 10086-10088 |
| 4 | M1, M2 (14B-LoRA) | 50/50 split (vl+sr / wt+ht) | werewolf | 10086, 10087 |

## Running Experiments

Each experiment script creates tmux sessions for the swarm servers and client:

```bash
# Run experiment 1 (two-model good guys)
bash tutorial/example_werewolves_swarm/multi_model_experiments/exp_1.bash

# Run experiment 2 (four-model, one per role)
bash tutorial/example_werewolves_swarm/multi_model_experiments/exp_2.bash

# Run experiment 3 (three-model werewolves)
bash tutorial/example_werewolves_swarm/multi_model_experiments/exp_3.bash

# Run experiment 4 (two-model 50/50 split)
bash tutorial/example_werewolves_swarm/multi_model_experiments/exp_4.bash
```

## Tmux Session Management

List all sessions:
```bash
tmux ls
```

Attach to a session:
```bash
tmux attach -t SWARM_SERVER_M1
```

Kill all sessions (example for exp_1):
```bash
tmux kill-session -t SWARM_SERVER_M1
tmux kill-session -t SWARM_SERVER_M2
tmux kill-session -t SWARM_CLIENT_EXP1
```

## Configuration

All experiments use:
- **Model**: Qwen2.5-14B-Instruct with LoRA (rank=32, alpha=32)
- **Opponent**: Qwen3-235B-A22B-Instruct-2507
- **Algorithm**: GRPO
- **LoRA**: Enabled by default

To customize, edit `agent_roll_v2.py` or create new experiment configs.
