#!/bin/bash
# ==============================================================================
# Experiment 8: Single-Model All Good Guys Training
# ==============================================================================
# Trainable (14B-LoRA): villager, seer, witch, hunter
# Collaborators: none (all controlled by trainable model)
# Opponents (235B): werewolf
# ==============================================================================

set -e

ray stop && ray start --head
PROJECT_DIR="/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet"

echo "=========================================="
echo "Experiment 8: Single-Model All Good Guys Training"
echo "  Trainable: villager, seer, witch, hunter (port 10086)"
echo "  Collaborators: none"
echo "  Opponents: werewolf (235B)"
echo "=========================================="

# --- Swarm Server (all good guys model) ---
tmux new-session -d -s "SWARM_SERVER_EXP8"
tmux send-keys -t "SWARM_SERVER_EXP8" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_EXP8" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_EXP8" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_EXP8" "ajet-swarm start --swarm-port=10086" Enter
echo "Started SWARM_SERVER_EXP8 on port 10086"

# --- Swarm Client ---
tmux new-session -d -s "SWARM_CLIENT_EXP8"
tmux send-keys -t "SWARM_CLIENT_EXP8" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_CLIENT_EXP8" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT_EXP8" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT_EXP8" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT_EXP8" "python -m tutorial.example_werewolves_swarm.agent_roll_v2 --config single-exp8" Enter
echo "Started SWARM_CLIENT_EXP8"

echo ""
echo "All sessions started. To attach:"
echo "  tmux attach -t SWARM_SERVER_EXP8"
echo "  tmux attach -t SWARM_CLIENT_EXP8"
echo ""
echo "To list all sessions: tmux ls"
echo "To kill all sessions:"
echo "  tmux kill-session -t SWARM_SERVER_EXP8"
echo "  tmux kill-session -t SWARM_CLIENT_EXP8"
