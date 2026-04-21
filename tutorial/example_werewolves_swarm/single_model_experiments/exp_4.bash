#!/bin/bash
# ==============================================================================
# Experiment 4: Single-Model Villager Training
# ==============================================================================
# Trainable (14B-LoRA): villager
# Collaborators (235B): seer, witch, hunter
# Opponents (235B): werewolf
# ==============================================================================

set -e

ray stop && ray start --head
PROJECT_DIR="/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet"

echo "=========================================="
echo "Experiment 4: Single-Model Villager Training"
echo "  Trainable: villager (port 10086)"
echo "  Collaborators: seer, witch, hunter (235B)"
echo "  Opponents: werewolf (235B)"
echo "=========================================="

# --- Swarm Server (villager model) ---
tmux new-session -d -s "SWARM_SERVER_EXP4"
tmux send-keys -t "SWARM_SERVER_EXP4" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_EXP4" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_EXP4" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_EXP4" "ajet-swarm start --swarm-port=10086" Enter
echo "Started SWARM_SERVER_EXP4 on port 10086"

# --- Swarm Client ---
tmux new-session -d -s "SWARM_CLIENT_EXP4"
tmux send-keys -t "SWARM_CLIENT_EXP4" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_CLIENT_EXP4" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT_EXP4" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT_EXP4" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT_EXP4" "python -m tutorial.example_werewolves_swarm.agent_roll_v2 --config single-exp4" Enter
echo "Started SWARM_CLIENT_EXP4"

echo ""
echo "All sessions started. To attach:"
echo "  tmux attach -t SWARM_SERVER_EXP4"
echo "  tmux attach -t SWARM_CLIENT_EXP4"
echo ""
echo "To list all sessions: tmux ls"
echo "To kill all sessions:"
echo "  tmux kill-session -t SWARM_SERVER_EXP4"
echo "  tmux kill-session -t SWARM_CLIENT_EXP4"
