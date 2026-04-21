#!/bin/bash
# ==============================================================================
# Experiment 5: Single-Model Witch Training
# ==============================================================================
# Trainable (14B-LoRA): witch
# Collaborators (235B): villager, seer, hunter
# Opponents (235B): werewolf
# ==============================================================================

set -e

ray stop && ray start --head
PROJECT_DIR="/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet"

echo "=========================================="
echo "Experiment 5: Single-Model Witch Training"
echo "  Trainable: witch (port 10086)"
echo "  Collaborators: villager, seer, hunter (235B)"
echo "  Opponents: werewolf (235B)"
echo "=========================================="

# --- Swarm Server (witch model) ---
tmux new-session -d -s "SWARM_SERVER_EXP5"
tmux send-keys -t "SWARM_SERVER_EXP5" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_EXP5" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_EXP5" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_EXP5" "ajet-swarm start --swarm-port=10086" Enter
echo "Started SWARM_SERVER_EXP5 on port 10086"

# --- Swarm Client ---
tmux new-session -d -s "SWARM_CLIENT_EXP5"
tmux send-keys -t "SWARM_CLIENT_EXP5" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_CLIENT_EXP5" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT_EXP5" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT_EXP5" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT_EXP5" "python -m tutorial.example_werewolves_swarm.agent_roll_v2 --config single-exp5" Enter
echo "Started SWARM_CLIENT_EXP5"

echo ""
echo "All sessions started. To attach:"
echo "  tmux attach -t SWARM_SERVER_EXP5"
echo "  tmux attach -t SWARM_CLIENT_EXP5"
echo ""
echo "To list all sessions: tmux ls"
echo "To kill all sessions:"
echo "  tmux kill-session -t SWARM_SERVER_EXP5"
echo "  tmux kill-session -t SWARM_CLIENT_EXP5"
