#!/bin/bash
# ==============================================================================
# Experiment 6: Single-Model Hunter Training
# ==============================================================================
# Trainable (14B-LoRA): hunter
# Collaborators (235B): villager, seer, witch
# Opponents (235B): werewolf
# ==============================================================================

set -e

ray stop && ray start --head
PROJECT_DIR="/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet"

echo "=========================================="
echo "Experiment 6: Single-Model Hunter Training"
echo "  Trainable: hunter (port 10086)"
echo "  Collaborators: villager, seer, witch (235B)"
echo "  Opponents: werewolf (235B)"
echo "=========================================="

# --- Swarm Server (hunter model) ---
tmux new-session -d -s "SWARM_SERVER_EXP6"
tmux send-keys -t "SWARM_SERVER_EXP6" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_EXP6" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_EXP6" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_EXP6" "ajet-swarm start --swarm-port=10086" Enter
echo "Started SWARM_SERVER_EXP6 on port 10086"

# --- Swarm Client ---
tmux new-session -d -s "SWARM_CLIENT_EXP6"
tmux send-keys -t "SWARM_CLIENT_EXP6" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_CLIENT_EXP6" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT_EXP6" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT_EXP6" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT_EXP6" "python -m tutorial.example_werewolves_swarm.agent_roll_v2 --config single-exp6" Enter
echo "Started SWARM_CLIENT_EXP6"

echo ""
echo "All sessions started. To attach:"
echo "  tmux attach -t SWARM_SERVER_EXP6"
echo "  tmux attach -t SWARM_CLIENT_EXP6"
echo ""
echo "To list all sessions: tmux ls"
echo "To kill all sessions:"
echo "  tmux kill-session -t SWARM_SERVER_EXP6"
echo "  tmux kill-session -t SWARM_CLIENT_EXP6"
