#!/bin/bash
# ==============================================================================
# Experiment 3: Single-Model Seer Training
# ==============================================================================
# Trainable (14B-LoRA): seer
# Collaborators (235B): villager, witch, hunter
# Opponents (235B): werewolf
# ==============================================================================

set -e

ray stop && ray start --head
PROJECT_DIR="/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet"

echo "=========================================="
echo "Experiment 3: Single-Model Seer Training"
echo "  Trainable: seer (port 10086)"
echo "  Collaborators: villager, witch, hunter (235B)"
echo "  Opponents: werewolf (235B)"
echo "=========================================="

# --- Swarm Server (seer model) ---
tmux new-session -d -s "SWARM_SERVER_EXP3"
tmux send-keys -t "SWARM_SERVER_EXP3" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_EXP3" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_EXP3" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_EXP3" "ajet-swarm start --swarm-port=10086" Enter
echo "Started SWARM_SERVER_EXP3 on port 10086"

# --- Swarm Client ---
tmux new-session -d -s "SWARM_CLIENT_EXP3"
tmux send-keys -t "SWARM_CLIENT_EXP3" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_CLIENT_EXP3" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT_EXP3" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT_EXP3" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT_EXP3" "python -m tutorial.example_werewolves_swarm.agent_roll_v2 --config single-exp3" Enter
echo "Started SWARM_CLIENT_EXP3"

echo ""
echo "All sessions started. To attach:"
echo "  tmux attach -t SWARM_SERVER_EXP3"
echo "  tmux attach -t SWARM_CLIENT_EXP3"
echo ""
echo "To list all sessions: tmux ls"
echo "To kill all sessions:"
echo "  tmux kill-session -t SWARM_SERVER_EXP3"
echo "  tmux kill-session -t SWARM_CLIENT_EXP3"
