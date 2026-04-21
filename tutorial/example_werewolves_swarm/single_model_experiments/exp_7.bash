#!/bin/bash
# ==============================================================================
# Experiment 7: Single-Model Seer + Witch + Hunter Training
# ==============================================================================
# Trainable (14B-LoRA): seer, witch, hunter
# Collaborators (235B): villager
# Opponents (235B): werewolf
# ==============================================================================

set -e

ray stop && ray start --head
PROJECT_DIR="/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet"

echo "=========================================="
echo "Experiment 7: Single-Model Seer + Witch + Hunter Training"
echo "  Trainable: seer, witch, hunter (port 10086)"
echo "  Collaborators: villager (235B)"
echo "  Opponents: werewolf (235B)"
echo "=========================================="

# --- Swarm Server (seer+witch+hunter model) ---
tmux new-session -d -s "SWARM_SERVER_EXP7"
tmux send-keys -t "SWARM_SERVER_EXP7" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_EXP7" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_EXP7" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_EXP7" "ajet-swarm start --swarm-port=10086" Enter
echo "Started SWARM_SERVER_EXP7 on port 10086"

# --- Swarm Client ---
tmux new-session -d -s "SWARM_CLIENT_EXP7"
tmux send-keys -t "SWARM_CLIENT_EXP7" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_CLIENT_EXP7" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT_EXP7" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT_EXP7" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT_EXP7" "python -m tutorial.example_werewolves_swarm.agent_roll_v2 --config single-exp7" Enter
echo "Started SWARM_CLIENT_EXP7"

echo ""
echo "All sessions started. To attach:"
echo "  tmux attach -t SWARM_SERVER_EXP7"
echo "  tmux attach -t SWARM_CLIENT_EXP7"
echo ""
echo "To list all sessions: tmux ls"
echo "To kill all sessions:"
echo "  tmux kill-session -t SWARM_SERVER_EXP7"
echo "  tmux kill-session -t SWARM_CLIENT_EXP7"
