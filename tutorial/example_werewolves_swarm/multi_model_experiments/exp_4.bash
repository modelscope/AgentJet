#!/bin/bash
# ==============================================================================
# Experiment 4: Two-Model Random 50/50 Split
# ==============================================================================
# M1 (14B-LoRA): 50% random non-werewolf characters (villager, seer)
# M2 (14B-LoRA): remaining 50% non-werewolf characters (witch, hunter)
# Opponents (235B): werewolf
# ==============================================================================

set -e

PROJECT_DIR="/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet"

echo "=========================================="
echo "Experiment 4: Two-Model Random 50/50 Split"
echo "  M1: villager, seer (port 10086)"
echo "  M2: witch, hunter (port 10087)"
echo "  Opponents: werewolf (235B)"
echo "=========================================="

# --- Swarm Server 1 (M1: villager, seer) ---
tmux new-session -d -s "SWARM_SERVER_M1"
tmux send-keys -t "SWARM_SERVER_M1" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_M1" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_M1" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_M1" "ajet-swarm start --swarm-port=10086" Enter
echo "Started SWARM_SERVER_M1 on port 10086"

# --- Swarm Server 2 (M2: witch, hunter) ---
tmux new-session -d -s "SWARM_SERVER_M2"
tmux send-keys -t "SWARM_SERVER_M2" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_M2" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_M2" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_M2" "ajet-swarm start --swarm-port=10087" Enter
echo "Started SWARM_SERVER_M2 on port 10087"

# --- Swarm Client ---
tmux new-session -d -s "SWARM_CLIENT_EXP4"
tmux send-keys -t "SWARM_CLIENT_EXP4" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_CLIENT_EXP4" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT_EXP4" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT_EXP4" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT_EXP4" "python -m tutorial.example_werewolves_swarm.agent_roll_v2 --config exp4" Enter
echo "Started SWARM_CLIENT_EXP4"

echo ""
echo "All sessions started. To attach:"
echo "  tmux attach -t SWARM_SERVER_M1"
echo "  tmux attach -t SWARM_SERVER_M2"
echo "  tmux attach -t SWARM_CLIENT_EXP4"
echo ""
echo "To list all sessions: tmux ls"
echo "To kill all sessions:"
echo "  tmux kill-session -t SWARM_SERVER_M1"
echo "  tmux kill-session -t SWARM_SERVER_M2"
echo "  tmux kill-session -t SWARM_CLIENT_EXP4"
