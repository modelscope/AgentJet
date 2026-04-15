#!/bin/bash
# ==============================================================================
# Experiment 1: Two-Model Good Guys Training
# ==============================================================================
# M1 (14B-LoRA): hunter, villager
# M2 (14B-LoRA): witch, seer
# Opponents (235B): werewolf
# ==============================================================================

set -e

ray stop && ray start --head
PROJECT_DIR="/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet"

echo "=========================================="
echo "Experiment 1: Two-Model Good Guys"
echo "  M1: hunter, villager (port 10086)"
echo "  M2: witch, seer (port 10087)"
echo "  Opponents: werewolf (235B)"
echo "=========================================="

# --- Swarm Server 1 (M1: hunter, villager) ---
tmux new-session -d -s "SWARM_SERVER_M1"    # warning: do not add command here, otherwise it will be executed immediately and the session will exit
tmux send-keys -t "SWARM_SERVER_M1" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_M1" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_M1" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_M1" "ajet-swarm start --swarm-port=10086" Enter
echo "Started SWARM_SERVER_M1 on port 10086"

# --- Swarm Server 2 (M2: witch, seer) ---
tmux new-session -d -s "SWARM_SERVER_M2"
tmux send-keys -t "SWARM_SERVER_M2" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_M2" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_M2" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_M2" "ajet-swarm start --swarm-port=10087" Enter
echo "Started SWARM_SERVER_M2 on port 10087"

# --- Swarm Client ---
tmux new-session -d -s "SWARM_CLIENT_EXP1"
tmux send-keys -t "SWARM_CLIENT_EXP1" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_CLIENT_EXP1" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT_EXP1" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT_EXP1" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT_EXP1" "python -m tutorial.example_werewolves_swarm.agent_roll_v2 --config exp1" Enter
echo "Started SWARM_CLIENT_EXP1"

echo ""
echo "All sessions started. To attach:"
echo "  tmux attach -t SWARM_SERVER_M1"
echo "  tmux attach -t SWARM_SERVER_M2"
echo "  tmux attach -t SWARM_CLIENT_EXP1"
echo ""
echo "To list all sessions: tmux ls"
echo "To kill all sessions: tmux kill-session -t SWARM_SERVER_M1 && tmux kill-session -t SWARM_SERVER_M2 && tmux kill-session -t SWARM_CLIENT_EXP1"
