#!/bin/bash
# ==============================================================================
# Experiment 3: Three-Model Werewolf Training
# ==============================================================================
# M1 (14B-LoRA): werewolf 1
# M2 (14B-LoRA): werewolf 2
# M3 (14B-LoRA): werewolf 3
# Opponents (235B): villager, seer, witch, hunter
# ==============================================================================

set -e

PROJECT_DIR="/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet"

echo "=========================================="
echo "Experiment 3: Three-Model Werewolves"
echo "  M1: werewolf 1 (port 10086)"
echo "  M2: werewolf 2 (port 10087)"
echo "  M3: werewolf 3 (port 10088)"
echo "  Opponents: villager, seer, witch, hunter (235B)"
echo "=========================================="

# --- Swarm Server 1 (M1: werewolf 1) ---
tmux new-session -d -s "SWARM_SERVER_M1"
tmux send-keys -t "SWARM_SERVER_M1" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_M1" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_M1" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_M1" "ajet-swarm start --swarm-port=10086" Enter
echo "Started SWARM_SERVER_M1 on port 10086"

# --- Swarm Server 2 (M2: werewolf 2) ---
tmux new-session -d -s "SWARM_SERVER_M2"
tmux send-keys -t "SWARM_SERVER_M2" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_M2" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_M2" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_M2" "ajet-swarm start --swarm-port=10087" Enter
echo "Started SWARM_SERVER_M2 on port 10087"

# --- Swarm Server 3 (M3: werewolf 3) ---
tmux new-session -d -s "SWARM_SERVER_M3"
tmux send-keys -t "SWARM_SERVER_M3" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_M3" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_M3" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_M3" "ajet-swarm start --swarm-port=10088" Enter
echo "Started SWARM_SERVER_M3 on port 10088"

# --- Swarm Client ---
tmux new-session -d -s "SWARM_CLIENT_EXP3"
tmux send-keys -t "SWARM_CLIENT_EXP3" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_CLIENT_EXP3" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT_EXP3" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT_EXP3" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT_EXP3" "python -m tutorial.example_werewolves_swarm.agent_roll_v2 --config exp3" Enter
echo "Started SWARM_CLIENT_EXP3"

echo ""
echo "All sessions started. To attach:"
echo "  tmux attach -t SWARM_SERVER_M1"
echo "  tmux attach -t SWARM_SERVER_M2"
echo "  tmux attach -t SWARM_SERVER_M3"
echo "  tmux attach -t SWARM_CLIENT_EXP3"
echo ""
echo "To list all sessions: tmux ls"
echo "To kill all sessions:"
echo "  tmux kill-session -t SWARM_SERVER_M1"
echo "  tmux kill-session -t SWARM_SERVER_M2"
echo "  tmux kill-session -t SWARM_SERVER_M3"
echo "  tmux kill-session -t SWARM_CLIENT_EXP3"
