#!/bin/bash
# ==============================================================================
# Experiment 2: Four-Model Training (One Model Per Role)
# ==============================================================================
# M1 (14B-LoRA): villager
# M2 (14B-LoRA): seer
# M3 (14B-LoRA): witch
# M4 (14B-LoRA): hunter
# Opponents (235B): werewolf
# ==============================================================================

set -e

PROJECT_DIR="/mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet"

echo "=========================================="
echo "Experiment 2: Four-Model (One Per Role)"
echo "  M1: villager (port 10086)"
echo "  M2: seer (port 10087)"
echo "  M3: witch (port 10088)"
echo "  M4: hunter (port 10089)"
echo "  Opponents: werewolf (235B)"
echo "=========================================="

# --- Swarm Server 1 (M1: villager) ---
tmux new-session -d -s "SWARM_SERVER_M1"
tmux send-keys -t "SWARM_SERVER_M1" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_M1" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_M1" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_M1" "ajet-swarm start --swarm-port=10086" Enter
echo "Started SWARM_SERVER_M1 on port 10086"

# --- Swarm Server 2 (M2: seer) ---
tmux new-session -d -s "SWARM_SERVER_M2"
tmux send-keys -t "SWARM_SERVER_M2" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_M2" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_M2" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_M2" "ajet-swarm start --swarm-port=10087" Enter
echo "Started SWARM_SERVER_M2 on port 10087"

# --- Swarm Server 3 (M3: witch) ---
tmux new-session -d -s "SWARM_SERVER_M3"
tmux send-keys -t "SWARM_SERVER_M3" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_M3" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_M3" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_M3" "ajet-swarm start --swarm-port=10088" Enter
echo "Started SWARM_SERVER_M3 on port 10088"

# --- Swarm Server 4 (M4: hunter) ---
tmux new-session -d -s "SWARM_SERVER_M4"
tmux send-keys -t "SWARM_SERVER_M4" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_SERVER_M4" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER_M4" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER_M4" "ajet-swarm start --swarm-port=10089" Enter
echo "Started SWARM_SERVER_M4 on port 10089"

# --- Swarm Client ---
tmux new-session -d -s "SWARM_CLIENT_EXP2"
tmux send-keys -t "SWARM_CLIENT_EXP2" "cd ${PROJECT_DIR}" Enter
tmux send-keys -t "SWARM_CLIENT_EXP2" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT_EXP2" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT_EXP2" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT_EXP2" "python -m tutorial.example_werewolves_swarm.agent_roll_v2 --config exp2" Enter
echo "Started SWARM_CLIENT_EXP2"

echo ""
echo "All sessions started. To attach:"
echo "  tmux attach -t SWARM_SERVER_M1"
echo "  tmux attach -t SWARM_SERVER_M2"
echo "  tmux attach -t SWARM_SERVER_M3"
echo "  tmux attach -t SWARM_SERVER_M4"
echo "  tmux attach -t SWARM_CLIENT_EXP2"
echo ""
echo "To list all sessions: tmux ls"
echo "To kill all sessions:"
echo "  tmux kill-session -t SWARM_SERVER_M1"
echo "  tmux kill-session -t SWARM_SERVER_M2"
echo "  tmux kill-session -t SWARM_SERVER_M3"
echo "  tmux kill-session -t SWARM_SERVER_M4"
echo "  tmux kill-session -t SWARM_CLIENT_EXP2"
