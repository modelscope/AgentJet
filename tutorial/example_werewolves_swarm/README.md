
## opponent model configuration



## Run swarm


tmux new-session -d -s "SWARM_SERVER"
tmux send-keys -t "SWARM_SERVER" "cd /mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet" Enter
tmux send-keys -t "SWARM_SERVER" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_SERVER" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_SERVER" "ajet-swarm start" Enter
ta "SWARM_SERVER"



tmux new-session -d -s "SWARM_CLIENT"
tmux send-keys -t "SWARM_CLIENT" "cd /mnt/data_cpfs/qingxu.fu/agentjet/hello-agentjet" Enter
tmux send-keys -t "SWARM_CLIENT" "source .venv/bin/activate" Enter
tmux send-keys -t "SWARM_CLIENT" "export SETUPTOOLS_USE_DISTUTILS=local" Enter
tmux send-keys -t "SWARM_CLIENT" "sleep 30s" Enter
tmux send-keys -t "SWARM_CLIENT" "python -m tutorial.example_werewolves_swarm.agent_roll_good_guys" Enter
ta "SWARM_CLIENT"
