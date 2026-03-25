# Spy Game Reinforcement Learning Agent

A trainable multi-agent system for the social deduction game "Who is the Spy" using reinforcement learning.

## Game Overview

"Who is the Spy" is a social deduction game where:
- **N players** participate (typically 6-9)
- Most are **civilians** with the same word
- A few are **spies** with a similar but different word
- Each round, players describe their word without saying it directly
- After descriptions, players vote to eliminate suspects
- **Civilians win** if all spies are eliminated
- **Spies win** if they equal or outnumber civilians

The agent learns to:
1. Generate strategic descriptions that help teammates while avoiding detection
2. Analyze other players' descriptions to identify spies
3. Make optimal voting decisions

## Project Structure

```
tutorial/opencode_build_spy_game/
├── mock_dataset.py          # Generate mock game configurations
├── mock_game_dataset.json   # 200 game scenarios with word pairs
├── game_engine.py           # Core game mechanics and player logic
├── agent_run.py             # Agent executor for agent_roll mode
├── agent_roll.py            # Training script for agent_roll mode
├── agent_run_adv.py         # Agent executor for adversarial mode
├── agent_roll_adv.py        # Training script for adversarial mode
└── readme.md                # This file
```

## Training Modes

### Mode 1: agent_roll (Civilians vs Fixed Opponent)

Train a 7B model as the civilian team against qwen-max (via DashScope API) as spies.

**Hardware:** 4 GPUs  
**Reward:** 1.0 if civilians win, 0.0 if spies win

#### Setup:

1. Ensure DASHSCOPE_API_KEY is set in environment:
   ```bash
   export DASHSCOPE_API_KEY="your_api_key_here"
   ```

2. Start swarm server in one terminal:
   ```bash
   cd /root/agentjet
   source .venv/bin/activate
   ajet-swarm start --swarm-port=10086
   ```

3. Run training in another terminal:
   ```bash
   cd /root/agentjet
   source .venv/bin/activate
   python -m tutorial.opencode_build_spy_game.agent_roll
   ```

### Mode 2: agent_roll_adv (Adversarial Training)

Train two 7B models competitively - one as civilians, one as spies.

**Hardware:** 8 GPUs total (4 per swarm server)  
**Reward:** Team-based (1.0 for winners, 0.0 for losers)

#### Setup:

1. Start swarm server 1 (civilians) in terminal 1:
   ```bash
   cd /root/agentjet
   source .venv/bin/activate
   ajet-swarm start --swarm-port=10086
   ```

2. Start swarm server 2 (spies) in terminal 2:
   ```bash
   cd /root/agentjet
   source .venv/bin/activate
   ajet-swarm start --swarm-port=10087
   ```

3. Run adversarial training in terminal 3:
   ```bash
   cd /root/agentjet
   source .venv/bin/activate
   export AJET_SWARM_URL_1="http://localhost:10086"
   export AJET_SWARM_URL_2="http://localhost:10087"
   python -m tutorial.opencode_build_spy_game.agent_roll_adv
   ```

## Debugging with tmux

For easier debugging, use tmux sessions:

### For agent_roll mode:

```bash
# Terminal 1: Start swarm server
tmux new -s spy-swarm-server
cd /root/agentjet && source .venv/bin/activate
ajet-swarm start --swarm-port=10086

# Detach: Ctrl+B, then D

# Terminal 2: Start training client
tmux new -s spy-swarm-client
cd /root/agentjet && source .venv/bin/activate
python -m tutorial.opencode_build_spy_game.agent_roll

# View sessions
tmux ls

# Attach to server: tmux attach -t spy-swarm-server
# Attach to client: tmux attach -t spy-swarm-client
```

### For agent_roll_adv mode:

```bash
# Terminal 1: Start swarm server 1
tmux new -s spy-swarm-server
cd /root/agentjet && source .venv/bin/activate
ajet-swarm start --swarm-port=10086

# Terminal 2: Start swarm server 2
tmux new -s spy-swarm-server-2
cd /root/agentjet && source .venv/bin/activate
ajet-swarm start --swarm-port=10087

# Terminal 3: Start training client
tmux new -s spy-swarm-client
cd /root/agentjet && source .venv/bin/activate
export AJET_SWARM_URL_1="http://localhost:10086"
export AJET_SWARM_URL_2="http://localhost:10087"
python -m tutorial.opencode_build_spy_game.agent_roll_adv
```

## Configuration

### Key Parameters (in agent_roll.py / agent_roll_adv.py):

- `LOCAL_GRPO_N = 4`: Number of rollouts per task (GRPO group size)
- `LOCAL_NUM_EPOCH = 100`: Number of training epochs
- `REMOTE_BATCH_SIZE = 16`: Batch size for policy updates
- `REMOTE_ALLOCATE_GPU = 4`: Number of GPUs per swarm server
- `REMOTE_TRAIN_MODEL`: Path to base model

### Dataset:

The `mock_game_dataset.json` contains 200 diverse game scenarios with word pairs like:
- apple vs pear
- coffee vs tea
- basketball vs football
- piano vs guitar
- etc.

Regenerate dataset:
```bash
python tutorial/opencode_build_spy_game/mock_dataset.py
```

## Game Mechanics Details

1. **Random Player Assignment**: Each episode randomly assigns player names and roles
2. **Description Phase**: Players generate descriptions using LLM without revealing their word
3. **Voting Phase**: Players vote to eliminate the most suspicious player
4. **Win Conditions**: 
   - Civilians win when all spies eliminated
   - Spies win when they equal/outnumber civilians
   - Draw if max rounds (10) reached

## Reward Structure

### agent_roll mode:
- Civilian team (trainable 7B): 1.0 for win, 0.0 for loss
- Spy team (qwen-max): Not trained

### agent_roll_adv mode:
- Civilian team (7B server 1): 1.0 for win, 0.0 for loss
- Spy team (7B server 2): 1.0 for win, 0.0 for loss
- Both models train competitively

## Expected Training Behavior

The agent should learn to:
1. Generate contextually appropriate descriptions
2. Balance between being informative and protective
3. Recognize inconsistent descriptions from opponents
4. Make strategic voting decisions
5. Adapt strategies based on role (civilian vs spy)

## Troubleshooting

### Import errors:
Make sure you're running from the agentjet root directory with proper Python path.

### Connection errors:
- Check swarm server is running: `ajet-swarm overwatch`
- Verify port availability: `netstat -an | grep 10086`
- Check firewall settings if running on different machines

### DASHSCOPE_API_KEY errors (agent_roll mode):
```bash
export DASHSCOPE_API_KEY="your_key"
echo $DASHSCOPE_API_KEY  # Verify it's set
```

### GPU memory issues:
- Reduce batch size in config
- Reduce number of parallel episodes
- Check GPU availability: `nvidia-smi`

## Monitoring Training

Use the swarm overwatch tool:
```bash
ajet-swarm overwatch --swarm-url=http://localhost:10086
```

This displays:
- Current training step
- Sample pool status
- Policy gradient updates
- Model loading status

## Notes

- Each episode creates random player names from a pool of 50 diverse names
- Game typically completes in 3-10 rounds depending on player strategies
- Training uses GRPO (Group Relative Policy Optimization) algorithm
- Models are trained with temperature=0.7-0.8 for creative descriptions
