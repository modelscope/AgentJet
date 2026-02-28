<!-- # ------- AI GENERATED --------
# ------- [Read tutorial/opencode_build_countdown_agent.prompt.md] -------- -->

# CountDown Number Puzzle Solver Agent

This is a CountDown number puzzle-solving intelligent agent trained using the AgentJet framework.

## Task Description

CountDown is a classic number puzzle game:

- Given a target number
- Given a set of source numbers (usually 6 numbers)
- Using basic arithmetic operations (+, -, ×, ÷)
- Each source number can only be used once
- The goal is to find a mathematical expression whose result equals the target number

### Example

```
Target: 100
Available numbers: [25, 3, 6, 2, 5, 1]
Possible solution: 25 * (6 - 2) = 100
```

## Project Structure

```
tutorial/opencode_build_countdown_agent/
├── agent_run.py                    # Intelligent agent execution and reward calculation
├── agent_roll.py                   # Training loop script
├── generate_countdown_dataset.py  # Dataset generation script
├── countdown_dataset/              # Generated dataset directory
│   ├── train.jsonl                # Training data
│   └── examples.json              # Example data
└── readme.md                       # This document
```

## File Description

### agent_run.py

Contains three core functions:

1. **`parse_and_verify_solution()`**: Parse and verify the agent's solution
   - Extract mathematical expressions
   - Verify that only available source numbers are used, and each number is used only once
   - Calculate the value of the expression
   - Check if the result matches the target number

2. **`_compute_reward()`**: Compute the reward value
   - Perfect match: 1.0
   - Very close result (error < 10): 0.5
   - Fairly close result (error < 50): 0.2
   - Otherwise: 0.0

3. **`run_agent_and_compute_reward()`**: Main execution function
   - Receive tasks and API credentials
   - Call the model to generate solutions
   - Compute and return the reward

### agent_roll.py

Training loop script containing:

- **Configuration parameters**:
  - `LOCAL_GRPO_N`: GRPO group size (number of rollouts per task)
  - `LOCAL_NUM_EPOCH`: Number of training epochs
  - `REMOTE_BATCH_SIZE`: Batch size
  - `REMOTE_ALLOCATE_GPU_PER_NODE`: Number of GPUs used
  - `REMOTE_TRAIN_MODEL`: Training model path

- **Training process**:
  1. Load the dataset
  2. Connect to the Swarm server
  3. Configure training parameters
  4. Execute training loop

### generate_countdown_dataset.py

Dataset generation script containing:

- **`generate_simple_countdown_problem()`**: Generate a single problem
  - Randomly select source numbers (4 small numbers + 2 large numbers)
  - Construct a solvable problem
  - Provide a reference solution

- **`generate_dataset()`**: Generate the full dataset
  - Default generates 50 training samples

- **`save_dataset()`**: Save the dataset in JSONL format

## Usage Guide

### Step 1: Generate Dataset

First, generate the training data:

```bash
cd /root/agentjet
python tutorial/opencode_build_countdown_agent/generate_countdown_dataset.py
```

This will generate training data in the `tutorial/opencode_build_countdown_agent/countdown_dataset/` directory.

### Step 2: Start Swarm Server

Start the Swarm server on a GPU server:

```bash
ajet-swarm start
```

Or run it in the background and start the monitoring dashboard simultaneously:

```bash
(ajet-swarm start &> ajet-swarm-server.log) & (ajet-swarm overwatch)
```

### Step 3: Configure Training Parameters

Edit the configuration parameters in `agent_roll.py`:

```python
# Local configuration
LOCAL_GRPO_N = 4              # GRPO group size
LOCAL_NUM_EPOCH = 100         # Number of training epochs
LOCAL_DATASET_PATH = "./tutorial/opencode_build_countdown_agent/countdown_dataset/train.jsonl"
REMOTE_SWARM_URL = "http://localhost:10086"  # Swarm server URL

# Remote configuration (effective on the Swarm server)
REMOTE_BATCH_SIZE = 16        # Batch size
REMOTE_ALLOCATE_GPU_PER_NODE = 8  # Number of GPUs
REMOTE_TRAIN_MODEL = '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2.5-7B-Instruct'
```

### Step 4: Start Training

Run the training script:

```bash
python -m tutorial.opencode_build_countdown_agent.agent_roll
```

## Training Configuration Instructions

### Current Configuration

- **Model**: Qwen2.5-7B-Instruct
- **Number of GPUs**: 8
- **Batch size**: 16
- **GRPO group size**: 4 (rollout 4 times per task)
- **Algorithm**: GRPO (Group Relative Policy Optimization)

### Adjustment Recommendations

1. **Increase dataset size**: If more training data is needed, modify `generate_countdown_dataset.py`:
   ```python
   dataset = generate_dataset(num_samples=500)  # Default is 50
   ```

2. **Adjust batch size**: Depending on GPU memory, adjust `REMOTE_BATCH_SIZE`

3. **Adjust GRPO group size**: `LOCAL_GRPO_N` affects exploration degree
   - Larger values: More varied solutions, but slower training
   - Smaller values: Faster training, but potentially insufficient exploration

4. **Modify reward function**: Adjust the reward strategy in `_compute_reward()` in `agent_run.py`

## Monitor Training

### Using Swarm Monitoring Dashboard

```bash
ajet-swarm overwatch --swarm-url=http://localhost:10086
```

The monitoring dashboard displays:

- Current training status (OFFLINE/BOOTING/ROLLING/ROLLING_POST/WEIGHT_SYNCING)
- Number of completed episodes
- Current batch progress
- Average reward
- GPU usage

### View Logs

During training, output includes:

- Target and available numbers for each problem
- Solutions generated by the agent
- Validation results and rewards
- Batch statistics

## Debugging Tips

### 1. Test a Single Problem

You can run `agent_run.py` separately to test a single problem:

```python
from ajet.schema.task import Task
from tutorial.opencode_build_countdown_agent.agent_run import run_agent_and_compute_reward

task = Task(
    main_query="Target: 100, Numbers: [25, 3, 6, 2, 5, 1]",
    metadata={
        "target": 100,
        "sources": [25, 3, 6, 2, 5, 1],
        "description": "Find a way to reach 100 using these numbers: [25, 3, 6, 2, 5, 1]."
    }
)

# Need to provide base_url and api_key
# workflow_output = run_agent_and_compute_reward(task, base_url, api_key)
```

### 2. Check Dataset

View generated example data:

```bash
cat tutorial/opencode_build_countdown_agent/countdown_dataset/examples.json
```

### 3. Mid-process Debugging

During training, you can:

- Change `end_episode` to `abort_episode` to discard certain episodes
- Remove `stop_engine` to continue training after interruption
- Modify the code and rerun (training progress will not be lost)

### 4. Distributed Debugging

You can run the same `agent_roll.py` script on multiple machines to achieve distributed training:

- All clients connect to the same Swarm server
- Clients can start or stop at any time
- Training progress is managed by the server and will not be lost

## Expansion Suggestions

### 1. Improve Reward Function

- Consider solution complexity (prefer simpler solutions)
- Consider number of calculation steps (prefer fewer steps)
- Use other models as judges (LLM as Judge)

### 2. Enhance Prompt Engineering

- Add Chain-of-Thought prompts
- Provide more examples
- Require the model to explain reasoning process

### 3. Multi-Agent Collaboration

- Agent 1: Quickly generate candidate solutions
- Agent 2: Verify and optimize solutions
- Agent 3: Explore alternative solutions

### 4. Data Augmentation

- Generate more diverse problems
- Include different difficulty levels
- Add unsolvable problems (let the model learn to identify them)

## Common Issues

### Q: What if training does not converge?

A: Try:

- Reduce the learning rate
- Increase training data
- Adjust the reward function to make it smoother
- Increase the GRPO group size to get better gradient estimates

### Q: How to change the model?

A: Modify the `REMOTE_TRAIN_MODEL` path in `agent_roll.py` to point to another compatible model.

### Q: Can I train on CPU?

A: Theoretically yes, but not recommended. AgentJet is designed for GPU-accelerated training. For CPU debugging, use a small model and limited data.

### Q: How to save a trained model?

A: The Swarm server automatically saves checkpoints. Check the experiment directory to get the saved model weights.

## References

- AgentJet official documentation: https://opencode.ai/docs
- GRPO algorithm paper: [Relevant paper link]
- CountDown game rules: [Game introduction]

## License

Please refer to the LICENSE file in the project root directory.