# Multi-Model Academic Translation Training

This directory implements a **3-agent academic translation workflow** that uses **two different models with independent reward functions** during training:

- **Agent 1 (Rough Translation)**: 7B model → rewarded by **final translation quality**
- **Agent 2 (Detect Proper Nouns)**: 14B model → rewarded by **proper noun detection quality**
- **Agent 3 (Final Translation)**: 7B model → rewarded by **final translation quality**

## Key Innovation: Independent Reward Functions

Unlike the original implementation where all agents share a single reward (final translation quality), this multi-model version uses **separate reward functions** tailored to each model's specific task:

### 7B Model Reward (trans_reward.py)
Evaluates the **final translation quality** considering:

- First-person pronoun usage
- Abbreviation translation
- Word order and sentence structure
- Subject clarity
- Overall translation accuracy

### 14B Model Reward (trans_reward_14B.py)
Evaluates the **proper noun detection quality** considering:

- Completeness: Did it detect all critical errors?
- Accuracy: Are the corrections appropriate?
- False positives: Did it flag correct translations as errors?
- JSON format validity

This allows each model to be trained specifically on its role in the pipeline.

## Architecture Overview

```
Task → Agent 1 (7B) → Agent 2 (14B) → Agent 3 (7B) → Final Translation
         ↓                ↓                  ↓
    Translation      Detection         Translation
      Quality         Quality            Quality
       (7B)           (14B)               (7B)
```

## Files

### `trans.py`
Main workflow execution file. Key features:

- `execute_agent()` accepts TWO `OpenaiBaseUrlAndApiKey` objects (one per model)
- Returns TWO `WorkflowOutput` objects with different rewards:
  - `workflow_output_7b`: Reward based on final translation quality
  - `workflow_output_14b`: Reward based on proper noun detection quality
- Uses `TranslationQualityGrader` for 7B model
- Uses `ProperNounDetectionGrader` for 14B model

### `trans_roll.py`
Training orchestration file. Implements parallel model training:

- Creates two `SwarmClient` instances:
  - `swarm_worker_7b`: Manages 7B model training on port 10086
  - `swarm_worker_14b`: Manages 14B model training on port 10087
- `play_with_two_models()`:
  - Begins episodes on both swarms
  - Executes workflow with both models
  - Reports different rewards to each swarm

### `trans_reward.py`
Original reward function evaluating **final translation quality**.
Used for the 7B model (agents 1 and 3).

### `trans_reward_14B.py` ⭐ NEW
Specialized reward function evaluating **proper noun detection quality**.
Used exclusively for the 14B model (agent 2).

Evaluates:

- **Detected errors**: What the agent successfully caught
- **Missed errors**: Critical errors the agent should have detected
- **False positives**: Incorrect flagging of non-errors
- **JSON validity**: Proper formatting of output

Score scale (0-2):

- 0 = Poor detection (missed critical errors, many false positives, invalid JSON)
- 1 = Acceptable detection (caught some errors but missed important ones)
- 2 = Excellent detection (caught all major errors, minimal false positives)

## Setup Instructions

### 1. Start Two Swarm Servers

```bash
# Terminal 1: Start swarm for 7B model
ajet-swarm start --swarm-port=10086

# Terminal 2: Start swarm for 14B model
ajet-swarm start --swarm-port=10087
```

### 2. Configure Model Paths

Edit `trans_roll.py`:

```python
# 7B model configuration (agents 1 and 3)
REMOTE_7B_TRAIN_MODEL = '/path/to/Qwen2.5-7B-Instruct'

# 14B model configuration (agent 2)
REMOTE_14B_TRAIN_MODEL = '/path/to/Qwen2.5-14B-Instruct'
```

### 3. Configure Dataset Path

```python
LOCAL_DATASET_PATH = "/path/to/your/arxiv_papers/train.parquet"
```

### 4. Configure DASHSCOPE API Key

The grader models use Qwen API:

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 5. Run Training

```bash
# tutorial/example_academic_trans_swarm/train-multi-model/trans_roll.py
python -m tutorial.example_academic_trans_swarm.train_multi_model.trans_roll
```

## How It Works

### Episode Execution Flow

For each training task:

1. **Begin Episodes**:
   - Start episode on 7B swarm → get `api_baseurl_key_7b`
   - Start episode on 14B swarm → get `api_baseurl_key_14b`

2. **Execute Workflow**:
   - Agent 1 (7B): Rough translation
   - Agent 2 (14B): Detect proper noun errors
   - Agent 3 (7B): Produce final translation

3. **Compute Separate Rewards**:
   - **7B Reward**: Evaluate final translation quality (TranslationQualityGrader)
   - **14B Reward**: Evaluate proper noun detection quality (ProperNounDetectionGrader)

4. **End Episodes**:
   - Report 7B reward to 7B swarm
   - Report 14B reward to 14B swarm

5. **Training**:
   - 7B model learns to improve translation quality
   - 14B model learns to improve error detection

## Comparison with Original Implementation

| Aspect | Original | Multi-Model |
|--------|----------|-------------|
| Models | Single model | 7B + 14B |
| Reward | Same for all agents | Different per model |
| 7B Training | Final translation quality | Final translation quality |
| 14B Training | Final translation quality | Detection quality |
| SwarmClients | 1 | 2 |
| Specialization | None | Model-specific tasks |

## Benefits

1. **Task-Specific Training**: Each model learns its specific role
2. **Resource Optimization**: Use smaller 7B for simpler tasks
3. **Better Signals**: Detection model gets direct feedback on detection quality
4. **Scalability**: Can scale components independently
5. **Cost Efficiency**: Don't waste 14B capacity on simple translation

## Example Output

```
7B Model Reward (Translation Quality): 0.85
14B Model Reward (Detection Quality): 0.92
```

The 14B model is specifically rewarded for how well it detects errors, not just whether the final translation is good.

## Troubleshooting

1. **Connection errors**: Ensure both swarm servers are running
2. **Model loading issues**: Verify model paths
3. **DASHSCOPE_API_KEY not set**: Export the environment variable
4. **Import errors**: Run from agentjet root directory

## Configuration Parameters

```python
LOCAL_GRPO_N = 4                      # GRPO group size
REMOTE_7B_BATCH_SIZE = 8              # Batch size for 7B model
REMOTE_14B_BATCH_SIZE = 8             # Batch size for 14B model
REMOTE_7B_ALLOCATE_GPU_PER_NODE = 8   # GPUs for 7B model
REMOTE_14B_ALLOCATE_GPU_PER_NODE = 8  # GPUs for 14B model
```
