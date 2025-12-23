# Task Judger

Task Judger evaluates agent outputs and assigns rewards during training. This page covers built-in judgers for common scenarios and how to create custom judgers for specific evaluation needs.

---

## Overview

A Task Judger evaluates the agent's execution results and returns two values:

- **`raw_reward`** (`float`): Numerical score representing output quality (typically 0.0 to 1.0)
- **`is_success`** (`bool`): Whether the task was successfully completed

These values guide the RL training process, helping agents learn which behaviors produce better outcomes.

---

## Base Interface

All Task Judgers inherit from `BaseJudge` and implement the `compute_reward` method:

```python
from astuner.task_judge.base_judge import BaseJudge
from astuner.workflow import WorkflowOutput, WorkflowTask

class BaseJudge:
    def __init__(self, config):
        self.config = config

    def compute_reward(
        self, 
        workflow_task: WorkflowTask, 
        workflow_output: WorkflowOutput
    ) -> tuple:
        """
        Args:
            workflow_task: Contains the task data, including metadata with reference answers
            workflow_output: Contains the agent's output, including metadata with generated answers
        
        Returns:
            tuple: (raw_reward: float, is_success: bool)
        """
        raise NotImplementedError
```

---

## Built-in Task Judgers

AgentScope Tuner provides four built-in judgers for common evaluation scenarios:

### 1. MathAnswerAsJudge

Evaluates mathematical answers by exact string matching, designed for tasks where answers are formatted in LaTeX `\boxed{}` notation.

**When to use:**
- Math problem solving tasks
- Tasks with deterministic, exact answers
- Answers formatted as `\boxed{result}`

**Configuration:**
```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: astuner.task_judge.math_answer_as_judge->MathAnswerAsJudge
```

**How it works:**
1. Extracts the answer from `\boxed{...}` in the agent's output
2. Compares with the reference answer from `workflow_task.task.metadata["answer"]`
3. Returns `(1.0, True)` for correct answers, `(0.0, False)` otherwise

**Required metadata:**
- `workflow_output.metadata["final_answer"]`: Agent's answer with `\boxed{}` format
- `workflow_task.task.metadata["answer"]`: Reference answer

---

### 2. CountdownAnswerAsJudge

Evaluates mathematical equations with partial credit for proper formatting.

**When to use:**
- Number puzzle tasks (e.g., Countdown game)
- Tasks where partial credit is appropriate
- Need to reward proper formatting even when answer is wrong

**Configuration:**
```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: astuner.task_judge.countdown_answer_as_judge->CountdownAnswerAsJudge
```

**Scoring:**
- `0.0`: Invalid or missing answer
- `0.1`: Properly formatted equation but wrong result
- `1.0`: Correct equation and result

**Required metadata:**
- `workflow_output.metadata["final_answer"]`: Equation string with `\boxed{}` format
- `workflow_output.metadata["target"]`: Target number
- `workflow_output.metadata["nums"]`: Available numbers for the equation

---

### 3. EnvServiceJudge

Delegates evaluation to an external environment service, useful for complex interactive environments.

**When to use:**
- Tasks with external simulators (e.g., AppWorld)
- Complex state-based evaluation
- Interactive environments with built-in evaluators

**Configuration:**
```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: astuner.task_judge.env_service_as_judge->EnvServiceJudge
```

**How it works:**
1. Calls `workflow_task.gym_env.evaluate()` to get a score from the environment
2. Converts the score to a normalized reward:
   - Success (score ≥ 1): `1.0 + score * 0.5`
   - Failure (score < 1): `0.0 + score * 0.5`

---

### 4. AutoGraderJudge (Advanced)

Automatically generates evaluation rubrics from reference samples using LLMs for subjective evaluations.

**When to use:**
- Subjective evaluation tasks (e.g., writing quality, dialogue coherence)
- Tasks without clear-cut correct answers
- When you have labeled training samples but need flexible evaluation criteria

**Configuration:**
```yaml
astuner:
  task_judge:
    judge_type: rubrics_auto_grader
    rubrics_auto_grader:
      # Model for rubric generation and evaluation
      model_name: "qwen-plus"  # or "gpt-4", "claude-3-sonnet"
      api_key: "your-api-key"  # or set DASHSCOPE_API_KEY env var
      
      # Evaluation mode
      grader_mode: "pointwise"  # or "listwise" for ranking multiple outputs
      language: "en"  # or "zh"
      
      # Pointwise mode settings
      min_score: 0
      max_score: 10
      
      # Reference data for rubric generation
      input_data_type: "dataset_file"
      dataset_file:
        training:
          file_path: "path/to/rubrics_train.jsonl"
      
      # Field mappings
      query_field: "main_query"      # field name for the query
      answer_field: "final_answer"   # field in workflow output
      reference_field: "answer"      # field in task metadata
```

**How it works:**
1. **Training Phase**: Generates rubrics from labeled samples using iterative refinement
2. **Evaluation Phase**: Uses the generated rubrics to score new outputs via LLM calls

**Required data format:**

For **pointwise** mode (scoring individual outputs):
```json
{
  "main_query": "Explain quantum entanglement",
  "metadata": {
    "answer": "Quantum entanglement is...",
    "score": 8
  }
}
```

For **listwise** mode (ranking multiple outputs):
```json
{
  "main_query": "Write a poem about spring",
  "metadata": {
    "candidates": [
      {"answer": "Flowers bloom...", "rank": 1},
      {"answer": "Spring is here...", "rank": 2}
    ]
  }
}
```


---

## Creating Custom Task Judgers

For specialized evaluation needs, create your own judger by inheriting `BaseJudge`:

### Step 1: Implement Your Judger

Create a new file (e.g., `tutorial/my_task/my_judge.py`):

```python
import re
from astuner.task_judge.base_judge import BaseJudge
from astuner.workflow import WorkflowOutput, WorkflowTask

class MyCustomJudge(BaseJudge):
    def __init__(self, config):
        super().__init__(config)
        # Initialize any resources (e.g., external APIs, models)
        self.threshold = 0.8
    
    def compute_reward(
        self, 
        workflow_task: WorkflowTask, 
        workflow_output: WorkflowOutput
    ) -> tuple:
        # Extract data from workflow_output
        agent_answer = workflow_output.metadata.get("final_answer", "")
        
        # Extract reference from workflow_task
        reference_answer = workflow_task.task.metadata.get("answer", "")
        
        # Your custom evaluation logic
        similarity = self._compute_similarity(agent_answer, reference_answer)
        
        # Determine success based on threshold
        is_success = similarity >= self.threshold
        
        # Return (reward, success)
        return similarity, is_success
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        # Your custom similarity metric
        # This is just a simple example
        return len(set(text1.split()) & set(text2.split())) / max(
            len(text1.split()), len(text2.split()), 1
        )
```

### Step 2: Configure Your Judger

In your YAML configuration:

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: tutorial.my_task.my_judge->MyCustomJudge
```

### Step 3: Pass Data to the Judger

In your workflow, populate `workflow_output.metadata` with the data your judger needs:

```python
class MyWorkflow(Workflow):
    async def execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        # Your agent logic
        final_answer = await self.agent.reply(msg)
        
        # Return output with metadata for the judger
        return WorkflowOutput(
            reward=None,  # Will be filled by the judger
            metadata={
                "final_answer": final_answer,
                # Add any other fields your judger needs
            }
        )
```

---


## Configuration Summary

### Using Built-in Judgers

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: astuner.task_judge.<module>-><ClassName>
```

### Using Auto Grader

```yaml
astuner:
  task_judge:
    judge_type: rubrics_auto_grader
    rubrics_auto_grader:
      model_name: "qwen-plus"
      grader_mode: "pointwise"
      # ... other config
```

