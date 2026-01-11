# Task Judger

!!! warning ""
    Task judger will be **disabled** automatically when the user-defined workflow returned an effective `WorkflowOutput.reward` and `WorkflowOutput.reward != None`


Task Judger evaluates agent outputs and assigns rewards during training. This page covers built-in judgers for common scenarios and how to create custom judgers for specific evaluation needs.


## Overview

A Task Judger evaluates the agent's execution results and returns two values:

| Return Value | Type | Description |
|--------------|------|-------------|
| `raw_reward` | `float` | Numerical score representing output quality (often 0.0 to 1.0) |
| `is_success` | `bool` | Whether the task was successfully completed |

These values guide the RL training process, helping agents learn which behaviors produce better outcomes.


## Base Interface

All Task Judgers inherit from `BaseJudge` and implement the `compute_reward` method:

```python title="base_judge.py"
from ajet.task_judge.base_judge import BaseJudge
from ajet.workflow import WorkflowOutput, WorkflowTask

class BaseJudge:
    def __init__(self, config):
        self.config = config

    def compute_reward(
        self,
        workflow_task: WorkflowTask,
        workflow_output: WorkflowOutput
    ) -> tuple[float, bool]:
        """
        Args:
            workflow_task: Contains the task data, including metadata with reference answers
            workflow_output: Contains the agent's output, including metadata with generated answers

        Returns:
            tuple: (raw_reward: float, is_success: bool)
        """
        raise NotImplementedError
```


## Built-in Task Judgers

AgentJet provides three built-in judgers for common evaluation scenarios:

### 1. MathAnswerAsJudge

Evaluates mathematical answers by exact string matching, designed for tasks where answers are formatted in LaTeX `\boxed{}` notation.

!!! tip "When to use"
    - Math problem solving tasks
    - Tasks with deterministic, exact answers
    - Answers formatted as `\boxed{result}`

=== "Configuration"

    ```yaml title="config.yaml"
    ajet:
      task_judge:
        judge_type: customized_protocol
        judge_protocol: tutorial.example_math_agent.math_answer_as_judge->MathAnswerAsJudge
    ```

=== "How it works"

    1. Extracts the answer from `\boxed{...}` in the agent's output
    2. Compares with the reference answer from `workflow_task.task.metadata["answer"]`
    3. Returns `(1.0, True)` for correct answers, `(0.0, False)` otherwise

**Required metadata:**

| Field | Source | Description |
|-------|--------|-------------|
| `final_answer` | `workflow_output.metadata` | Agent's answer with `\boxed{}` format |
| `answer` | `workflow_task.task.metadata` | Reference answer |


### 2. CountdownAnswerAsJudge

Evaluates mathematical equations with partial credit for proper formatting.

!!! tip "When to use"
    - Number puzzle tasks (e.g., Countdown game)
    - Tasks where partial credit is appropriate
    - Need to reward proper formatting even when answer is wrong

=== "Configuration"

    ```yaml title="config.yaml"
    ajet:
      task_judge:
        judge_type: customized_protocol
        judge_protocol: tutorial.example_countdown.countdown_answer_as_judge->CountdownAnswerAsJudge
    ```

=== "Scoring"

    | Score | Condition |
    |-------|-----------|
    | `0.0` | Invalid or missing answer |
    | `0.1` | Properly formatted equation but wrong result |
    | `1.0` | Correct equation and result |


### 3. EnvServiceJudge

Delegates evaluation to an external environment service, useful for complex interactive environments.

!!! tip "When to use"
    - Tasks with external simulators (e.g., AppWorld)
    - Complex state-based evaluation
    - Interactive environments with built-in evaluators

```yaml title="config.yaml"
ajet:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: ajet.task_judge.env_service_as_judge->EnvServiceJudge
```

!!! note "How it works"
    1. Calls `workflow_task.gym_env.evaluate()` to get a score from the environment
    2. Converts the score to a normalized reward:
        - Success (score ≥ 1): `1.0 + score * 0.5`
        - Failure (score < 1): `0.0 + score * 0.5`


## Creating Custom Task Judgers

For specialized evaluation needs, create your own judger by inheriting `BaseJudge`:

<div class="workflow-single">
<div class="workflow-header">Custom Judger Steps</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>Implement Your Judger</strong>

Create a new file with your custom judger class.</li>
<li><strong>Configure Your Judger</strong>

Point to your custom class in the YAML configuration.</li>
<li><strong>Pass Data to the Judger</strong>

Populate `workflow_output.metadata` with the data your judger needs.</li>
</ol>
</div>
</div>

### Step 1: Implement Your Judger

```python title="tutorial/my_task/my_judge.py"
from ajet.task_judge.base_judge import BaseJudge
from ajet.workflow import WorkflowOutput, WorkflowTask

class MyCustomJudge(BaseJudge):
    def __init__(self, config):
        super().__init__(config)
        self.threshold = 0.8

    def compute_reward(
        self,
        workflow_task: WorkflowTask,
        workflow_output: WorkflowOutput
    ) -> tuple[float, bool]:
        agent_answer = workflow_output.metadata.get("final_answer", "")
        reference_answer = workflow_task.task.metadata.get("answer", "")

        similarity = self._compute_similarity(agent_answer, reference_answer)
        is_success = similarity >= self.threshold
        return similarity, is_success

    def _compute_similarity(self, text1: str, text2: str) -> float:
        return len(set(text1.split()) & set(text2.split())) / max(
            len(text1.split()), len(text2.split()), 1
        )
```

### Step 2: Configure Your Judger

```yaml title="config.yaml"
ajet:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: tutorial.my_task.my_judge->MyCustomJudge
```

### Step 3: Pass Data to the Judger

```python title="workflow.py"
class MyWorkflow(Workflow):
    async def execute(self, task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:
        final_answer = await self.agent.reply(msg)
        return WorkflowOutput(
            reward=None,  # Will be filled by the judger
            metadata={
                "final_answer": final_answer,
            }
        )
```


## Configuration Summary

```yaml title="config.yaml"
ajet:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: ajet.task_judge.<module>-><ClassName>
```


## Next Steps

<div class="card-grid">
<a href="../configuration/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:cog.svg" class="card-icon card-icon-tool" alt=""><h3>Configuration</h3></div><p class="card-desc">Complete reference for all configuration options.</p></a>
<a href="../example_math_agent/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:calculator-variant.svg" class="card-icon card-icon-math" alt=""><h3>Math Agent</h3></div><p class="card-desc">See MathAnswerAsJudge in a complete training example.</p></a>
</div>
