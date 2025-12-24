# Auto Grader Judge

A data-driven judge that automatically generates evaluation rubrics from reference samples using RM Gallery's Iterative Rubrics Generator.

## What is Auto Grader Judge?

Auto Grader Judge is an intelligent evaluation system that **learns how to grade your AI agent's outputs** by analyzing examples of good and bad responses. Instead of manually writing evaluation rules, it automatically discovers scoring criteria from your training data through an iterative Propose-Evaluate-Revise process, then generates structured rubrics (Theme-Tips format) that can be inspected and understood.

**Key Features:**
- Automatic rubric generation from reference samples
- Support for both pointwise (scoring) and listwise (ranking) evaluation
- MCR²-based smart sampling for large datasets
- Optional LLM-based categorization
- Seamless integration with astuner's workflow system

### When to Use Auto Grader Judge?

**✅ Ideal For:**
- **Open-ended tasks**: Dialogue generation, creative writing, explanations
- **Subjective quality assessment**: Where "correctness" has nuance (helpfulness, clarity, style)
- **Complex multi-aspect evaluation**: Need to assess accuracy, completeness, fluency, etc.
- **Large-scale RL training**: Need automated, consistent evaluation with reward signals

**⚠️ Not Recommended For:**
- **Tasks with exact answers**: Use `EnvServiceJudge` or exact match instead
- **Fully objective tasks**: API calls, code execution, mathematical computation

## Quick Start

### 1. Configuration

Add to your `astune_default.yaml`:

```yaml
astuner:
  task_judge:
    judge_type: rubrics_auto_grader

    rubrics_auto_grader:
      # Model settings
      model_name: qwen-max

      # Grader configuration
      grader_mode: pointwise  # or "listwise"
      language: en  # or "zh"

      # auto grader configuration
      query_specific_generate_number: 1
      enable_categorization: false
      categories_number: 5

      # Custom evaluation prompt
      custom_evaluation_prompt: null

      # Field mappings
      query_field: main_query
      answer_field: final_answer
      reference_field: answer

      # Training data
      input_data_type: dataset_file
      dataset_file:
        training:
          file_path: "path/to/training_data.jsonl"

      # Pointwise mode only
      min_score: 0
      max_score: 10
```

### 2. Training Data Format

#### Pointwise Mode
Each sample contains a query, answer, and score:

```json
{
  "main_query": "What is 2 + 2?",
  "metadata": {
    "answer": "2 + 2 = 4",
    "score": 1
  }
}
```

#### Listwise Mode
Each sample contains a query with multiple ranked candidates:

```json
{
  "main_query": "What is 2 + 2?",
  "metadata": {
    "candidates": [
      {"answer": "2 + 2 = 4", "rank": 1},
      {"answer": "2 + 2 = 5", "rank": 2},
      {"answer": "I don't know", "rank": 3}
    ]
  }
}
```

### 3. Basic Usage

```python
from astuner.task_judge.rm_auto_grader_judge import AutoGraderJudge

# Initialize judge
judge = AutoGraderJudge(config)

# Generate rubrics (one-time setup)
await judge.generate_rubrics_from_samples()

# Or load from cache
await judge.load_rubrics_from_cache()

# Evaluate outputs
result = await judge._async_compute_reward(task, workflow_output)

# For pointwise: result is a GraderScore object
print(f"Score: {result.score}, Reason: {result.reason}")

# For listwise: result is a GraderRank object
print(f"Ranks: {result.rank}, Reason: {result.reason}")
```

## Configuration Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | str | DashScope model name (e.g., qwen-max, qwen-plus) |
| `grader_mode` | str | Evaluation mode: "pointwise" or "listwise" |
| `language` | str | Language: "en" or "zh" |
| `input_data_type` | str | Data source type: "dataset_file", "env_service", etc. |

### Field Mapping

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query_field` | "main_query" | Field name containing the query |
| `answer_field` | "final_answer" | Field name containing the answer |
| `reference_field` | "answer" | Field name containing the reference |

### Pointwise Mode Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_score` | 0 | Minimum score value |
| `max_score` | 10 | Maximum score value |

### Advanced Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grader_name` | "RM Iterative Rubric Grader" | Name of the grader |
| `enable_categorization` | false | Enable LLM-based rubric categorization |
| `categories_number` | 5 | Number of categories (if enabled) |
| `query_specific_generate_number` | 1 | Number of rubrics per sample |
| `custom_evaluation_prompt` | null | Path to custom evaluation prompt file |


## Evaluation Modes

### Pointwise Mode
Evaluates individual outputs independently.

**Returns:** `GraderScore` object
- `score`: Numerical score (float)
- `reason`: Explanation for the score
- `metadata`: Additional information

**Use case:** Absolute quality assessment

### Listwise Mode
Ranks multiple outputs together.

**Returns:** `GraderRank` object
- `rank`: List of rankings (e.g., [1, 3, 2])
- `reason`: Explanation for the ranking
- `metadata`: Additional information

**Use case:** Relative comparison, preference ranking

## Cache Management

Generated rubrics are automatically saved to:
```
{experiment_dir}/auto_grader.json
```

To reuse rubrics:
```python
await judge.load_rubrics_from_cache()
```

This skips the generation phase and loads pre-generated rubrics.

## Example: Pointwise Evaluation

```python
# Create reference samples
reference_samples = [
    Task(
        task_id="1",
        main_query="What is 5 + 3?",
        metadata={"answer": "5 + 3 = 8", "score": 1}
    ),
    # ... more samples
]

# Initialize and generate rubrics
judge = AutoGraderJudge(config)
await judge.generate_rubrics_from_samples(reference_samples)

# Create test task and output
test_task = Task(task_id="test_1", main_query="What is 7 + 2?")
test_output = WorkflowOutput(metadata={"final_answer": "7 + 2 = 9"})

# Evaluate
result = await judge._async_compute_reward(test_task, test_output)

print(f"Score: {result.score}")
print(f"Reasoning: {result.reason}")
```

## Example: Listwise Evaluation

```python
# Create reference samples with rankings
reference_samples = [
    Task(
        task_id="1",
        main_query="Explain photosynthesis",
        metadata={
            "candidates": [
                {"answer": "Detailed scientific explanation...", "rank": 1},
                {"answer": "Brief explanation...", "rank": 2},
                {"answer": "Incorrect explanation...", "rank": 3}
            ]
        }
    ),
    # ... more samples
]

# Initialize and generate rubrics
judge = AutoGraderJudge(config)
await judge.generate_rubrics_from_samples(reference_samples)

# Create test task with multiple candidates
test_task = Task(task_id="test_1", main_query="What is the water cycle?")
candidates = [
    WorkflowOutput(metadata={"final_answer": "Water evaporates, forms clouds, and rains"}),
    WorkflowOutput(metadata={"final_answer": "It's when water moves around"}),
    WorkflowOutput(metadata={"final_answer": "Detailed explanation of evaporation, condensation, precipitation..."})
]

# Evaluate
result = await judge._async_compute_reward(test_task, candidates)

print(f"Rankings: {result.rank}")  # e.g., [2, 3, 1]
print(f"Reasoning: {result.reason}")
```
