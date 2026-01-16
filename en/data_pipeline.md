# Task Reader

AgentJet loads training tasks from various data sources through Task Reader. This page covers the Task schema definition and different built-in Task Readers for common scenarios.

---

## Overview

In agent training, all training data must be represented as **tasks** following a unified schema.

!!! info "Key Concepts"
    - **Unified Schema**: All tasks conform to the `Task` structure regardless of source
    - **Multiple Sources**: Load from local files, HuggingFace datasets, interactive environments, or auto-generate new tasks
    - **Automatic Routing**: The framework selects the appropriate reader based on `ajet.task_reader.type`

```
Data Source → Task Reader → Unified Task Schema → Training Pipeline
```

---

## Task Schema

All training tasks must be defined according to the following structure:

```python
class Task(BaseModel):
    main_query: str = Field(default="")
    init_messages: List[dict] = Field(default=[])
    task_id: str = Field(default="")
    env_type: str = Field(default="")
    metadata: dict = Field(default_factory=dict)
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `main_query` | `str` | The main instruction or question for the agent to solve |
| `init_messages` | `List[dict]` | Initial conversation messages (e.g., system prompts). Each must have `role` and `content` fields |
| `task_id` | `str` | Unique identifier for the task |
| `env_type` | `str` | Environment type (e.g., "math", "appworld") |
| `metadata` | `dict` | Additional context information (e.g., reference answers for reward calculation) |

### Example Task

```json title="example_task.json"
{
  "main_query": "What is 15 * 23?",
  "init_messages": [
    {
      "role": "system",
      "content": "You are a helpful math assistant."
    }
  ],
  "task_id": "math_001",
  "env_type": "math",
  "metadata": {
    "answer": "345",
    "difficulty": "easy"
  }
}
```

!!! tip "Best Practices"
    - Use `metadata` to store information needed for reward computation (e.g., reference answers, scoring rubrics)
    - Keep `main_query` clear and concise
    - Use `init_messages` for system prompts or few-shot examples

---

## Built-in Task Readers

AgentJet provides multiple built-in Task Readers for different scenarios. The framework automatically routes to the correct reader based on `ajet.task_reader.type`.

### Quick Selection Guide

<div class="card-grid">
<div class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:file-document.svg" class="card-icon card-icon-data" alt=""><h3>JSONL File</h3></div><p class="card-desc">You have prepared task data in JSONL format locally.</p></div>
<div class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/simple-icons:huggingface.svg" class="card-icon card-icon-agent" alt=""><h3>HuggingFace</h3></div><p class="card-desc">Load tasks from HuggingFace Hub (e.g., GSM8K, MATH).</p></div>
<div class="feature-card-sm"><div class="card-header"><img src="https://api.iconify.design/mdi:web.svg" class="card-icon card-icon-tool" alt=""><h3>EnvService</h3></div><p class="card-desc">Tasks come from a running environment service.</p></div>
</div>

---

### 1. JSONL File Reader

**When to use:** You have prepared training tasks in JSONL format locally.

=== "Configuration"

    ```yaml title="config.yaml"
    ajet:
      task_reader:
        type: jsonl_dataset_file
        jsonl_dataset_file:
          training:
            file_path: "data/train.jsonl"
          validation:
            file_path: "data/val.jsonl"
    ```

=== "JSONL Format"

    Each line should be a JSON object conforming to the Task schema:

    ```json title="data/train.jsonl"
    {"main_query": "Solve: x + 5 = 12", "task_id": "algebra_01", "env_type": "math", "metadata": {"answer": "7"}}
    {"main_query": "What is the capital of France?", "task_id": "geo_01", "env_type": "qa", "metadata": {"answer": "Paris"}}
    ```

!!! note "How it works"
    - Reads tasks line-by-line from specified JSONL files
    - Automatically validates against Task schema
    - Supports separate training and validation splits

---

### 2. HuggingFace Dataset Reader

**When to use:** Load tasks from HuggingFace Hub datasets (e.g., GSM8K, MATH).

```yaml title="config.yaml"
ajet:
  task_reader:
    type: huggingface_dat_repo
    huggingface_dat_repo:
      dataset_path: "gsm8k"           # HF dataset repo name
      dataset_name: "main"            # Optional: dataset subset name
      training_split: "train"         # Training split name
      validation_split: "test"        # Validation split name
```

!!! note "How it works"
    - Downloads dataset from HuggingFace Hub using `datasets` library
    - Automatically maps dataset fields to Task schema
    - Caches downloaded data locally for faster subsequent runs

---

### 3. EnvService Reader

**When to use:** Tasks are provided by an interactive environment service (e.g., AppWorld, RL gym environments).

```yaml title="config.yaml"
ajet:
  task_reader:
    type: env_service
    env_service:
      env_type: "appworld"                 # Environment type
      env_url: "http://127.0.0.1:8080"    # Service URL
      env_action_preference: code          # Action format: code/text/box
      training_split: train
      validation_split: dev
```

!!! note "How it works"
    - Connects to a running environment service via HTTP
    - Pulls task instances from the environment
    - Supports dynamic task generation from interactive environments

!!! example "Use Cases"
    - Training agents in simulated environments (e.g., FrozenLake, game environments)
    - Complex interactive scenarios where tasks are generated dynamically

---

## Next Steps

<div class="card-grid">
<a href="../task_judger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>Task Judger</h3></div><p class="card-desc">Set up reward functions to evaluate agent outputs.</p></a>
<a href="../configuration/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:cog.svg" class="card-icon card-icon-tool" alt=""><h3>Configuration</h3></div><p class="card-desc">Complete reference for all configuration options.</p></a>
</div>
