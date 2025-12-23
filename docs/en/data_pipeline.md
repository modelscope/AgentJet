# Task Reader

AgentScope Tuner loads training tasks from various data sources through Task Reader. This page covers the Task schema definition and different built-in Task Readers for common scenarios.

## Overview

In agent training, all training data must be represented as **tasks** following a unified schema. AgentScope Tuner provides multiple Task Readers to load tasks from different data sources:

- **Unified Schema**: All tasks conform to the `Task` structure regardless of source
- **Multiple Sources**: Load from local files, HuggingFace datasets, interactive environments, or auto-generate new tasks
- **Automatic Routing**: The framework selects the appropriate reader based on `astuner.task_reader.type` in your configuration

```
Data Source → Task Reader → Unified Task Schema → Training Pipeline
```

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

```json
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

**Best Practices:**
- Use `metadata` to store information needed for reward computation (e.g., reference answers, scoring rubrics)
- Keep `main_query` clear and concise
- Use `init_messages` for system prompts or few-shot examples

## Built-in Task Readers

AgentScope Tuner provides six built-in Task Readers for different scenarios. The framework automatically routes to the correct reader based on `astuner.task_reader.type` in your configuration.

### Quick Selection Guide

| Scenario | Reader Type | When to Use |
|----------|-------------|-------------|
| **Local JSONL file** | `jsonl_dataset_file` | You have prepared task data in JSONL format |
| **HuggingFace dataset** | `huggingface_dat_repo` | Load tasks from HuggingFace Hub (e.g., GSM8K) |
| **Interactive environment** | `env_service` | Tasks come from a running environment service (e.g., AppWorld, FrozenLake) |
| **Auto-generate from documents** | `data_generation` | Generate tasks from knowledge documents or existing tasks |

---

### 1. JSONL File Reader

**When to use:** You have prepared training tasks in JSONL format locally.

**Configuration:**

```yaml
astuner:
  task_reader:
    type: jsonl_dataset_file
    jsonl_dataset_file:
      training:
        file_path: "data/train.jsonl"
      validation:
        file_path: "data/val.jsonl"
```

**JSONL Format:**

Each line should be a JSON object conforming to the Task schema:

```json
{"main_query": "Solve: x + 5 = 12", "task_id": "algebra_01", "env_type": "math", "metadata": {"answer": "7"}}
{"main_query": "What is the capital of France?", "task_id": "geo_01", "env_type": "qa", "metadata": {"answer": "Paris"}}
```

**How it works:**
- Reads tasks line-by-line from specified JSONL files
- Automatically validates against Task schema
- Supports separate training and validation splits

---

### 2. HuggingFace Dataset Reader

**When to use:** Load tasks from HuggingFace Hub datasets (e.g., GSM8K, MATH).

**Configuration:**

```yaml
astuner:
  task_reader:
    type: huggingface_dat_repo
    huggingface_dat_repo:
      dataset_path: "gsm8k"           # HF dataset repo name
      dataset_name: "main"            # Optional: dataset subset name
      training_split: "train"         # Training split name
      validation_split: "test"        # Validation split name
```

**How it works:**
- Downloads dataset from HuggingFace Hub using `datasets` library
- Automatically maps dataset fields to Task schema
- Caches downloaded data locally for faster subsequent runs

**Supported datasets:** Any HuggingFace dataset that can be mapped to the Task schema.

---

### 3. EnvService Reader

**When to use:** Tasks are provided by an interactive environment service (e.g., AppWorld, RL gym environments).

**Configuration:**

```yaml
astuner:
  task_reader:
    type: env_service
    env_service:
      env_type: "appworld"                 # Environment type
      env_url: "http://127.0.0.1:8080"    # Service URL
      env_action_preference: code          # Action format: code/text/box
      training_split: train
      validation_split: dev
```

**How it works:**
- Connects to a running environment service via HTTP
- Pulls task instances from the environment
- Supports dynamic task generation from interactive environments

**Use cases:**
- Training agents in simulated environments (e.g., FrozenLake, game environments)
- Complex interactive scenarios where tasks are generated dynamically

---

### 4. Data Generation Reader

**When to use:** Automatically generate training tasks from knowledge documents or augment existing tasks.

**Configuration:**

```yaml
astuner:
  task_reader:
    type: data_generation
    data_generation:
      augmentor_type: knowledge        # 'knowledge' or 'task'
      num_workers: 4                   # Parallel generation workers
      
      # For knowledge-based generation:
      knowledge:
        doc_reader_type: pdf
        doc_reader_config:
          base_url: "docs/knowledge_base/"
        generator_config:
          model_name: "gpt-4"
          num_tasks_per_doc: 5
      
      # For task-based augmentation:
      task:
        base_tasks_path: "data/seed_tasks.jsonl"
        augmentor_config:
          model_name: "gpt-4"
          augmentation_ratio: 2.0
```

**Two Generation Modes:**

1. **Knowledge Augmentation** (`augmentor_type: knowledge`)
   - Reads knowledge documents (PDF, TXT, Markdown)
   - Uses LLM to generate tasks based on document content
   - Useful for domain-specific knowledge training

2. **Task Augmentation** (`augmentor_type: task`)
   - Takes existing seed tasks
   - Uses LLM to create variations and similar tasks
   - Expands training data from a small set of examples

**How it works:**
- Generates tasks using configured LLM
- Applies deduplication filters to ensure diversity
- Caches generated tasks (keyed by configuration hash)
- Supports parallel generation with multiple workers

**Use cases:**
- Bootstrap training data from documentation
- Augment limited training examples
- Create diverse task variations


