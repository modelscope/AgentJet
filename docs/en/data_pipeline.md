# Data

AgentScope Tuner establishes a unified training data structure and complete data loading methods for data from various sources. This page provides a detailed description of the data schema and data loaders in AgentScope Tuner.

## Data Schema
Data structures that can be read and used by AgentScope Tuner must be defined according to the following format:

```python
class Task(BaseModel):
    main_query: str = Field(default="")
    init_messages: List[dict] = Field(default=[])
    task_id: str = Field(default="")
    env_type: str = Field(default="")
    metadata: dict = Field(default_factory=dict)
```

The meanings of each field are as follows:

+ `main_query`: The main instruction or question of the task.
+ `init_messages`: List of initial conversation messages, typically used to include system messages, etc. Each element must contain `role` and `content` fields.
+ `task_id`: The unique identifier of the task.
+ `env_type`: The environment type corresponding to the task.
+ `metadata`: Metadata dictionary for the task, used to store additional context information.

The fields in `metadata` are related to the actual training task currently being processed. For example, we can use `metadata` to save data used for reward calculation, etc.

## Data Readers
To facilitate usage, we have prepared various data readers for common scenarios, including reading from files, reading from Huggingface repos, reading from EnvService, reading with custom code. In this section, we will introduce each reader in detail.

### File Reader
The File Reader can read a dataset in jsonl format from a local path. Setting `astuner.task_reader.type` to `dataset_file` in the configuration file enables this reader.

After enabling this reader, you also need to set the file paths for the training and validation sets.

```yaml
astuner:
  task_reader:
    dataset_file:
      training:
        # the path of training dataset
        file_path: "xxxx.jsonl"
      validation:
        # the path of validation dataset
        file_path: "xxxx.jsonl"
```



Each line in the JSONL should be a JSON object with the following structure:

```json
{
  "main_query": "the query",
  "init_messages": [
    {
      "role": "system",
      "content": "openai format message",
    }
  ],
  "task_id": "the task id",
  "env_type": "the environment of the task",
  "metadata": {
    "other": "other metadata",
  }
}
```

JSONL files conforming to the above format will be automatically loaded and used as datasets.

### Huggingface Repo Reader
The Huggingface Repo Reader can read a remote dataset from a Huggingface Repo. Setting `astuner.task_reader.type` to `huggingface_dat_repo` in the configuration file enables this reader.

After enabling this reader, you also need to set the repo name and split names:

```yaml
astuner:
  task_reader:
    huggingface_dat_repo:
      # the repo name
      dataset_path: "gsm8k"
      # the name of training split
      training_split: "train"
      # the name of validation split
      validation_split: "validation"
```

### EnvService Reader
The EnvService Reader can automatically pull remote datasets from the EnvService. Setting `astuner.task_reader.type` to `env_service` in the configuration file enables this reader.

After enabling this reader, you also need to set the service URL, environment type, and split names:

```yaml
astuner:
  task_reader:
    env_service:
      # the type of env, must be init in EnvService first
      env_type: "appworld"
      # the url of the service
      env_url: "http://127.0.0.1:8080"
      # code, text, box
      env_action_preference: code
      # the name of training split in this environment
      training_split: train
      # the name of validation split in this environment
      validation_split: dev
```

### Random Dummy Reader
If you want to customize the data pipeline, simply use this reader.

The Random Dummy Reader will leave all things to you, passing a random integer as `workflow_task.task_id` in the `Workflow` for you to load the data with your own process.

```yaml
astuner:
  task_reader:
    type: random_dummy
```
