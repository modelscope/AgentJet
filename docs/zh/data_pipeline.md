# 数据管线

AgentScope Tuner 为不同数据源提供了统一的数据结构，以及对应的数据加载机制。本文将详细介绍 AgentScope Tuner 中的数据格式与数据读取器。

## 数据格式
AgentScope Tuner 定义了如下的数据结构：

```python
class Task(BaseModel):
    main_query: str = Field(default="")
    init_messages: List[dict] = Field(default=[])
    task_id: str = Field(default="")
    env_type: str = Field(default="")
    metadata: dict = Field(default_factory=dict)
```

各字段含义如下：

+ `main_query`：该任务的主查询或问题。
+ `init_messages`：初始的对话消息列表，通常可用于包含 system 消息等。每个元素必须包含 `role` 和 `content` 字段。
+ `task_id`：该任务的唯一标识符。
+ `env_type`：该任务所对应的环境类型。
+ `metadata`：任务的元数据，用于存储额外的上下文信息。

`metadata` 中的字段与当前实际在做的训练任务相关。例如，可以使用 `metadata` 保存用于奖励计算的数据等。

## 数据读取器
为方便使用，我们为常见场景准备了多种数据读取器，包括：从文件读取、从 Huggingface 仓库读取、从 EnvService 读取，以及完全自定义代码读取。下面将分别进行介绍。

### File Reader
File Reader 可以从本地路径读取 jsonl 格式的数据集。在配置文件中将 `astuner.task_reader.type` 设置为 `dataset_file` 即可启用该 reader。

启用后，还需要分别配置训练集与验证集的文件路径：

```yaml
astuner:
  task_reader:
    dataset_file:
      training:
        # 训练数据集路径
        file_path: "xxxx.jsonl"
      validation:
        # 验证数据集路径
        file_path: "xxxx.jsonl"
```

JSONL 文件中的每一行都应为一个 JSON 对象，结构如下：

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

符合上述格式的 JSONL 文件会被自动加载并作为数据集使用。

### Huggingface Repo Reader
Huggingface Repo Reader 可以从 Huggingface 仓库中读取远程数据集。在配置文件中将 `astuner.task_reader.type` 设置为 `huggingface_dat_repo` 即可启用该 reader。

启用后，需要指定仓库名称与训练集划分、验证集划分的名称：

```yaml
astuner:
  task_reader:
    huggingface_dat_repo:
      # 数据集所在仓库名称
      dataset_path: "gsm8k"
      # 训练集划分的名称
      training_split: "train"
      # 验证集划分的名称
      validation_split: "validation"
```

### EnvService Reader
EnvService Reader 可以自动从 EnvService 中拉取远程数据集。在配置文件中将 `astuner.task_reader.type` 设置为 `env_service` 即可启用该 reader。

启用后，需要配置服务 URL、环境类型以及训练集划分、验证集划分的名称：

```yaml
astuner:
  task_reader:
    env_service:
      # 环境类型，需要事先在 EnvService 中初始化该环境才能使用
      env_type: "appworld"
      # EnvService 服务地址
      env_url: "http://127.0.0.1:8080"
      # 支持的 action 形式：code, text, box
      env_action_preference: code
      # 该环境中训练集的划分名称
      training_split: train
      # 该环境中验证集的划分名称
      validation_split: dev
```

### Random Dummy Reader
如果你希望完全自定义数据管线，可以使用该 reader。

Random Dummy Reader 会将数据加载过程完全交给用户。在 `Workflow` 的 `workflow_task.task_id` 参数中可以读取到一个传入的随机整数，你可以将其作为任务 ID，自行完成数据加载逻辑。

```yaml
astuner:
  task_reader:
    type: random_dummy
```
