# 任务加载器

AgentScope Tuner 通过 Task Reader 从多种数据源加载训练任务。本页面介绍 Task 的统一 Schema 定义，以及面向常见场景的内置 Task Reader。

## 概览

在智能体训练中，所有训练数据都必须以**任务（task）**的形式表示，并遵循统一的 Schema。AgentScope Tuner 提供了多个 Task Reader，用于从不同数据源加载任务：

* **统一 Schema**：无论数据源是什么，所有任务都统一映射为 `Task` 结构
* **多种来源**：支持从本地文件、HuggingFace 数据集、交互式环境加载，或自动生成新任务
* **自动路由**：框架会根据配置中的 `astuner.task_reader.type` 自动选择合适的 reader

```
Data Source → Task Reader → Unified Task Schema → Training Pipeline
```

## 任务结构

所有训练任务都必须按照以下结构定义：

```python
class Task(BaseModel):
    main_query: str = Field(default="")
    init_messages: List[dict] = Field(default=[])
    task_id: str = Field(default="")
    env_type: str = Field(default="")
    metadata: dict = Field(default_factory=dict)
```

### 字段说明

| 字段              | 类型           | 说明                                                      |
| --------------- | ------------ | ------------------------------------------------------- |
| `main_query`    | `str`        | 智能体需要解决的主要指令或问题                                         |
| `init_messages` | `List[dict]` | 初始对话消息（例如 system prompt）。每条消息必须包含 `role` 和 `content` 字段 |
| `task_id`       | `str`        | 任务的唯一标识                                                 |
| `env_type`      | `str`        | 环境类型（例如 `"math"`、`"appworld"`）                          |
| `metadata`      | `dict`       | 额外的上下文信息（例如用于奖励计算的参考答案）                                 |

### Task 示例

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

**最佳实践：**

* 使用 `metadata` 存储奖励计算所需的信息（例如参考答案、评分规则）
* 保持 `main_query` 清晰且简洁
* 使用 `init_messages` 提供 system prompt 或 few-shot 示例

## 内置 Task Readers

AgentScope Tuner 提供了 6 个内置 Task Reader，覆盖不同场景。框架会根据配置中的 `astuner.task_reader.type` 自动路由到正确的 reader。

### 快速选择指南

| 场景                  | Reader Type            | 适用情况                                 |
| ------------------- | ---------------------- | ------------------------------------ |
| **本地 JSONL 文件**     | `jsonl_dataset_file`   | 你已经准备好 JSONL 格式的任务数据                 |
| **HuggingFace 数据集** | `huggingface_dat_repo` | 从 HuggingFace Hub 加载任务（例如 GSM8K）     |
| **交互式环境**           | `env_service`          | 任务来自运行中的环境服务（例如 AppWorld、FrozenLake） |
| **从文档自动生成**         | `data_generation`      | 从知识文档生成任务或基于已有任务进行扩增                 |

---

### 1. JSONL 文件 Reader

**适用场景：**你在本地以 JSONL 格式准备了训练任务。

**配置方式：**

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

**JSONL 格式：**

每一行都应是一个符合 Task Schema 的 JSON 对象：

```json
{"main_query": "Solve: x + 5 = 12", "task_id": "algebra_01", "env_type": "math", "metadata": {"answer": "7"}}
{"main_query": "What is the capital of France?", "task_id": "geo_01", "env_type": "qa", "metadata": {"answer": "Paris"}}
```

**工作原理：**

* 按行读取指定的 JSONL 文件
* 自动按 Task Schema 校验数据
* 支持训练集与验证集分离

---

### 2. HuggingFace 数据集 Reader

**适用场景：**从 HuggingFace Hub 的数据集中加载任务（例如 GSM8K、MATH）。

**配置方式：**

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

**工作原理：**

* 使用 `datasets` 库从 HuggingFace Hub 下载数据集
* 自动将数据集字段映射到 Task Schema
* 将下载的数据缓存在本地，便于后续更快重复运行

**支持的数据集：**任意可映射到 Task Schema 的 HuggingFace 数据集。

---

### 3. EnvService Reader

**适用场景：**任务由交互式环境服务提供（例如 AppWorld、RL gym 环境）。

**配置方式：**

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

**工作原理：**

* 通过 HTTP 连接到正在运行的环境服务
* 从环境中拉取任务实例
* 支持从交互式环境中动态生成任务

**典型用例：**

* 在模拟环境中训练智能体（例如 FrozenLake、各类游戏环境）
* 任务需要动态生成、评测依赖环境状态的复杂交互场景
