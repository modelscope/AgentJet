# 数据生成

## 概述
`Data Generation`是一个智能化的数据增强框架，旨在简化高质量训练数据的创建过程。框架设计灵活、高效，提供两种强大的方法来生成领域特定的查询（Query）：

+ **基于文档的数据生成**：自动从文档（PDF、Word、文本文件）中提取知识，生成与上下文相关的查询
+ **Few-shot数据生成**：利用现有查询作为参考，创建风格一致、语义相似的新查询

## 基于文档的数据生成
### 📖 方法简介
`Document-based Data Generation` 能够基于文档自动生成高质量的训练任务。该模块借助大语言模型（LLM）的知识增强能力，自动生成新的查询（Query）数据及其对应的上下文信息。

### 🔧 架构
模块由两个主要组件组成：

1. **DocReader**: 解析文档（PDF、TXT、Word等）并提供智能缓存
2. **KnowledgeAugmentor**: 从文档内容生成新的任务

### 🌟 核心特性
**知识增强特性**

+ ✅ **全面覆盖**: 从文档中提取事实性、概念性、分析性和应用性任务
+ ✅ **上下文关联**: 每个生成任务都包含对应的文档摘录，确保答案可追溯
+ ✅ **可配置输出**: 可自定义生成任务数量（暂只支持N<10，若N>10，建议分batch跑）

### 🚀 快速开始
#### 步骤 1: 准备数据
将文档放置在指定目录：

```bash
mkdir -p dataset/document
cp your-document.pdf dataset/document/
```

#### 步骤 2: 编写配置文件
你需要创建一个 `.yaml` 配置文件，以下是一个配置示例 (`tests/data_gen.yaml`)：

```yaml
# tests/data_gen.yaml
astune:
  data_generation:
    document_reader:
      document_path: 'dataset/document/your-document.pdf'
      languages: ['eng']
      cache_enabled: true
    llm_model: qwen-long
    knowledge_augmentor:
      n: 10  # 生成10个任务
```

#### 步骤 3: 运行生成脚本
**方式A：使用测试脚本**

```bash
cd /path/to/astune
export DASHSCOPE_API_KEY='sk-xxxxxx|sk-yyyyyy'
export DASHSCOPE_API_KEY_BACKUP='sk-zzzzzz'
python tests/data_gen.py
```

**方式B：自定义脚本**

```python
import sys
sys.path.insert(0, '/path/to/astune')
import dotenv
dotenv.load_dotenv()

from astune.utils.config_utils import read_astune_config
from astune.task_reader.document_reader.doc_reader import DocReader
from astune.data_generator.knowledge_augmentation import KnowledgeAugmentor

# 加载配置
config = read_astune_config('tests/data_gen.yaml')

# 初始化组件
document_reader = DocReader(config)
knowledge_augmentor = KnowledgeAugmentor(config)

# 加载文档（带缓存）
document = document_reader.get_document()
print(f"文档已加载：{len(document.content)} 字符")

# 生成基于知识的任务
generated_tasks = knowledge_augmentor.generate_task(
    document=document
)

print(f"生成了 {len(generated_tasks)} 个任务")
for i, task in enumerate(generated_tasks[:3]):
    print(f"{i+1}. {task.main_query}")
```

### 示例输出
```json
[
  {
    "main_query": "What are the key requirements of Customer Due Diligence in AML procedures?",
    "related_doc": "Customer Due Diligence measures should include: (a) identifying the customer and verifying the customer's identity..."
  },
  {
    "main_query": "How should financial institutions handle Suspicious Transaction Reports?",
    "related_doc": "When someone knows or suspects that any property represents the proceeds of an indictable offense..."
  }
]
```



## 少样本数据生成
### 📖 方法简介
`Few-shot Data Generation`是一个基于 Few-shot Learning 的数据生成模块，它可以帮助你**自动生成新的查询（Query）数据**：

+ 给定一些现有的查询（Query）作为参考示例
+ 可选地提供一个文档（Document）作为背景知识
+ 该模块会利用大语言模型（LLM）生成风格相似、语义相关的新查询

### 🔧 架构
模块由三个主要组件组成：

1. **TaskReader**: 解析用户所提供的任务
2. **DocReader**: 解析文档（PDF、文本、Word等）并提供智能缓存
3. **TaskAugmentation**: 从用户所给的任务和提供的文档（可选）内容生成新的任务

### 🌟 核心特性
**任务增强特性**

+ ✅ **智能仿写模式**：基于参考查询生成风格一致、语义相关的新查询
+ ✅ **文档知识融合**：可选地结合文档上下文，生成主题契合的高质量查询
+ ✅ **可追溯性设计**：每个生成任务自动记录来源信息

### 🚀 快速开始
#### 步骤 1: 准备数据
准备原始查询数据

```json
{"main_query": "What is the capital of France?", "answer": "..."}
{"main_query": "How to cook pasta?", "answer": "..."}
```

准备文档（可选），将文档放置在指定目录：

```bash
mkdir -p dataset/document
cp your-document.pdf dataset/document/
```

#### 步骤 2: 编写配置文件
你需要创建一个 `.yaml` 配置文件，以下是一个配置示例 (`tests/data_gen.yaml`)：

```yaml
astune:

  data_generator:
    # (可选) 配置背景知识文档的读取方式
    document_reader:
      document_path: 'dataset/document/your-document.pdf'
      languages:
        - eng
    # 配置源任务文件的读取方式
    query_reader:
      type: dataset_file # 指定从本地文件读取
      dataset_file:
        training:
          file_path: 'dataset/jsonl/your-queries.jsonl' # 源任务文件路径

    # 配置用于生成任务的大语言模型
    llm_model: qwen-long # 使用的模型，例如 qwen-long
    llm_response_length: 8192
    sampling_params:
      temperature: 0
```

#### 步骤 3: 运行生成脚本
**方式A：使用测试脚本**

```bash
cd /path/to/astune
export DASHSCOPE_API_KEY='sk-xxxxxx|sk-yyyyyy'
export DASHSCOPE_API_KEY_BACKUP='sk-zzzzzz'
python tests/data_gen.py
```

**方式B：自定义脚本**

```python
# generate_tasks.py
import sys
sys.path.insert(0, '/path/to/astune')
import dotenv
dotenv.load_dotenv()

from astune.utils.config_utils import read_astune_config
from astune.task_reader import TaskReaderRouterV2
from astune.task_reader.document_reader.doc_reader import DocReader
from astune.data_generator.task_augmentation import TaskAugmentor

# 加载配置
config = read_astune_config('tests/data_gen.yaml')

# 初始化组件
task_reader = TaskReaderRouterV2(
    reader_type=config.task_reader.data_generation.query_reader.type,
    reader_config=config.task_reader.data_generation.query_reader
)
document_reader = DocReader(config)
task_augmentor = TaskAugmentor(config)

# 加载数据
original_tasks = task_reader.get_training_tasks()
document = document_reader.get_document()
print(f"原始任务数：{len(original_tasks)}。")
print(f"文档已加载：{len(document.content)} 字符。\n")

# 生成新任务
new_tasks = []
for task in original_tasks[:5]: # 先测试 5 个，每次读取一个query作为参考
    new_task = task_augmentor.generate_task(
        source_task=task,
        document=document
    )
    new_tasks.append(new_task)

print(f"生成了 {len(new_tasks)} 个新任务：")
for i, task in enumerate(new_tasks):
    print(f"{i+1}. {task.main_query}")

```

### 示例输出
```json
[
  {
    "main_query": "Are financial institutions required to verify the source of funds for corporate clients during account opening?"
  },
  {
    "main_query": "What are the requirements for banks to verify customer identities under anti-money laundering regulations?"
  }
]
```



## 详细配置
### 基础配置 (`data_gen.yaml`)
```yaml
astune:
  # 数据生成器配置
  data_generation:
    # 文档读取器配置
    document_reader:
      document_path: 'dataset/document/your-document.pdf'
      languages:
        - eng
      cache_enabled: true
      cache_format: json
    # 任务读取器（用于现有任务）
    query_reader:
      type: dataset_file
      dataset_file:
        training:
          file_path: 'dataset/jsonl/your-tasks.jsonl'

    # LLM配置
    llm_model: qwen-long
    llm_response_length: 8192
    sampling_params:
      temperature: 0

    # 知识增强设置
    knowledge_augmentor:
      n: 10    # 从文档生成的任务数量
```

### 配置选项
**知识增强器选项**

| 选项 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| n | integer | 10 | 生成任务数量 |
| llm_model | string | 必需 | 用于生成的LLM模型 |
| llm_response_length | integer | 8192 | 最大响应长度 |
| sampling_params | dict | {} | LLM采样参数 |