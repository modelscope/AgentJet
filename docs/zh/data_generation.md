# 数据生成

## 概述
`Data Generation Reader` 提供了一个智能化的数据生成方法，旨在简化高质量训练数据的创建过程。方法设计灵活、高效，可以基于Few-shot数据与文档（可选）来生成领域特定的任务（Query）。

## 方法简介
`Data Generation Reader` 采用两阶段任务生成流程：

### 第一阶段（可选）：基于文档的数据生成
此阶段为可选步骤。`Document-based Data Generation` 会基于提供的文档内容，生成知识类提问任务。用户可以提供一个或多个文档（支持 PDF、Word、TXT 等格式）：

```plain
According to the Anti-Money Laundering and Counter-Terrorist Financing Ordinance and related Guideline, banks are required to identify and take reasonable measures to verify the identity of the beneficial owner of corporate customers so that the bank is ...
```

生成器会读取文档内容，并借助大语言模型批量生成与文档知识相关的提问任务：

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
  ...
]
```

若提供文档进行生成数据，该部分生成的数据会补充到后续的训练过程中的验证任务集合。

### 第二阶段：少样本数据生成
此阶段会生成最终的训练任务。`Few-shot Data Generation` 将少量用户提供的任务与第一阶段生成的知识类任务的组合，并参考文档内容生成训练任务。首先，用户需要提供少量的任务示例：

```json
{"main_query": "Can banks ask corporate customers to provide information of its ownership?", "answer": "According to the Anti-Money Laundering and ..."}
{"main_query": "Can a bank close my account?", "answer": "Either a customer or a bank may close an account at any time subject to any specific terms and ..."}
...
```

这些示例将与第一阶段生成的任务合并，构成一个完整的示例任务集合。生成器会从此集合中进行采样，作为少样本（Few-shot）任务演示，并结合相关的文档内容，引导大模型批量生成训练任务：

```json
[
  {
    "main_query": "Are financial institutions required to verify the source of funds for corporate clients during account opening?"
  },
  {
    "main_query": "What are the requirements for banks to verify customer identities under anti-money laundering regulations?"
  }
  ...
]
```

## 🚀 快速开始
`Data Generation Reader` 可以从本地路径读取用户提供的少量任务以及PDF、Word、TXT等多种格式的文档（可选），生成任务并读取为训练任务。

### 步骤 1: 准备数据
提供少量原始任务数据：

```json
{"main_query": "What is the capital of France?", "answer": "..."}
{"main_query": "How to cook pasta?", "answer": "..."}
```

提供文档（可选），将文档放置在指定目录：

```bash
mkdir -p dataset/document
cp your-document.pdf dataset/document/
```

### 步骤 2: 生成训练任务
#### 方式1：将生成任务接入训练流程
拷贝并修改 `astuner/default_config/astune_default.yaml` 中的关键配置参数，将`astuner.task_reader.type` 设置为 `data_generation` 即可启用该 reader。

```yaml
astuner:
  task_reader:
    type: data_generation
    # when `type == data_generation`
    data_generation:
      # 文档读取器配置
      document_reader:
        document_path:
          - 'dataset/document/your-document1.pdf'
          - 'dataset/document/your-document2.pdf'
        languages:
          - eng
      # 任务读取器（用于现有任务）
      query_reader:
        type: jsonl_dataset_file
        jsonl_dataset_file:
          training:
            file_path: 'dataset/jsonl/your-queries.jsonl'
      # 生成任务的数量
      task_num: 10
      # LLM配置
      llm_model: qwen-long
      llm_response_length: 8192
      num_workers: 32
      sampling_params:
        temperature: 0
      # 任务过滤配置
      deduplication_filter:
        enabled: true
        params:
          similarity_threshold: 0.8
          db_path: ./.similarity_db
          model: text-embedding-v4
          api_key: null # load from the env
          base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```

#### 方式2：单独运行生成脚本
```python
from astuner.data_generator.config import *
from astuner.task_reader.data_generator_reader import DataGeneratorTaskReader

def run():
    config = TaskReaderConfig(
        data_generation=DataGenerationConfig(
            document_reader=DocumentReaderConfig(
                document_path=['dataset/document/your-document1.pdf', 'dataset/document/your-document2.pdf'],
                languages=["eng"],
                chunk_size=5120,
                split_by="sentence",
            ),
            query_reader=QueryReaderConfig(
                type="jsonl_dataset_file",
                jsonl_dataset_file=DatasetFileConfig(
                    training=TrainingDatasetConfig(file_path='dataset/jsonl/your-queries.jsonl')
                ),
            ),
            task_num=50,
            llm_model="qwen-long",
            num_workers=16,
            sampling_params=SamplingParamsConfig(temperature=0.0),
            deduplication_filter=DeduplicationFilterConfig(
                enabled=True,
                params=DeduplicationFilterParamsConfig(
                    similarity_threshold=0.8,
                    model="text-embedding-v4",
                ),
            ),
        )
    )
    reader = DataGeneratorTaskReader(reader_config=config)

run()
```

## 生成任务示例
`Data Generation Reader`基于用户提供的文档（可选）与少量任务示例，即可批量生成训练任务：

```json
[
  {
    "main_query": "Are financial institutions required to verify the source of funds for corporate clients during account opening?"
  },
  {
    "main_query": "What are the requirements for banks to verify customer identities under anti-money laundering regulations?"
  }
  ...
]
```

## 详细配置选项
| 参数路径 | 类型 | 默认值 | 必填 | 说明 |
| --- | --- | --- | --- | --- |
| `document_reader.document_path` | list[str] | - | ❌ | 文档文件路径列表，支持 PDF、Word、TXT 等多种格式 |
| `document_reader.languages` | list[str] | `['eng']` | ❌ | 文档语言列表，用于 OCR 和文本解析，如 `eng`（英语）、`chs`（简体中文） |
| `query_reader.type` | str | `jsonl_dataset_file` | ✅ | 读取器类型，可选：`jsonl_dataset_file`、`env_service`、`huggingface_dat_repo` |
| `query_reader.jsonl_dataset_file.training.file_path` | str | - | ✅ | 训练任务 JSONL 文件路径（当 `type: jsonl_dataset_file` 时） |
| `task_num` | int | `10` | ✅ | 要生成的任务数量，实际数量可能因过滤而减少 |
| `llm_model` | str | `qwen-long` | ✅ | 用于生成任务的 LLM 模型名称 |
| `llm_response_length` | int | `8192` | ❌ | LLM 响应的最大 token 长度 |
| `num_workers` | int | `32` | ❌ | 并行工作线程数，用于多线程加速任务生成 |
| `sampling_params.temperature` | float | `0` | ❌ | 采样温度，0 表示贪婪解码（确定性输出），值越高输出越随机 |
| `deduplication_filter.enabled` | bool | `true` | ❌ | 是否启用去重过滤器 |
| `deduplication_filter.params.similarity_threshold` | float | `0.8` | ✅ | 相似度阈值 (0-1)，超过此值的任务会被过滤 |
| `deduplication_filter.params.db_path` | str | `./.similarity_db` | ❌ | 相似度数据库存储路径，用于缓存 embedding |
| `deduplication_filter.params.model` | str | `text-embedding-v4` | ✅ | 用于计算相似度的 embedding 模型 |
| `deduplication_filter.params.api_key` | str | `null` | ❌ | API Key，为 `null` 时从环境变量 `DASHSCOPE_API_KEY` 加载 |
| `deduplication_filter.params.base_url` | str | `https://dashscope.aliyuncs.com/compatible-mode/v1` | ❌ | Embedding API 的基础 URL |