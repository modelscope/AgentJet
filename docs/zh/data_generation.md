# 数据生成

`Data Generation Reader` 提供了一个智能化的数据生成方法，旨在简化高质量训练数据的创建过程。方法设计灵活、高效，可以基于 Few-shot 数据与文档（可选）来生成领域特定的任务（Query）。

---

## 概述

!!! info "核心特点"
    `Data Generation Reader` 采用两阶段任务生成流程，可以从少量样本和文档中自动生成大量训练任务。

---

## 方法简介

### 第一阶段（可选）：基于文档的数据生成

此阶段为可选步骤。`Document-based Data Generation` 会基于提供的文档内容，生成知识类提问任务。

=== "输入：文档内容"

    ```plain
    According to the Anti-Money Laundering and Counter-Terrorist 
    Financing Ordinance and related Guideline, banks are required 
    to identify and take reasonable measures to verify the identity 
    of the beneficial owner of corporate customers so that the bank is ...
    ```

=== "输出：生成的任务"

    ```json
    [
      {
        "main_query": "What are the key requirements of Customer Due Diligence in AML procedures?",
        "related_doc": "Customer Due Diligence measures should include: (a) identifying the customer..."
      },
      {
        "main_query": "How should financial institutions handle Suspicious Transaction Reports?",
        "related_doc": "When someone knows or suspects that any property represents the proceeds..."
      }
    ]
    ```

!!! tip "文档生成的数据用途"
    若提供文档进行生成数据，该部分生成的数据会补充到后续的训练过程中的验证任务集合。

### 第二阶段：少样本数据生成

此阶段会生成最终的训练任务。`Few-shot Data Generation` 将少量用户提供的任务与第一阶段生成的知识类任务组合，并参考文档内容生成训练任务。

=== "输入：少量任务示例"

    ```json
    {"main_query": "Can banks ask corporate customers to provide information of its ownership?", "answer": "According to the Anti-Money Laundering..."}
    {"main_query": "Can a bank close my account?", "answer": "Either a customer or a bank may close an account at any time..."}
    ```

=== "输出：批量生成的任务"

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

---

## 快速开始

`Data Generation Reader` 可以从本地路径读取用户提供的少量任务以及 PDF、Word、TXT 等多种格式的文档（可选），生成任务并读取为训练任务。

### 步骤 1：准备数据

**提供少量原始任务数据：**

```json title="your-queries.jsonl"
{"main_query": "What is the capital of France?", "answer": "..."}
{"main_query": "How to cook pasta?", "answer": "..."}
```

**提供文档（可选）：**

```bash
mkdir -p dataset/document
cp your-document.pdf dataset/document/
```

### 步骤 2：生成训练任务

=== "方式 1：接入训练流程"

    修改配置文件，将 `astuner.task_reader.type` 设置为 `data_generation`：

    ```yaml title="config.yaml"
    astuner:
      task_reader:
        type: data_generation
        data_generation:
          # 文档读取器配置
          document_reader:
            document_path:
              - 'dataset/document/your-document1.pdf'
              - 'dataset/document/your-document2.pdf'
            languages:
              - eng
          # 任务读取器配置
          query_reader:
            type: jsonl_dataset_file
            jsonl_dataset_file:
              training:
                file_path: 'dataset/jsonl/your-queries.jsonl'
          # 生成任务的数量
          task_num: 10
          # LLM 配置
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
              api_key: null
              base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    ```

=== "方式 2：单独运行脚本"

    ```python title="generate_data.py"
    from agentscope_tuner.data_generator.config import *
    from agentscope_tuner.task_reader.data_generator_reader import DataGeneratorTaskReader

    def run():
        config = TaskReaderConfig(
            data_generation=DataGenerationConfig(
                document_reader=DocumentReaderConfig(
                    document_path=['dataset/document/your-document1.pdf'],
                    languages=["eng"],
                    chunk_size=5120,
                    split_by="sentence",
                ),
                query_reader=QueryReaderConfig(
                    type="jsonl_dataset_file",
                    jsonl_dataset_file=DatasetFileConfig(
                        training=TrainingDatasetConfig(
                            file_path='dataset/jsonl/your-queries.jsonl'
                        )
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

---

## 生成任务示例

`Data Generation Reader` 基于用户提供的文档（可选）与少量任务示例，即可批量生成训练任务：

```json title="生成的任务"
[
  {
    "main_query": "Are financial institutions required to verify the source of funds for corporate clients during account opening?"
  },
  {
    "main_query": "What are the requirements for banks to verify customer identities under anti-money laundering regulations?"
  }
]
```

---

## 详细配置选项

### 文档读取器配置

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `document_path` | `list[str]` | - | 否 | 文档文件路径列表，支持 PDF、Word、TXT 等 |
| `languages` | `list[str]` | `['eng']` | 否 | 文档语言列表，如 `eng`、`chs` |

### 任务读取器配置

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `type` | `str` | `jsonl_dataset_file` | 是 | 读取器类型：`jsonl_dataset_file`、`env_service`、`huggingface_dat_repo` |
| `file_path` | `str` | - | 是 | 训练任务 JSONL 文件路径 |

### 生成配置

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `task_num` | `int` | `10` | 是 | 要生成的任务数量 |
| `llm_model` | `str` | `qwen-long` | 是 | 用于生成任务的 LLM 模型 |
| `llm_response_length` | `int` | `8192` | 否 | LLM 响应的最大 token 长度 |
| `num_workers` | `int` | `32` | 否 | 并行工作线程数 |
| `temperature` | `float` | `0` | 否 | 采样温度，0 表示贪婪解码 |

### 去重过滤配置

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `enabled` | `bool` | `true` | 否 | 是否启用去重过滤器 |
| `similarity_threshold` | `float` | `0.8` | 是 | 相似度阈值（0-1），超过此值的任务会被过滤 |
| `db_path` | `str` | `./.similarity_db` | 否 | 相似度数据库存储路径 |
| `model` | `str` | `text-embedding-v4` | 是 | 用于计算相似度的 embedding 模型 |
| `api_key` | `str` | `null` | 否 | API Key，为 `null` 时从环境变量加载 |

---

## 下一步

<div class="card-grid">
<a href="./data_pipeline/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:database.svg" class="card-icon card-icon-data" alt=""><h3>数据管道</h3></div><p class="card-desc">了解完整的数据加载和处理流程。</p></a>
<a href="./task_judger/" class="feature-card"><div class="card-header"><img src="https://api.iconify.design/mdi:check-decagram.svg" class="card-icon card-icon-general" alt=""><h3>任务评判器</h3></div><p class="card-desc">学习如何评估生成的任务质量。</p></a>
</div>
