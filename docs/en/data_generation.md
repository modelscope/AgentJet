# Data Generation

## Introduction
`Data Generation Reader` provides an intelligent data generation method designed to simplify the creation of high-quality training data. The method is flexible and efficient, capable of generating domain-specific tasks based on few-shot examples and optional documents.

## Method
`Data Generation Reader` employs a two-stage task generation process:

### Stage 1 (Optional): Document-based Data Generation
This stage is optional. `Document-based Data Generation` generates knowledge-based tasks based on the provided documents. Users can provide one or more documents (supporting formats like PDF, Word, TXT, etc.):

```plain
According to the Anti-Money Laundering and Counter-Terrorist Financing Ordinance and related Guideline, banks are required to identify and take reasonable measures to verify the identity of the beneficial owner of corporate customers so that the bank is ...
```

The generator reads the document content and guides the LLM to batch-generate tasks related to the document content:

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

If documents are provided for data generation, the data generated in this stage will be added to the validation task set for the subsequent training process.

### Stage 2: Few-shot Data Generation
This stage generates the final training tasks. `Few-shot Data Generation` combines a few user-provided tasks with the knowledge-based tasks generated in the first stage, and use the documents as references to generate training tasks. First, the user needs to provide a few task examples:

```json
{"main_query": "Can banks ask corporate customers to provide information of its ownership?", "answer": "According to the Anti-Money Laundering and ..."}
{"main_query": "Can a bank close my account?", "answer": "Either a customer or a bank may close an account at any time subject to any specific terms and ..."}
...
```

These examples will be merged with the tasks generated in the first stage to form an example task set. The generator will sample from this set to be used as few-shot demonstrations, and combined with relevant documents, guide the LLM to batch-generate training tasks:

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

## Quick Start
`Data Generation Reader` can load a few user-provided tasks and optional documents (in various formats such as PDF, Word, and TXT) from a local path, then generates tasks and loads them as training tasks.

### Step 1: Prepare data
Provide a few example tasks:

```json
{"main_query": "What is the capital of France?", "answer": "..."}
{"main_query": "How to cook pasta?", "answer": "..."}
```

(Optional) Provide documents and place them in the specified directory:

```bash
mkdir -p dataset/document
cp your-document.pdf dataset/document/
```

### Step 2: Generate Training Tasks
#### Method 1: Integrate Data Generation into the Training Pipeline
Copy and modify the key configuration parameters in `ajet/default_config/ajet_default.yaml`, and set `ajet.task_reader.type` to `data_generation` to enable this reader.

```yaml
ajet:
  task_reader:
    type: data_generation
    # when `type == data_generation`
    data_generation:
      # Document reader configuration
      document_reader:
        document_path:
          - 'dataset/document/your-document1.pdf'
          - 'dataset/document/your-document2.pdf'
        languages:
          - eng
      # Task reader (for existing tasks)
      query_reader:
        type: jsonl_dataset_file
        jsonl_dataset_file:
          training:
            file_path: 'dataset/jsonl/your-queries.jsonl'
      # Number of tasks to generate
      task_num: 10
      # LLM config
      llm_model: qwen-long
      llm_response_length: 8192
      num_workers: 32
      sampling_params:
        temperature: 0
      # Task filtering config
      deduplication_filter:
        enabled: true
        params:
          similarity_threshold: 0.8
          db_path: ./.similarity_db
          model: text-embedding-v4
          api_key: null # load from the env
          base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```

#### Method 2: Run the Generation Script
```python
from ajet.data_generator.config import *
from ajet.task_reader.data_generator_reader import DataGeneratorTaskReader

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

## **Generated Task Examples**
Based on user-provided documents (optional) and a few task examples, the `Data Generation Reader` can batch-generate training tasks:

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

## Detailed Config Options
| Parameter Path | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `document_reader.document_path` | list[str] | - | No | List of document file paths. Supports PDF, Word, TXT, and more. |
| `document_reader.languages` | list[str] | `['eng']` | No | List of document languages for OCR and text parsing, e.g., `eng` (English), `chs` (Simplified Chinese). |
| `query_reader.type` | str | `jsonl_dataset_file` | Yes | Reader type. Options: `jsonl_dataset_file`, `env_service`, `huggingface_dat_repo`. |
| `query_reader.jsonl_dataset_file.training.file_path` | str | - | Yes | Path to the training tasks JSONL file (when `type: jsonl_dataset_file`). |
| `task_num` | int | `10` | Yes | Number of tasks to generate. The actual number may be reduced by filtering. |
| `llm_model` | str | `qwen-long` | Yes | LLM model name used for task generation. |
| `llm_response_length` | int | `8192` | No | Maximum number of tokens in the LLM response. |
| `num_workers` | int | `32` | No | Number of parallel worker threads for speeding up task generation. |
| `sampling_params.temperature` | float | `0` | No | Sampling temperature. `0` means greedy decoding (deterministic output); higher values make outputs more random. |
| `deduplication_filter.enabled` | bool | `true` | No | Whether to enable the deduplication filter. |
| `deduplication_filter.params.similarity_threshold` | float | `0.8` | Yes | Similarity threshold (0–1). Tasks above this threshold will be filtered out. |
| `deduplication_filter.params.db_path` | str | `./.similarity_db` | No | Path to the similarity database used to cache embeddings. |
| `deduplication_filter.params.model` | str | `text-embedding-v4` | Yes | Embedding model used to compute similarity. |
| `deduplication_filter.params.api_key` | str | `null` | No | API key. If `null`, it will be loaded from the `DASHSCOPE_API_KEY` environment variable. |
| `deduplication_filter.params.base_url` | str | `https://dashscope.aliyuncs.com/compatible-mode/v1` | No | Base URL for the embedding API. |
