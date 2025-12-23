# Data Generation

## Introduction
**Data Generation** is a smart data augmentation framework designed to simplify the generation of high-quality training data. Flexible and efficient by design, the framework provides two powerful methods to generate domain-specific queries:

+ **Document-based data generation**: Automatically extracts knowledge from documents (PDF, Word, text files) to generate context-relevant queries
+ **Few-shot data generation**: Uses existing queries as references to generate new queries with consistent style and semantic similarity

## Few-shot Data Generation
### Overview
Few-shot Data Generation is a module based on few-shot learning that helps you **automatically generate new queries**:

+ Provide some existing queries as reference examples
+ Optionally provide a document as background knowledge
+ The module leverages LLMs to generate new queries with similar style and related semantics

### Architecture
The module consists of three main components:

1. **TaskReader**: Parses the user-provided queries
2. **DocReader**: Parses documents (PDF, TXT, Word, etc.) and provides smart caching
3. **TaskAugmentation**: Generates new tasks from the user-provided queries and optional document content

### Features
**Task augmentation**

+ ✅ **Smart Imitation**: Generates new queries consistent in style and semantically related to the references.
+ ✅ **Document Knowledge Integration**: Optionally integrates document context to generate high-quality queries that align with the document's theme.
+ ✅ **Traceability**: Automatically records source information for each generated query.

### 🚀 Quick Start
#### Step 1: Prepare data
**Prepare reference query data:**

```json
{"main_query": "What is the capital of France?", "answer": "..."}
{"main_query": "How to cook pasta?", "answer": "..."}
```

**Prepare document data** (optional), place your document in the specified directory:

```bash
mkdir -p dataset/document
cp your-document.pdf dataset/document/
```

#### Step 2: Create a configuration file
Create a `.yaml` config file. Example (`tests/data_gen.yaml`):

```yaml
astuner:
  data_generation:
    # (Optional) Configure how to read the document data
  document_reader:
    document_path: 'dataset/document/your-document.pdf'
      languages:
      - eng
    # Configure how to read the reference query file
    query_reader:
      type: dataset_file # read from a local file
      dataset_file:
        training:
          file_path: 'dataset/jsonl/your-queries.jsonl'  # path to the reference query data

    # Configure the LLM for generation
    llm_model: qwen-long # Model, e.g., qwen-long
    llm_response_length: 8192
    sampling_params:
      temperature: 0
```

#### Step 3: Run the generation script
**Option A: Use the test script**

```bash
cd /path/to/astuner
export DASHSCOPE_API_KEY='sk-xxxxxx|sk-yyyyyy'
export DASHSCOPE_API_KEY_BACKUP='sk-zzzzzz'
python tests/data_gen.py
```

**Option B: Custom script**

```python
# generate_tasks.py
import sys
sys.path.insert(0, '/path/to/astuner')
import dotenv
dotenv.load_dotenv()

from astuner.utils.config_utils import read_astuner_config
from astuner.task_reader import TaskReaderRouterV2
from astuner.task_reader.document_reader.doc_reader import DocReader
from astuner.data_generator.task_augmentation import TaskAugmentor

# Load config
config = read_astuner_config('tests/data_gen.yaml')

# Initialize
task_reader = TaskReaderRouterV2(
    reader_type=config.task_reader.data_generation.query_reader.type,
    reader_config=config.task_reader.data_generation.query_reader
)
document_reader = DocReader(config)
task_augmentor = TaskAugmentor(config)

# Load data
original_tasks = task_reader.get_training_tasks()
document = document_reader.get_document()
print(f"Reference query: {len(original_tasks)}.")
print(f"Document loaded: {len(document.content)} characters.\n")

# Generate new queries
new_tasks = []
for task in original_tasks[:5]: # Test with 5 queries first, using one query as reference
    new_task = task_augmentor.generate_task(
        source_task=task,
        document=document
    )
    new_tasks.append(new_task)

print(f"Generated {len(new_tasks)} queries.")
for i, task in enumerate(new_tasks):
    print(f"{i+1}. {task.main_query}")

```

### Sample Output
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


## Document-based Data Generation
### Overview
**Document-based Data Generation** automatically produces high-quality training tasks based on documents. Leveraging the knowledge augmentation capability of LLMs, this module generates new queries along with their corresponding context.

### Architecture
The module consists of two main components:

1. **DocReader**: Parses documents (PDF, TXT, Word, etc.) and provides smart caching.
2. **KnowledgeAugmentor**: Generates new queries from the document content.

### Features
**Knowledge augmentation**

+ ✅ **Comprehensive Coverage**: Extracts factual, conceptual, analytical, and applicational queries from the document.
+ ✅ **Context Alignment**: Each generated query includes a related document context to ensure answer traceability.
+ ✅ **Configurable Output**: You can specify how many queries to generate (currently supports N < 10; if N > 10, please run in batches).

### 🚀 Quick Start
#### Step 1: Prepare data
Place your document in the specified directory:

```bash
mkdir -p dataset/document
cp your-document.pdf dataset/document/
```

#### Step 2: Create a configuration file
Create a `.yaml` config file. Example (`tests/data_gen.yaml`):

```yaml
# tests/data_gen.yaml
astuner:
  data_generation:
    document_reader:
      document_path: 'dataset/document/your-document.pdf'
      languages: ['eng']
    cache_enabled: true
    llm_model: qwen-long
    knowledge_augmentor:
      n: 10  # generate 10 queries
```

#### Step 3: Run the Generation Script
**Option A: Use the test script**

```bash
cd /path/to/astuner
export DASHSCOPE_API_KEY='sk-xxxxxx|sk-yyyyyy'
export DASHSCOPE_API_KEY_BACKUP='sk-zzzzzz'
python tests/data_gen.py
```

**Option B: Custom script**

```python
import sys
sys.path.insert(0, '/path/to/astuner')
import dotenv
dotenv.load_dotenv()

from astuner.utils.config_utils import read_astuner_config
from astuner.task_reader.document_reader.doc_reader import DocReader
from astuner.data_generator.knowledge_augmentation import KnowledgeAugmentor

# Load config
config = read_astuner_config('tests/data_gen.yaml')

# Initialize
document_reader = DocReader(config)
knowledge_augmentor = KnowledgeAugmentor(config)

# Load document (with caching)
document = document_reader.get_document()
print(f"Document loaded: {len(document.content)} characters.")

# Generate knowledge-based queries
generated_tasks = knowledge_augmentor.generate_task(
    document=document
)

print(f"Generated {len(generated_tasks)} queries.")
for i, task in enumerate(generated_tasks[:3]):
    print(f"{i+1}. {task.main_query}")
```

### Sample Output
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


## Detailed Config
### Basic Config (`data_gen.yaml`)
```yaml
astuner:
  # Data generator configuration
  data_generation:
  # Document reader configuration
    document_reader:
      document_path: 'dataset/document/your-document.pdf'
      languages:
        - eng
    cache_enabled: true
    cache_format: json
    # Task reader (for existing queries)
    query_reader:
      type: dataset_file
      dataset_file:
        training:
          file_path: 'dataset/jsonl/your-tasks.jsonl'

    # LLM configuration
    llm_model: qwen-long
    llm_response_length: 8192
    sampling_params:
      temperature: 0

    # Knowledge augmentation settings
    knowledge_augmentor:
      n: 10    # number of queries generated from the document
```

### Config Options
#### Document Reader Options
| **Option** | **Type** | **Default** | **Description** |
| --- | --- | --- | --- |
| `document_path` | string | Required | Path to the source document |
| `languages` | list | `['eng']` | Languages for document parsing |
| `cache_enabled` | boolean | `true` | Enable/disable caching |


#### Knowledge Augmentation Options
| **Option** | **Type** | **Default** | **Description** |
| --- | --- | --- | --- |
| `n` | integer | `10` | Number of queries to generate |
| `llm_model` | string | Required | LLM model for generation |
| `llm_response_length` | integer | `8192` | Maximum response length |
| `sampling_params` | dict | `{}` | LLM sampling parameters |
