from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SamplingParamsConfig(BaseModel):
    """Sampling parameters configuration"""

    temperature: float = Field(
        default=0.0, description="Sampling temperature, 0 means greedy decoding"
    )


class DeduplicationFilterParamsConfig(BaseModel):
    """Deduplication filter parameters configuration"""

    similarity_threshold: float = Field(
        default=0.8, description="Similarity threshold. Tasks above this value will be filtered."
    )
    db_path: str = Field(
        default="./.similarity_db", description="Storage path for the similarity database"
    )
    model: str = Field(default="text-embedding-v4", description="Embedding model name")
    api_key: Optional[str] = Field(
        default=None, description="API Key. If None, it is loaded from environment variables."
    )
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="Embedding API base URL",
    )


class TrainingDatasetConfig(BaseModel):
    """Training dataset configuration"""

    file_path: str = Field(default="", description="Path to the training data file")


class DeduplicationFilterConfig(BaseModel):
    """Deduplication filter configuration"""

    enabled: bool = Field(default=True, description="Whether to enable the filter")
    params: DeduplicationFilterParamsConfig = Field(default_factory=DeduplicationFilterParamsConfig)


class DocumentReaderConfig(BaseModel):
    """Document reader configuration"""

    document_path: List[str] = Field(default=[], description="List of document file paths")
    languages: List[str] = Field(default=["eng"], description="List of document languages")
    chunk_size: int = Field(default=5120, description="Chunk size")
    split_by: str = Field(
        default="sentence", description="Split method: sentence, paragraph, character"
    )
    cache_enabled: bool = Field(default=True, description="Whether to enable caching")

    @field_validator("split_by")
    @classmethod
    def validate_split_by(cls, v: str) -> str:
        allowed = ["sentence", "paragraph", "character"]
        if v not in allowed:
            raise ValueError(f"split_by must be one of {allowed}, current value: {v}")
        return v


class DatasetFileConfig(BaseModel):
    """Dataset file configuration"""

    training: TrainingDatasetConfig = Field(default_factory=TrainingDatasetConfig)


class QueryReaderConfig(BaseModel):
    """Query reader configuration"""

    type: str = Field(default="jsonl_dataset_file", description="Reader type")
    jsonl_dataset_file: DatasetFileConfig = Field(default_factory=DatasetFileConfig)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = ["jsonl_dataset_file", "env_service", "huggingface_dat_repo"]
        if v not in allowed:
            raise ValueError(f"type must be one of {allowed}, current value: {v}")
        return v


class DataGenerationConfig(BaseModel):
    """Data generation configuration"""

    document_reader: DocumentReaderConfig = Field(default_factory=DocumentReaderConfig)
    query_reader: QueryReaderConfig = Field(default_factory=QueryReaderConfig)
    task_num: int = Field(default=10, description="Number of tasks to generate")
    llm_model: str = Field(default="qwen-long", description="LLM model name")
    llm_response_length: int = Field(default=8192, description="LLM maximum response length")
    num_workers: int = Field(default=32, description="Number of parallel worker threads")
    sampling_params: SamplingParamsConfig = Field(default_factory=SamplingParamsConfig)
    deduplication_filter: DeduplicationFilterConfig = Field(
        default_factory=DeduplicationFilterConfig
    )


class TaskReaderConfig(BaseModel):
    """Task reader configuration"""

    data_generation: DataGenerationConfig = Field(default_factory=DataGenerationConfig)
