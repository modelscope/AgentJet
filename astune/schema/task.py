from pydantic import BaseModel, Field
from typing import List, Dict, Any


class Task(BaseModel):
    main_query: str = Field(default="")
    init_messages: List[dict] = Field(default=[])
    task_id: str = Field(default="")
    env_type: str = Field(default="")
    metadata: dict = Field(default_factory=dict)


class TaskLaunchCoreArgument(BaseModel):
    env_type: str = Field(default="")
    task_id: str = Field(default="")
    task_thread_index: int = Field(default=0)
    task_batch_index: int = Field(default=0)
    task_tag: str = Field(default="")
    task_env_uuid: str = Field(default="")
    obs_window: dict = Field(default={})
    llm_chat_fn: Any = Field(default=None)
    tokenizer: Any = Field(default=None)
    task: Task = Field(default=None)    # type: ignore

