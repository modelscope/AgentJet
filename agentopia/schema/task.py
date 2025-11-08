from pydantic import BaseModel, Field
from typing import List, Dict, Any


class Task(BaseModel):
    task_id: str = Field(default=...)

    env_type: str = Field(default="appworld")

    metadata: dict = Field(default_factory=dict)

    query: List | str = Field(default="")




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

