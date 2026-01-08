from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

"""
The basic schema for task_reader module
"""


class Task(BaseModel):
    main_query: str = Field(default="")
    init_messages: List[dict] = Field(default=[])
    task_id: str = Field(default="")
    env_type: str = Field(default="")
    metadata: dict = Field(default_factory=dict)


"""
For workflow execution, include task uuid and gym client if needed
"""


class WorkflowTask(BaseModel):
    env_type: str = Field(default="")
    task_id: str = Field(default="")
    task_thread_index: int = Field(default=0)
    task_batch_index: int = Field(default=0)
    task_tag: str = Field(default="")
    episode_uuid: str = Field(default="")
    observation_window: dict = Field(default={})
    llm_inference_fn: Any = Field(default=None)
    tokenizer: Any = Field(default=None)
    task: Task = Field(default=Task())
    gym_env: Any = Field(default=None)  # agentscope runtime handle or env service handle


"""
workflow output, user should provide as workflow output
"""


class WorkflowOutput(BaseModel):
    reward: Union[float, List[float], None] = Field(default=None)
    is_success: Union[bool, None] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
