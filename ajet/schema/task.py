from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

"""
The basic schema for task_reader module
"""


class Task(BaseModel):
    main_query: str = Field(default="", description="main query or instruction for the task, maybe absent if the task has valid init_messages.")
    init_messages: List[dict] = Field(default=[], description="initial messages for the task, maybe absent if the task has valid main_query.")
    task_id: str = Field(default="", description="same task_id mean same task, and of course, same GRPO group.")
    env_type: str = Field(default="", description="valid when the task need to interact with a gym env.")
    metadata: dict = Field(default_factory=dict, description="additional metadata for the task, e.g., reference answer for eval tasks.")


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
    log_metrics: Dict[str, Union[float, List[float], Dict[str, Any]]] = Field(default_factory=dict)
