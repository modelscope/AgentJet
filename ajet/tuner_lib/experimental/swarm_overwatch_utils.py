from typing import List, Dict
from pydantic import BaseModel


class CurrentBatchRolloutPoolInformation(BaseModel):
    sample_collection_method: str = ""
    completed_episodes: int = 0
    completed_episode_target: int = 0
    completed_tasks: int = 0
    completed_task_target: int = 0
    completed_non_dummy_tasks: int = 0
    completed_non_dummy_task_target: int = 0
    task_expected_num_repeat: int = 0
    completed_tasks_details: Dict[str, List[str]] = {}  # task_id -> list of episode_uuids
    completed_tasks_rewards: Dict[str, List[float]] = {}  # task_id -> list of rewards (one per episode)
    running_episode_details: Dict[str, Dict[str, str]] | None = None # episode_uuid -> { "episode_status": ..., "time_since_last_activity": ..., "discard_episode_timeout": ..., "llm_call_count": ...}
    engine_status: str | None = None
    global_step: int | None = None
    booting_start_time: float | None = None  # timestamp when ENGINE.BOOTING started
