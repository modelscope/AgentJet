from typing import Dict, List
from pydantic import BaseModel, Field
from astune import ModelTuner
from astune.schema.task import WorkflowTask, WorkflowOutput


class Workflow(BaseModel):

    model_config = {"extra": "allow"}
    # Workflow
    name: str = "default_workflow"
    # which agents to train, empty means all agents are trainable
    trainable_targets: List[str] = Field(default=[])
    # Use external environment provided by the trainer (True: read environment handle from input; False: AgentScope runs environment and tools)
    require_gym_env: bool = Field(default=True)


    async def agentscope_execute(self, task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        """Run the workflow on a given task."""
        raise NotImplementedError


