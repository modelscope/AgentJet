"""Define workflow related interfaces."""

from typing import Dict

from pydantic import BaseModel, Field

from astune.context_manager.tuner import ModelTuner


class WorkflowOutput(BaseModel):
    """Workflow Output Structure."""

    reward: float
    metadata: Dict


class Workflow(BaseModel):
    """Workflow Protocol.

    A workflow defines how to use a model to complete a task.
    """

    name: str = "my_workflow"
    config: Dict = Field(default_factory=dict)

    def run(self, task: Dict, tuner: ModelTuner) -> WorkflowOutput:
        """Run the workflow on a given task."""
        raise NotImplementedError
