from astuner.workflow import WorkflowOutput, WorkflowTask


class JudgeBase:
    def __init__(self, config):
        self.config = config

    def compute_reward(self, workflow_task: WorkflowTask, workflow_output: WorkflowOutput) -> tuple:
        raise NotImplementedError
