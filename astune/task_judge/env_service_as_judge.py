from astune.task_judge.judge_base import JudgeBase
from astune.workflow import WorkflowOutput, WorkflowTask


class EnvServiceJudge(JudgeBase):

    def __init__(self, config):
        self.config = config

    def compute_reward(self, workflow_task: WorkflowTask, workflow_output: WorkflowOutput) -> tuple:
        raw_reward = 0

        env = workflow_task.gym_env
        raw_reward = env.evaluate(workflow_task.task_env_uuid, params={"sparse": False})
        if raw_reward >= 1:
            is_success = True
        else:
            is_success = False

        if self.config.astune.rollout.add_special_success_reward:
            if is_success:
                raw_reward = 1.0 + raw_reward * 0.5
            else:
                raw_reward = 0.0 + raw_reward * 0.5

        if self.config.astune.rollout.binary_reward:
            raw_reward = 1.0 if is_success else 0.0

        return raw_reward, is_success
