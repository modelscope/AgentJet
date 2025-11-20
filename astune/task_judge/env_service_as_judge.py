from astune.task_judge.judge_base import JudgeBase


class EnvServiceJudge(JudgeBase):

    def __init__(self, config):
        self.config = config

    def compute_reward(self, judge_input_dictionary) -> tuple:
        raw_reward = 0

        env = judge_input_dictionary['env']
        workflow_task = judge_input_dictionary['workflow_task']

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