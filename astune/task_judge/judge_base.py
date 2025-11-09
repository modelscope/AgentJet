class JudgeBase():

    def __init__(self, config):
        self.config = config

    def compute_reward(self, judge_input_dictionary) -> tuple:
        # judge_input_dictionary['env']: env_service 外部环境 （如果使用了env_service）
        # judge_input_dictionary['task_core_arg']: 任务信息（如果里面包含了参考答案，可以从中取出）
        # judge_input_dictionary['grouped_steps']: LLM的每一次历史对话记录（如果中间过程比较重要，可以从中取出）

        raise NotImplementedError