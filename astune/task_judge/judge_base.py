class JudgeBase():

    def __init__(self, config):
        self.config = config

    def compute_reward(self, judge_input_dictionary) -> tuple:
        raise NotImplementedError