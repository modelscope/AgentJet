# flake8: noqa
import time

from agentscope_tuner.utils.testing_utils import BenchmarkProbe, singleton


@singleton
class TestProbe(BenchmarkProbe):
    def __init__(self):
        # fmt: off
        self.expected_train_time = 3600 * 24 # 24 hours
        self.begin_time = time.time()
        self.reward_array = []
        self.reward_expectation_avg_window = 20
        self.reward_expectation = {
            # step    : expected local average reward range
            # step    :       [low,    high ]
                50     :       [0.0,  99999.0],
               100     :       [0.0,  99999.0],
               200     :       [0.0,  99999.0],
        }
        # fmt: on
        self.probe_list = ["reward_probe"]
        self.reward_key = "reward_for_test_robot"
        self.probe_key = "reward_probe"

    def __call__(self, key, log_dict):
        return super().__call__(key, log_dict)
