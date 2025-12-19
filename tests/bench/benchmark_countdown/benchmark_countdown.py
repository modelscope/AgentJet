# flake8: noqa
import time

from astuner.utils.testing_utils import BenchmarkProbe, singleton


@singleton
class TestProbe(BenchmarkProbe):
    def __init__(self):
        # fmt: off
        self.expected_train_time = 3600 * 12  # 12 hours budget for countdown benchmark
        self.begin_time = time.time()
        self.reward_array = []
        self.reward_expectation_avg_window = 30
        self.reward_expectation = {
            # step    : expected local average reward range
            # step    :       [low,    high ]
               30     :       [0.30,  99999.0],
               60     :       [0.40,  99999.0],
               90     :       [0.45,  99999.0],
              120     :       [0.50,  99999.0],
              150     :       [0.55,  99999.0],
        }
        # fmt: on
        self.probe_list = ["reward_probe"]
        self.reward_key = "reward_for_test_robot"
        self.probe_key = "reward_probe"

    def __call__(self, key, log_dict):
        return super().__call__(key, log_dict)
