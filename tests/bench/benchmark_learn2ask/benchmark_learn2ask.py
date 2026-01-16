# flake8: noqa
import time

from ajet.utils.testing_utils import BenchmarkProbe, singleton

# trinity b.b. expectation
# [TestProbe] Step 50: local average reward over last self.reward_expectation_avg_window steps: 2.6618, expected range: [0.0, 99999.0]
# [TestProbe] Step 100: local average reward over last self.reward_expectation_avg_window steps: 2.8733, expected range: [0.0, 99999.0]
# [TestProbe] Step 200: local average reward over last self.reward_expectation_avg_window steps: 2.9725, expected range: [0.0, 99999.0]

# verl b.b. expectation
# [TestProbe] Step 50: local average reward over last self.reward_expectation_avg_window steps: 3.1562, expected range: [0.0, 99999.0]
# [TestProbe] Step 100: local average reward over last self.reward_expectation_avg_window steps: 3.4732, expected range: [0.0, 99999.0]
# [TestProbe] Step 200: local average reward over last self.reward_expectation_avg_window steps: 3.5645, expected range: [0.0, 99999.0]


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
                50     :       [2.5,  99999.0],
               100     :       [2.7,  99999.0],
               200     :       [2.9,  99999.0],
        }
        # fmt: on
        self.probe_list = ["reward_probe"]
        self.reward_key = "reward_for_test_robot"
        self.probe_key = "reward_probe"

    def __call__(self, key, log_dict):
        return super().__call__(key, log_dict)
