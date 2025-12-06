# flake8: noqa
import time

from beast_logger import print_dict
from loguru import logger

from astuner.utils.testing_utils import GoodbyeException, TestFailException, singleton


@singleton
class TestProbe(object):
    def __init__(self):
        # fmt: off
        self.expected_train_time = 3600 * 24 # 24 hours
        self.begin_time = time.time()
        self.reward_array = []
        self.reward_expectation_avg_window = 5
        self.reward_expectation = {
            # step    : expected local average reward range
            #         :       [low,    high ]
               5     :        [0.90,  99999.0],
            #  5     :        [0.50,  99999.0],
               10     :       [0.45,  99999.0],
               20     :       [0.68,  99999.0],
               30     :       [0.85,  99999.0],
        }
        # fmt: on

    def __call__(self, key, log_dict):
        explore_reward_key = "experience_pipeline/group_advantages/reward_mean/mean"
        trainer_reward_key = "critic/score/mean"
        if key == "reward_probe":
            step = log_dict["step"]

            if time.time() - self.begin_time > self.expected_train_time:
                raise TestFailException(
                    f"Training time exceeded expected limit of {self.expected_train_time} seconds."
                )

            # if new data, add
            logger.bind(benchmark=True).info(f"log_dict: {str(log_dict)}")
            logger.bind(benchmark=True).info(f"reward_key: {str(explore_reward_key)}")
            logger.bind(benchmark=True).info(f"self.reward_array before: {str(self.reward_array)}")
            if explore_reward_key in log_dict:
                reward = log_dict[explore_reward_key]
                self.reward_array += [reward]
            if trainer_reward_key in log_dict:
                return  # ignore trainer, only focus on explorer

            # begin test
            if step in self.reward_expectation:
                if len(self.reward_array) == 0:
                    raise TestFailException(f"No reward logged at step {step}")
                # compute local average reward over last self.reward_expectation_avg_window steps
                local_avg_reward = sum(
                    self.reward_array[-self.reward_expectation_avg_window :]
                ) / min(self.reward_expectation_avg_window, len(self.reward_array))
                # get expected range
                low, high = self.reward_expectation[step]
                # log
                logger.bind(benchmark=True).info(
                    f"[TestProbe] Step {step}: local average reward over last self.reward_expectation_avg_window steps: {local_avg_reward:.4f}, expected range: [{low}, {high}]"
                )
                # check
                if not (low <= local_avg_reward <= high):
                    print_dict(
                        {
                            "step": step,
                            "local_avg_reward": local_avg_reward,
                            "expected_low": low,
                            "expected_high": high,
                        },
                        mod="benchmark",
                    )
                    logger.bind(benchmark=True).error(
                        f"[TestProbe] Reward test failed at step {step}: local average reward {local_avg_reward:.4f} not in expected range [{low}, {high}]"
                    )
                    raise TestFailException(
                        f"Reward test failed at step {step}: local average reward {local_avg_reward:.4f} not in expected range [{low}, {high}]"
                    )
                else:
                    logger.bind(benchmark=True).info(
                        f"[TestProbe] Reward test passed at step {step}."
                    )
                # congrats, all tests passed, let's crash and escape this test early.
                if step == max(self.reward_expectation.keys()):
                    logger.bind(benchmark=True).info(
                        f"[TestProbe] All reward tests passed. Exiting training early."
                    )
                    raise GoodbyeException("All reward tests passed. Exiting training early.")

        else:
            logger.bind(benchmark=True).error(f"Unrecognized test key: {key}")
            raise TestFailException(f"Unrecognized test key: {key}")
