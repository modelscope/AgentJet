# flake8: noqa
import os
import time

from beast_logger import print_dict
from loguru import logger

from astuner.utils.testing_utils import (
    BaseProbe,
    GoodbyeException,
    TestFailException,
    send_test_result,
    singleton,
)


def update_benchmark_status(status, status_detail, append_log="", data_dashboard_url=""):
    git_hash = os.environ["ASTUNER_GIT_HASH"]
    req_txt = os.environ["ASTUNER_REQ_TXT"]
    target_name = os.environ["ASTUNER_BENCHMARK_NAME"]

    if not append_log:
        append_log = status_detail

    send_test_result(
        git_hash=git_hash,
        target=target_name,
        status=status,
        status_detail=status_detail,  #
        req_txt=req_txt,  # get pip freeze
        append_log=append_log,
        data_dashboard_url=data_dashboard_url,
        timeout=10.0,
    )


@singleton
class TestProbe(BaseProbe):
    def __init__(self):
        # fmt: off
        self.expected_train_time = 3600 * 24 # 24 hours
        self.begin_time = time.time()
        self.reward_array = []
        self.reward_expectation_avg_window = 5
        self.reward_expectation = {
            # step    : expected local average reward range
            # step    :       [low,    high ]
                5     :       [0.10,  99999.0],
               10     :       [0.45,  99999.0],
               20     :       [0.68,  99999.0],
               30     :       [0.85,  99999.0],
        }
        # fmt: on
        self.probe_list = ["reward_probe"]

    def __call__(self, key, log_dict):
        reward = "reward_for_test_robot"
        if key == "reward_probe":
            step = log_dict["step"]

            if time.time() - self.begin_time > self.expected_train_time:
                msg = (
                    f"Training time exceeded expected limit of {self.expected_train_time} seconds."
                )
                update_benchmark_status(
                    status="fail",
                    status_detail=msg,
                    append_log=msg,
                )
                raise TestFailException(msg)

            # if new data, add
            logger.bind(benchmark=True).info(f"log_dict: {str(log_dict)}")
            logger.bind(benchmark=True).info(f"reward_key: {str(reward)}")
            logger.bind(benchmark=True).info(f"self.reward_array before: {str(self.reward_array)}")
            if reward in log_dict:
                reward = log_dict[reward]
                self.reward_array += [reward]

            update_benchmark_status(
                status="running",
                status_detail=f"Current step: {step}",
                append_log=f"Step {step}: reward logged, {str(self.reward_array)}.",
                data_dashboard_url=log_dict["data_dashboard_url"],
            )

            # begin test
            if step in self.reward_expectation:
                if len(self.reward_array) == 0:
                    err = f"No reward logged at step {step}"
                    update_benchmark_status(
                        status="fail",
                        status_detail=err,
                    )
                    raise TestFailException(err)
                # compute local average reward over last self.reward_expectation_avg_window steps
                local_avg_reward = sum(
                    self.reward_array[-self.reward_expectation_avg_window :]
                ) / min(self.reward_expectation_avg_window, len(self.reward_array))
                # get expected range
                low, high = self.reward_expectation[step]
                # log
                msg = f"[TestProbe] Step {step}: local average reward over last self.reward_expectation_avg_window steps: {local_avg_reward:.4f}, expected range: [{low}, {high}]"
                logger.bind(benchmark=True).info(msg)
                update_benchmark_status(
                    status="running",
                    status_detail=msg,
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

                    err = f"[TestProbe] Reward test failed at step {step}: local average reward {local_avg_reward:.4f} not in expected range [{low}, {high}]"
                    logger.bind(benchmark=True).error(err)
                    update_benchmark_status(
                        status="fail",
                        status_detail=err,
                    )
                    raise TestFailException(err)
                else:
                    msg = f"[TestProbe] Reward test passed at step {step}."
                    logger.bind(benchmark=True).info(msg)
                    update_benchmark_status(status="running", status_detail=msg)
                # congrats, all tests passed, let's crash and escape this test early.
                if step == max(self.reward_expectation.keys()):
                    msg = "[TestProbe] All reward tests passed. Exiting training early."
                    logger.bind(benchmark=True).info(msg)
                    update_benchmark_status(
                        status="successful",
                        status_detail=msg,
                    )
                    raise GoodbyeException(msg)
