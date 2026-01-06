# flake8: noqa E261, E131, E241

import os
import subprocess
import sys
import time
from typing import List

import requests
from beast_logger import print_dict
from loguru import logger

from ajet.utils.dynamic_import import dynamic_import


class TestSuccessException(Exception):
    """
    All test is done, end the program early with exception.
    """

    pass


class TestFailException(Exception):
    """
    Test has failed, end the program early with exception.
    """

    pass


class BaseProbe(object):
    """
    The basic test probe class, capture keyword if matched `self.probe_list`, and do test.
    """

    def __init__(self):
        self.probe_list: List[str] = []

    def __call__(self, key: str, log_dict: dict):
        raise NotImplementedError

    def mock(self, key: str):
        raise NotImplementedError


def get_test_lambda(test_name) -> BaseProbe:
    test_cls = dynamic_import(test_name)()
    return test_cls


def _test_if_test_mode(key, value, config):
    from ajet.backbone.warm_up import init_parallel_rollout_logger

    if not config.ajet.execute_test:
        return
    if config.ajet.execute_test == "do_not_test":
        return
    init_parallel_rollout_logger(config.ajet.experiment_name)
    test_lambda = get_test_lambda(config.ajet.execute_testing_lambda)
    if key not in test_lambda.probe_list:
        return
    return test_lambda(key, value)


def _mock_if_test_mode(key, value, config):
    if not config.ajet.execute_test:
        return value
    if config.ajet.execute_test == "do_not_test":
        return value
    test_lambda = get_test_lambda(config.ajet.execute_testing_lambda)
    if key not in test_lambda.probe_list:
        return value
    return test_lambda.mock(key)


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def send_test_result(
    git_hash: str,
    target: str,
    status: str,
    status_detail: str = "",
    req_txt: str | None = None,
    append_log: str | None = None,
    data_dashboard_url: str | None = None,
    timeout: float = 10.0,
) -> dict:
    """
    Post a single experiment result to the /report_test_result endpoint.
    Raises requests.HTTPError on non-2xx responses.
    """
    payload = {
        "git_hash": git_hash,
        "target": target,
        "status": status,
        "status_detail": status_detail,
        "req_txt": req_txt or "",
        "append_log": append_log or "",
        "data_dashboard_url": data_dashboard_url or "",
    }
    resp = requests.post(
        r"https://benchmark-report.agent-matrix.com/report_test_result",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def populate_test_env_metadata(workspace_dir: str) -> tuple[str, str]:
    """Capture git hash and pip freeze output, store them in env, return both."""
    git_hash = "unknown"
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace_dir)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    req_txt = ""
    try:
        req_txt = (
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], cwd=workspace_dir)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return git_hash, req_txt


def update_benchmark_status(status, status_detail, append_log="", data_dashboard_url=""):
    if "ASTUNER_GIT_HASH" not in os.environ:
        raise RuntimeError(
            "ASTUNER_GIT_HASH not found in environment variables. Please set `ajet.execute_test=False`."
        )

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


class BenchmarkProbe(BaseProbe):
    """
    A benchmark probe to test reward during training.
    Major module input:
        - self.reward_expectation: dict, key is step, value is [low, high] expected reward range
        - self.reward_expectation_avg_window: int, number of steps to average reward over
        - self.expected_train_time: int, expected training time in seconds
    """

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

        self.reward_key = "reward_for_test_robot"
        self.probe_key = "reward_probe"

    def __call__(self, key, log_dict):
        reward = self.reward_key
        if key == self.probe_key:
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
                # compute local average reward
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
                    # test failed
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
                    # test passed
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
                    raise TestSuccessException(msg)
