import subprocess
import sys

import requests

from astuner.backbone.common_warm_up import init_parallel_rollout_logger
from astuner.utils.dynamic_import import dynamic_import


def get_test_lambda(test_name):
    test_cls = dynamic_import(test_name)()
    return test_cls


def _test_if_test_mode(key, value, config):
    if not config.astuner.execute_test:
        return
    if config.astuner.execute_test == "do_not_test":
        return
    init_parallel_rollout_logger(config.astuner.experiment_name)
    test_lambda = get_test_lambda(config.astuner.execute_testing_lambda)
    return test_lambda(key, value)


def _mock_if_test_mode(key, value, config):
    if not config.astuner.execute_test:
        return value
    if config.astuner.execute_test == "do_not_test":
        return value
    test_lambda = get_test_lambda(config.astuner.execute_testing_lambda)
    return test_lambda.mock(key)


class GoodbyeException(Exception):
    pass


class TestFailException(Exception):
    pass


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
