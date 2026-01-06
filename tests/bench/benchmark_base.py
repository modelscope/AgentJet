import os
import unittest
from pathlib import Path
from typing import Callable, List, Optional

from beast_logger import print_dict

from ajet.utils.dynamic_import import dynamic_import
from ajet.utils.smart_daemon import LaunchCommandWhenAbsent
from ajet.utils.testing_utils import (
    populate_test_env_metadata,
    send_test_result,
)


class BenchmarkTestCase(unittest.TestCase):
    def execute_benchmark(
        self,
        *,
        backbone: str,
        test_target: str,
        probe_target: str,
        target_name: str,
        python_executable: str,
        extra_cmd_args: Optional[List[str]] = None,
        pre_launch: Optional[Callable[[], None]] = None,
        use_ray_cluster: bool = False,
        enable_ray_for_trinity: bool = True,
    ) -> None:
        """Run a benchmark with shared boilerplate for setup and process management."""
        workspace_dir = Path(__file__).resolve().parents[2]

        git_hash, req_txt = populate_test_env_metadata(str(workspace_dir))
        os.environ["ASTUNER_GIT_HASH"] = git_hash
        os.environ["ASTUNER_REQ_TXT"] = req_txt
        os.environ["ASTUNER_BENCHMARK_NAME"] = target_name

        if pre_launch:
            pre_launch()

        send_test_result(
            git_hash=git_hash,
            target=target_name,
            status="running",
            status_detail="",
            req_txt=req_txt,
            append_log="",
            data_dashboard_url="",
            timeout=10.0,
        )

        timeout_seconds = dynamic_import(probe_target)().expected_train_time + 600

        cmd = [
            python_executable,
            "-m",
            "ajet.cli.launcher",
            "--conf",
            test_target,
            "--backbone",
            backbone,
            "--autokill",
        ]
        if extra_cmd_args:
            cmd += extra_cmd_args
        if use_ray_cluster:
            cmd += ["--with-ray-cluster"]
        elif enable_ray_for_trinity and backbone == "trinity":
            cmd += ["--with-ray"]

        companion = LaunchCommandWhenAbsent(
            full_argument_list=cmd,
            dir=str(workspace_dir),
            tag=target_name,
        )

        test_successful = False
        terminate_str = companion.launch(
            launch_wait_time=timeout_seconds,
            success_std_string=[
                "TestSuccessException",
                "TestFailException",
                "You can force stop the `Trainer` process by pressing Ctrl+C",
                "torch.OutOfMemoryError: CUDA out of memory",
            ],
            env_dict=os.environ,
            force_restart=True,
        )
        test_successful = True
        companion.kill_self()
        if terminate_str == "TestSuccessException":
            test_successful = True
        elif terminate_str == "TestFailException":
            test_successful = False
            raise RuntimeError("Benchmark test failed during execution.")
        elif terminate_str == "You can force stop the `Trainer` process by pressing Ctrl+C":
            test_successful = False
            raise RuntimeError("Unknown trinity exception.")
        else:
            test_successful = False
            raise RuntimeError(f"Benchmark test timed out or crashed. {test_successful}")

        print_dict(
            {
                "TestTarget": test_target,
                "TestSuccessful": test_successful,
            }
        )
