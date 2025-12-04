import os
import sys
import unittest

from beast_logger import print_dict

from astuner.utils.dynamic_import import dynamic_import
from astuner.utils.smart_daemon import LaunchCommandWhenAbsent


class TestBenchmarkMath(unittest.TestCase):
    def test_begin(self):
        # get probe target, so as to get timeout settings
        TEST_TARGET = "tests/bench/benchmark_math/benchmark_math.yaml"
        PROBE_TARGET = "tests/bench/benchmark_math/benchmark_math.py->TestProbe"
        TARGET_NAME = "benchmark_math"
        timeout_seconds = (
            dynamic_import(PROBE_TARGET)().expected_train_time + 600
        )  # add buffer time
        cmd = [
            sys.executable,
            "launcher.py",
            "--conf",
            TEST_TARGET,
            "--backbone",
            "trinity",
            "--with-ray",
            "--autokill",
        ]

        cur_dir = os.path.dirname(__file__)
        workspace_dir = os.path.abspath(os.path.join(cur_dir, "../.."))
        companion = LaunchCommandWhenAbsent(
            full_argument_list=cmd,
            dir=workspace_dir,
            tag=TARGET_NAME,
        )

        test_successful = False
        terminate_str = companion.launch(
            launch_wait_time=timeout_seconds,
            success_std_string=[
                "GoodbyeException",
                "TestFailException",
                "You can force stop the `Trainer` process by pressing Ctrl+C",
            ],
            env_dict=os.environ,
            force_restart=True,
        )
        test_successful = True
        companion.kill_self()
        if terminate_str == "TestFailException":
            test_successful = False
            raise RuntimeError("Benchmark math test failed during execution.")
        if terminate_str == "You can force stop the `Trainer` process by pressing Ctrl+C":
            test_successful = False
            raise RuntimeError("Unknown trinity exception.")
        if terminate_str == "GoodbyeException":
            test_successful = True

        print_dict(
            {
                "TestTarget": TEST_TARGET,
                "TestSuccessful": test_successful,
            }
        )
