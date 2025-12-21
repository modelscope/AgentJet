import os
import unittest

from beast_logger import print_dict

from astuner.utils.dynamic_import import dynamic_import
from astuner.utils.smart_daemon import LaunchCommandWhenAbsent
from astuner.utils.testing_utils import populate_test_env_metadata, send_test_result


class TestBenchmarkCountdown(unittest.TestCase):
    def test_01_begin_verl(self):
        BACKBONE = "verl"
        TEST_TARGET = "tests/bench/benchmark_countdown/benchmark_countdown.yaml"
        PROBE_TARGET = "tests/bench/benchmark_countdown/benchmark_countdown.py->TestProbe"
        TARGET_NAME = f"benchmark_countdown_{BACKBONE}"
        PYTHON_EXECUTABLE = ".verl/bin/python"
        self.execute_benchmark(
            BACKBONE=BACKBONE,
            TEST_TARGET=TEST_TARGET,
            PROBE_TARGET=PROBE_TARGET,
            TARGET_NAME=TARGET_NAME,
            PYTHON_EXECUTABLE=PYTHON_EXECUTABLE,
        )

    def test_02_begin_trinity(self):
        BACKBONE = "trinity"
        TEST_TARGET = "tests/bench/benchmark_countdown/benchmark_countdown.yaml"
        PROBE_TARGET = "tests/bench/benchmark_countdown/benchmark_countdown.py->TestProbe"
        TARGET_NAME = f"benchmark_countdown_{BACKBONE}"
        PYTHON_EXECUTABLE = ".venv/bin/python"
        self.execute_benchmark(
            BACKBONE=BACKBONE,
            TEST_TARGET=TEST_TARGET,
            PROBE_TARGET=PROBE_TARGET,
            TARGET_NAME=TARGET_NAME,
            PYTHON_EXECUTABLE=PYTHON_EXECUTABLE,
        )

    def execute_benchmark(
        self, BACKBONE, TEST_TARGET, PROBE_TARGET, TARGET_NAME, PYTHON_EXECUTABLE
    ):
        cur_dir = os.path.dirname(__file__)
        workspace_dir = os.path.abspath(os.path.join(cur_dir, "../../.."))

        git_hash, req_txt = populate_test_env_metadata(workspace_dir)
        os.environ["ASTUNER_GIT_HASH"] = git_hash
        os.environ["ASTUNER_REQ_TXT"] = req_txt
        os.environ["ASTUNER_BENCHMARK_NAME"] = TARGET_NAME

        send_test_result(
            git_hash=git_hash,
            target=TARGET_NAME,
            status="running",
            status_detail="",
            req_txt=req_txt,
            append_log="",
            data_dashboard_url="",
            timeout=10.0,
        )
        timeout_seconds = dynamic_import(PROBE_TARGET)().expected_train_time + 600
        cmd = [
            PYTHON_EXECUTABLE,
            "-m",
            "astuner.cli.launcher",
            "--conf",
            TEST_TARGET,
            "--backbone",
            BACKBONE,
            "--autokill",
        ]
        if BACKBONE == "trinity":
            cmd += ["--with-ray"]
        companion = LaunchCommandWhenAbsent(
            full_argument_list=cmd,
            dir=workspace_dir,
            tag=TARGET_NAME,
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
                "TestTarget": TEST_TARGET,
                "TestSuccessful": test_successful,
            }
        )
