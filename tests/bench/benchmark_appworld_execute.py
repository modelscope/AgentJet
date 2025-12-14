import os
import subprocess
import time
import unittest

from beast_logger import print_dict
from loguru import logger

from astuner.utils.dynamic_import import dynamic_import
from astuner.utils.smart_daemon import LaunchCommandWhenAbsent
from astuner.utils.testing_utils import populate_test_env_metadata, send_test_result


class TestBenchmarkAppworld(unittest.TestCase):
    def test_01_begin_verl(self):
        # get probe target, so as to get timeout settings
        BACKBONE = "verl"
        TEST_TARGET = "tests/bench/benchmark_appworld/benchmark_appworld.yaml"
        PROBE_TARGET = "tests/bench/benchmark_appworld/benchmark_appworld.py->TestProbe"
        # tests/bench/benchmark_appworld/benchmark_appworld.py
        # tests/bench/benchmark_appworld/benchmark_appworld.yaml
        TARGET_NAME = f"benchmark_appworld_{BACKBONE}"
        PYTHON_EXECUTABLE = ".verl/bin/python"
        self.execute_benchmark(
            BACKBONE=BACKBONE,
            TEST_TARGET=TEST_TARGET,
            PROBE_TARGET=PROBE_TARGET,
            TARGET_NAME=TARGET_NAME,
            PYTHON_EXECUTABLE=PYTHON_EXECUTABLE,
        )

    def test_02_begin_trinity(self):
        # get probe target, so as to get timeout settings
        BACKBONE = "trinity"
        TEST_TARGET = "tests/bench/benchmark_appworld/benchmark_appworld_2nodes.yaml"
        PROBE_TARGET = "tests/bench/benchmark_appworld/benchmark_appworld.py->TestProbe"
        TARGET_NAME = f"benchmark_appworld_{BACKBONE}"
        PYTHON_EXECUTABLE = ".venv/bin/python"
        self.execute_benchmark(
            BACKBONE=BACKBONE,
            TEST_TARGET=TEST_TARGET,
            PROBE_TARGET=PROBE_TARGET,
            TARGET_NAME=TARGET_NAME,
            PYTHON_EXECUTABLE=PYTHON_EXECUTABLE,
        )

    def execute_benchmark(
        self, BACKBONE, TEST_TARGET, PROBE_TARGET, TARGET_NAME, PYTHON_EXECUTABLE, MULTI_NODES=True
    ):
        cur_dir = os.path.dirname(__file__)
        workspace_dir = os.path.abspath(os.path.join(cur_dir, "../.."))

        git_hash, req_txt = populate_test_env_metadata(workspace_dir)
        os.environ["ASTUNER_GIT_HASH"] = git_hash
        os.environ["ASTUNER_REQ_TXT"] = req_txt
        os.environ["ASTUNER_BENCHMARK_NAME"] = TARGET_NAME

        self.install_appworld()

        send_test_result(
            git_hash=git_hash,
            target=TARGET_NAME,
            status="running",
            status_detail="",  #
            req_txt=req_txt,  # get pip freeze
            append_log="",
            data_dashboard_url="",
            timeout=10.0,
        )
        timeout_seconds = (
            dynamic_import(PROBE_TARGET)().expected_train_time + 600
        )  # add buffer time
        cmd = [
            PYTHON_EXECUTABLE,
            "launcher.py",
            "--conf",
            TEST_TARGET,
            "--backbone",
            BACKBONE,
            "--with-appworld",
            "--autokill",
        ]

        if BACKBONE == "trinity" and (not MULTI_NODES):
            cmd += ["--with-ray"]
        if MULTI_NODES:
            cmd += ["--with-ray-cluster"]

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

    def clear_system_processes(self):
        # kill all python + ray + vllm processes
        from astuner.utils.cleaner import fast_kill_by_keyword_bash

        total_seconds = 15
        for i in range(total_seconds):
            logger.warning(
                f"Warning: To install Appworld, we have kill all `python / VLLM / vllm / ray` processes in your system. IF this is NOT acceptable, TERMINATE NOW! Execute in {total_seconds - i} seconds..."
            )
            time.sleep(1)

        kill = "ray|vllm|VLLM|python"
        for keyword in kill.split("|"):
            logger.info(f"Killing processes matching keyword: {keyword}")
            killed_pids = fast_kill_by_keyword_bash(keyword)
            if killed_pids:
                logger.success(f"Successfully killed processes with PIDs: {killed_pids}")
            else:
                logger.warning(f"No processes found matching keyword: {keyword}")

    def install_appworld(self):
        # run:
        # `rm -rf /tmp/pack_all_in_one & wget https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/astuner_archive/appworld_pack_v2.tar.gz  &&   tar   -xzf   ./appworld_pack_v2.tar.gz  -C /tmp`
        self.clear_system_processes()
        import shutil

        if os.path.exists("/tmp/pack_all_in_one"):
            shutil.rmtree("/tmp/pack_all_in_one")
        if os.path.exists("./appworld_pack_v2.tar.gz"):
            os.remove("./appworld_pack_v2.tar.gz")
        subprocess.run(
            [
                "wget",
                "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/astuner_archive/appworld_pack_v2.tar.gz",
            ]
        )
        subprocess.run(
            [
                "tar",
                "-xzf",
                "./appworld_pack_v2.tar.gz",
                "-C",
                "/tmp",
            ]
        )
        # write
        os.environ["APPWORLD_PATH"] = "/tmp/pack_all_in_one"
        os.environ["APPWORLD_SCRIPT"] = "bash EnvService/env_sandbox/appworld.sh"
