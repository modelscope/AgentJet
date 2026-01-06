import os
import subprocess
import time

from loguru import logger

from tests.bench.benchmark_base import BenchmarkTestCase


class TestBenchmarkAppworld(BenchmarkTestCase):
    def test_01_begin_verl(self):
        # get probe target, so as to get timeout settings
        BACKBONE = "verl"
        TEST_TARGET = "tests/bench/benchmark_appworld/benchmark_appworld.yaml"
        PROBE_TARGET = "tests/bench/benchmark_appworld/benchmark_appworld.py->TestProbe"
        # tests/bench/benchmark_appworld/benchmark_appworld.py
        # tests/bench/benchmark_appworld/benchmark_appworld.yaml
        TARGET_NAME = f"benchmark_appworld_{BACKBONE}"
        PYTHON_EXECUTABLE = ".verl/bin/python"
        multi_nodes = True

        self.execute_benchmark(
            backbone=BACKBONE,
            test_target=TEST_TARGET,
            probe_target=PROBE_TARGET,
            target_name=TARGET_NAME,
            python_executable=PYTHON_EXECUTABLE,
            extra_cmd_args=["--with-appworld"],
            pre_launch=self.install_appworld,
            use_ray_cluster=multi_nodes,
            enable_ray_for_trinity=not multi_nodes,
        )

    def test_02_begin_trinity(self):
        # get probe target, so as to get timeout settings
        BACKBONE = "trinity"
        TEST_TARGET = "tests/bench/benchmark_appworld/benchmark_appworld_2nodes.yaml"
        PROBE_TARGET = "tests/bench/benchmark_appworld/benchmark_appworld.py->TestProbe"
        TARGET_NAME = f"benchmark_appworld_{BACKBONE}"
        PYTHON_EXECUTABLE = ".venv/bin/python"
        multi_nodes = True

        self.execute_benchmark(
            backbone=BACKBONE,
            test_target=TEST_TARGET,
            probe_target=PROBE_TARGET,
            target_name=TARGET_NAME,
            python_executable=PYTHON_EXECUTABLE,
            extra_cmd_args=["--with-appworld"],
            pre_launch=self.install_appworld,
            use_ray_cluster=multi_nodes,
            enable_ray_for_trinity=not multi_nodes,
        )

    def clear_system_processes(self):
        # kill all python + ray + vllm processes
        from ajet.utils.cleaner import fast_kill_by_keyword_bash

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
