import os
import unittest

from tests.bench.benchmark_base import BenchmarkTestCase




class TestBenchmarkCountdown(BenchmarkTestCase, unittest.TestCase):

    def test_01_begin_verl(self):
        BACKBONE = "verl"
        TEST_TARGET = "tests/bench/benchmark_countdown/benchmark_countdown.yaml"
        PROBE_TARGET = "tests/bench/benchmark_countdown/benchmark_countdown.py->TestProbe"
        TARGET_NAME = f"benchmark_countdown_{BACKBONE}"
        PYTHON_EXECUTABLE = os.environ.get("VERL_PYTHON", ".verl/bin/python")

        self.execute_benchmark(
            backbone=BACKBONE,
            test_target=TEST_TARGET,
            probe_target=PROBE_TARGET,
            target_name=TARGET_NAME,
            python_executable=PYTHON_EXECUTABLE,
        )

    def test_02_begin_trinity(self):
        BACKBONE = "trinity"
        TEST_TARGET = "tests/bench/benchmark_countdown/benchmark_countdown.yaml"
        PROBE_TARGET = "tests/bench/benchmark_countdown/benchmark_countdown.py->TestProbe"
        TARGET_NAME = f"benchmark_countdown_{BACKBONE}"
        PYTHON_EXECUTABLE = os.environ.get("TRINITY_PYTHON", ".venv/bin/python")

        self.execute_benchmark(
            backbone=BACKBONE,
            test_target=TEST_TARGET,
            probe_target=PROBE_TARGET,
            target_name=TARGET_NAME,
            python_executable=PYTHON_EXECUTABLE,
        )
