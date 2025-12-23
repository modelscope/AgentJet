import unittest

from tests.bench.benchmark_base import BenchmarkTestCase


class TestBenchmarkFrozenLake(BenchmarkTestCase):
    def test_01_begin_trinity(self):
        BACKBONE = "trinity"
        TEST_TARGET = "tests/bench/benchmark_frozenlake/benchmark_frozenlake.yaml"
        PROBE_TARGET = "tests/bench/benchmark_frozenlake/benchmark_frozenlake.py->TestProbe"
        TARGET_NAME = f"benchmark_frozenlake_{BACKBONE}"
        PYTHON_EXECUTABLE = ".venv/bin/python"
        self.execute_benchmark(
            backbone=BACKBONE,
            test_target=TEST_TARGET,
            probe_target=PROBE_TARGET,
            target_name=TARGET_NAME,
            python_executable=PYTHON_EXECUTABLE,
        )

    def test_02_begin_verl(self):
        BACKBONE = "verl"
        TEST_TARGET = "tests/bench/benchmark_frozenlake/benchmark_frozenlake.yaml"
        PROBE_TARGET = "tests/bench/benchmark_frozenlake/benchmark_frozenlake.py->TestProbe"
        TARGET_NAME = f"benchmark_frozenlake_{BACKBONE}"
        PYTHON_EXECUTABLE = ".verl/bin/python"
        self.execute_benchmark(
            backbone=BACKBONE,
            test_target=TEST_TARGET,
            probe_target=PROBE_TARGET,
            target_name=TARGET_NAME,
            python_executable=PYTHON_EXECUTABLE,
        )
