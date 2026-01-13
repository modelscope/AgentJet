import os

from tests.bench.benchmark_base import BenchmarkTestCase


class TestBenchmarkLearnToAsk(BenchmarkTestCase):

    def test_01_begin_verl(self):
        # get probe target, so as to get timeout settings
        BACKBONE = "verl"
        TEST_TARGET = "tests/bench/benchmark_learn2ask/benchmark_learn2ask.yaml"
        PROBE_TARGET = "tests/bench/benchmark_learn2ask/benchmark_learn2ask.py->TestProbe"
        TARGET_NAME = f"benchmark_learn2ask_{BACKBONE}"
        PYTHON_EXECUTABLE = os.environ.get("VERL_PYTHON", ".verl/bin/python")

        self.execute_benchmark(
            backbone=BACKBONE,
            test_target=TEST_TARGET,
            probe_target=PROBE_TARGET,
            target_name=TARGET_NAME,
            python_executable=PYTHON_EXECUTABLE,
        )

    def test_02_begin_trinity(self):
        # get probe target, so as to get timeout settings
        BACKBONE = "trinity"
        TEST_TARGET = "tests/bench/benchmark_learn2ask/benchmark_learn2ask.yaml"
        PROBE_TARGET = "tests/bench/benchmark_learn2ask/benchmark_learn2ask.py->TestProbe"
        TARGET_NAME = f"benchmark_learn2ask_{BACKBONE}"
        PYTHON_EXECUTABLE = os.environ.get("TRINITY_PYTHON", ".venv/bin/python")

        self.execute_benchmark(
            backbone=BACKBONE,
            test_target=TEST_TARGET,
            probe_target=PROBE_TARGET,
            target_name=TARGET_NAME,
            python_executable=PYTHON_EXECUTABLE,
        )
