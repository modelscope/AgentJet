Note: `tests/bench` source code is for test robot only, therefore `yaml` configurations will contain dataset files stored in benchmarking-docker-image.

- To get these dataset files, please refer to `tutorial/*`.

- Benchmarking-docker-image for test-robot will be released in 2026 Feb.

## Running Command

```python
python -m pytest -s tests/bench/benchmark_math/execute_benchmark_math.py
python -m pytest -s tests/bench/benchmark_learn2ask/execute_benchmark_learn2ask.py
python -m pytest -s tests/bench/benchmark_frozenlake/execute_benchmark_frozenlake.py
python -m pytest -s tests/bench/benchmark_countdown/execute_benchmark_countdown.py
python -m pytest -s tests/bench/benchmark_appworld/execute_benchmark_appworld.py

python -m pytest -s tests/bench/benchmark_math/execute_benchmark_math.py::TestBenchmarkMath::test_02_begin_trinity
```
