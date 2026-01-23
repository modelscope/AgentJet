0. test docker installation

```
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia-smi
```


1. Run `docker build -f scripts/docker/dockerfile -t ajet:latest .` to build docker container

2. By the way, for users in China, using the alternative dockerfile script `docker build -f scripts/docker/dockerfile_zh -t ajet:latest .` can optimize the download speed using alibaba public cloud.

3. To run build-in tests, please follow instructions to mount test models and datasets.
    - Download model manually, or use the helper script `python ./scripts/download_model.py`.
    - For example, if your model is in `./modelscope_cache/Qwen/Qwen2___5-14B-Instruct`.
    - and your dataset is in `benchmark_datasets/dataset`
    - run
    ```
docker run -it --gpus all --shm-size="64g" --rm \
    -v "$(pwd)/modelscope_cache/Qwen/Qwen2___5-14B-Instruct:/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-7B-Instruct" \
    -v "$(pwd)/benchmark_datasets/dataset:/mnt/data_cpfs/model_cache/modelscope/dataset" \
    -e VERL_PYTHON="./.venv/bin/python" \
    -e NCCL_NVLS_ENABLE=0 \
    ajet:latest \
    python -m pytest -s tests/bench/benchmark_math/execute_benchmark_math.py::TestBenchmarkMath::test_01_begin_verl
    ```
