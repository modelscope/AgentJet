
To build the test image with dataset, please follow the instruction below.

1. test docker installation:
```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia-smi
```

2. Run the following command to build docker container:
```bash
docker build -f scripts/docker/dockerfile -t ajet:latest .
```

3. By the way, for users in China, using the alternative dockerfile script can optimize the download speed using alibaba public cloud.
```bash
docker build -f scripts/docker/dockerfile_zh -t ajet:latest .
```

4. To run build-in tests, please follow instructions to mount test models and datasets.
- Download model manually, or use the helper script `python ./scripts/download_model.py`.
- For example, if your model is in `./modelscope_cache/Qwen/Qwen2___5-14B-Instruct`.
- Run the instruction below to run the first training program
    ```bash
    clear && docker run -it --gpus all --shm-size="64g" --rm \
        -v "$(pwd)/modelscope_cache/Qwen/Qwen2___5-7B-Instruct:/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-7B-Instruct" \
        -e SWANLAB_API_KEY="xxxxxxxxxxxxxxxxxx" \
        -e DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx" \
        -e CUDA_VISIBLE_DEVICES="4,5,6,7" \
        -e VERL_PYTHON="/opt/venv/bin/python" \
        -e NCCL_NVLS_ENABLE=0 \
        ajet:latest \
        python -m pytest -s tests/bench/benchmark_math/execute_benchmark_math.py::TestBenchmarkMath::test_01_begin_verl
    ```
