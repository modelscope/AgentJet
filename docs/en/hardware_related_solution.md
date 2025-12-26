This document records a list of **Hardware Related** issues for future reference.

## 1. ncclUnhandledCudaError: Call to CUDA function failed.

- Problem:

    ```python
    File "/root/agentscope-tuner/.venv/lib/python3.10/site-packages/torch/distributed/utils.py", line 322, in _sync_params_and_buffers
    dist._broadcast_coalesced(
    torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/NCCLUtils.cpp:77, unhandled cuda error (run with NCCL_DEBUG=INFO for details), NCCL version 2.26.2
    ncclUnhandledCudaError: Call to CUDA function failed.
    Last error:
    Cuda failure 1 'invalid argument'
    ```

- Solution:

    ```bash
    export NCCL_NVLS_ENABLE=0
    ```