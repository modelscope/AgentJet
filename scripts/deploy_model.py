import argparse
import os
import sys

import torch

sys.path.append(os.getcwd())
from astuner.utils.smart_daemon import LaunchCommandWhenAbsent

parser = argparse.ArgumentParser(description="deploy Hugging Face model")
parser.add_argument(
    "--target",
    default="/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen3-30B-A3B-Instruct-2507",
    type=str,
    help="要下载的数据集仓库名称",
)
parser.add_argument("--port", default="2888", type=str, help="端口")
args = parser.parse_args()


def companion_launch():
    print("Launching companion process for async LLM server...")
    model_path = args.target
    n_avail_gpus = torch.cuda.device_count()
    tensor_parallel_size = n_avail_gpus
    if tensor_parallel_size > n_avail_gpus:
        print(
            f"Warning: tensor_parallel_size {tensor_parallel_size} is greater than available GPUs {n_avail_gpus}. Setting tensor_parallel_size to {n_avail_gpus}."
        )
        tensor_parallel_size = n_avail_gpus
    gpu_memory_utilization = 0.95
    # max_num_seqs = config.actor_rollout_ref.rollout.max_num_seqs
    # max_model_len = config.astuner.rollout.max_model_len
    # seed = config.astuner.debug.debug_vllm_seed
    # vllm_port = config.astuner.debug.debug_vllm_port
    vllm_port = args.port
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            f"{model_path}",
            "--tensor-parallel-size",
            f"{tensor_parallel_size}",
            "--dtype",
            "auto",
            "--enforce-eager",
            "--gpu-memory-utilization",
            f"{gpu_memory_utilization}",
            "--disable-custom-all-reduce",
            # f"--max-num-seqs", f"{max_num_seqs}",
            # f"--max-model-len", f"{max_model_len}",
            "--load-format",
            "auto",
            "--enable-chunked-prefill",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--enable-prefix-caching",
            # f"--seed", f"{seed}",
            "--port",
            f"{vllm_port}",
        ],
        dir="./",
        tag="api_vllm_server",
    )
    companion.launch(
        launch_wait_time=1800,
        success_std_string="Application startup complete",
        env_dict={**os.environ},
    )


companion_launch()
