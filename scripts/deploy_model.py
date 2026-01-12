import argparse
import os
import sys

import torch

# Add current directory to path before other imports
sys.path.append(os.getcwd())  # noqa: E402

from loguru import logger  # noqa: E402

from ajet.utils.cleaner import fast_kill_by_keyword_bash  # noqa: E402
from ajet.utils.smart_daemon import LaunchCommandWhenAbsent  # noqa: E402

parser = argparse.ArgumentParser(description="deploy Hugging Face model")
parser.add_argument(
    "--target",
    # default="/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen3-235B-A22B-Instruct-2507/",
    default="/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen3-Coder-480B-A35B-Instruct",
    type=str,
    help="Model path",
)
parser.add_argument(
    "--alias",
    default="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    type=str,
    help="Model alias",
)
parser.add_argument(
    "--kill",
    default="",
    type=str,
    help="Keywords to kill related processes, separated by |",
)
parser.add_argument(
    "--autokill",
    default=False,
    action="store_true",
    help="Automatically kill related processes",
)
parser.add_argument("--port", default="2888", type=str, help="Port number")
args = parser.parse_args()

if args.autokill:
    args.kill = "ray|vllm|VLLM|python"

# Handle kill-keywords argument if provided
if args.kill:
    logger.info(f"Killing processes matching keywords: {args.kill}")
    for keyword in args.kill.split("|"):
        logger.info(f"Killing processes matching keyword: {keyword}")
        killed_pids = fast_kill_by_keyword_bash(keyword)
        if killed_pids:
            logger.success(f"Successfully killed processes with PIDs: {killed_pids}")
        else:
            logger.warning(f"No processes found matching keyword: {keyword}")


def companion_launch():
    logger.info("Launching companion process for async LLM server...")
    model_path = args.target
    n_avail_gpus = torch.cuda.device_count()
    tensor_parallel_size = n_avail_gpus
    if tensor_parallel_size > n_avail_gpus:
        logger.warning(
            f"Warning: tensor_parallel_size {tensor_parallel_size} is greater than available GPUs {n_avail_gpus}. Setting tensor_parallel_size to {n_avail_gpus}."
        )
        tensor_parallel_size = n_avail_gpus

    # gpu_memory_utilization = 0.95
    # max_num_seqs = config.actor_rollout_ref.rollout.max_num_seqs
    # max_model_len = config.ajet.rollout.max_model_len
    # seed = config.ajet.debug.debug_vllm_seed
    # vllm_port = config.ajet.debug.debug_vllm_port
    vllm_port = args.port
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            model_path,
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--dtype",
            "auto",
            # "--enforce-eager",
            # "--gpu-memory-utilization", str(gpu_memory_utilization),
            # "--disable-custom-all-reduce",
            # "--max-num-seqs", str(max_num_seqs),
            # "--max-model-len", str(max_model_len),
            "--load-format",
            "auto",
            "--served-model-name",
            args.alias,
            "--enable-chunked-prefill",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--enable-prefix-caching",
            # "--seed", str(seed),
            "--port",
            vllm_port,
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
