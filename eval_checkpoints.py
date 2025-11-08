import os
import sys
import glob
from agentopia.utils.smart_daemon import LaunchCommandWhenAbsent

env_path = [
    "appworld-bz32-tp4-linear-new@0",
    # "appworld-bz32-tp4-linear-think@100",
    # "appworld-qwen3-bsz32-tp4-slidingwindow-max_prompt_length++@40",
    # "appworld-qwen3-contextclip-bz32-tp4-v5@100",
]

# scan './launcher' for percise location of each env
def scan_launcher_for_path(exp, base_dir="./launcher", max_depth=2):
    # exp = "appworld-bz32-tp4-linear-think"
    # return "launcher/appworld-linear-think/appworld-bz32-tp4-linear-think"
    exp = exp.split("@")[0]
    launcher_dir = base_dir
    base_depth = os.path.abspath(launcher_dir).count(os.sep)
    for root, dirs, files in os.walk(launcher_dir):
        current_depth = os.path.abspath(root).count(os.sep)
        if current_depth - base_depth >= max_depth:
            # Prevent os.walk from going deeper
            dirs[:] = []
            continue
        if exp in dirs:
            return os.path.join(root, exp)
    raise ValueError(f"Environment {exp} not found in launcher directory.")

env_path_percise = [
    {
        "launcher": scan_launcher_for_path(p, base_dir="./launcher"),
        "checkpoint": scan_launcher_for_path(p, base_dir="./checkpoints"),
        "step": None if "@" not in p else int(p.split("@")[-1])
    } for p in env_path
]

def convert_model(fsdp_checkpoint_dir):
    if os.path.exists(fsdp_checkpoint_dir + "/hf"):
        print("Merged model already exists, skipping conversion.")
        return fsdp_checkpoint_dir + "/hf"
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", fsdp_checkpoint_dir,
        "--target_dir", fsdp_checkpoint_dir + "/hf"
    ]
    import subprocess
    subprocess.run(cmd, check=True)
    if os.path.exists(fsdp_checkpoint_dir + "/hf"):
        return fsdp_checkpoint_dir + "/hf"
    else:
        raise RuntimeError(f"Model conversion failed for {fsdp_checkpoint_dir}")

def vllm_companion_launch(hf_model_path, model_name):
    print("Launching companion process for async LLM server...")
    model_path = hf_model_path
    tensor_parallel_size = 8
    gpu_memory_utilization = 0.95
    max_num_seqs = 12 * tensor_parallel_size
    max_model_len = 25000
    seed = 12345
    port = 18000
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[
            sys.executable, "-m",
            f"vllm.entrypoints.cli.main",
            f"serve", f"{model_path}",
            f"--tensor-parallel-size", f"{tensor_parallel_size}",
            f"--dtype", f"auto",
            f"--enforce-eager",
            f"--gpu-memory-utilization", f"{gpu_memory_utilization}",
            f"--disable-custom-all-reduce",
            f"--max-num-seqs", f"{max_num_seqs}",
            f"--max-model-len", f"{max_model_len}",
            f"--load-format", "auto",
            f"--enable-chunked-prefill",
            f"--enable-prefix-caching",
            f"--served-model-name", f"{model_name}",
            f"--seed", f"{seed}",
            f"--port", f"{port}",
        ],
        dir='./',
        tag="external_vllm_server"
    )
    companion.launch(
        launch_wait_time=1800,
        success_std_string="Application startup complete",
        env_dict={**os.environ}
    )
    return companion

def eval_companion_launch(launcher_dir, launcher_yaml_path, hf_model_path):
    print("Launching companion process for async LLM server...")
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[
            sys.executable, "rollout_eval.py",
            "--config-path", launcher_dir,
            "--config-name", os.path.basename(launcher_yaml_path),
            f"trainer.hfmodelpath={hf_model_path}"
        ],
        dir='./',
        tag="eval_companion"
    )
    companion.launch(
        launch_wait_time=36000,
        success_std_string="all eval task done",
        env_dict={**os.environ}
    )
    return companion

for p in env_path_percise:
    base_checkpoint_dir = p["checkpoint"]
    launcher_dir = p["launcher"]
    step = p["step"]
    launcher_yaml_path = os.path.join(launcher_dir, "yaml_backup.yaml")
    # Read yaml and get field actor_rollout_ref.model.path
    import yaml
    with open(launcher_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    model_name = None
    try:
        model_name = yaml_data['actor_rollout_ref']['model']['path']
        print(f"actor_rollout_ref.model.path: {model_name}")
    except Exception as e:
        print(f"Failed to get actor_rollout_ref.model.path from {launcher_yaml_path}: {e}")

    # scan for all checkpoints in the directory (global_step_*) using glob
    checkpoint_paths = glob.glob(os.path.join(base_checkpoint_dir, "global_step_*", "actor"))
    if (step is not None) and (step != 0):
        checkpoint_paths = glob.glob(os.path.join(base_checkpoint_dir, f"global_step_{step}", "actor"))

    if step == 0: # preserve only one
        checkpoint_paths = [checkpoint_paths[0]]
        use_raw_model = True
    else:
        use_raw_model = False

    for cp in checkpoint_paths:
        if use_raw_model:
            hf_model_path = model_name
        else:
            print(f"Converting model at {cp}")
            hf_model_path = convert_model(cp)
            print(f"Converted model saved at {hf_model_path}")

        vllm_companion = vllm_companion_launch(hf_model_path, model_name)
        eval_companion_launch(launcher_dir, launcher_yaml_path, hf_model_path)
        vllm_companion.shutdown()

        try:
            killer = LaunchCommandWhenAbsent(
                full_argument_list=[
                    f"pkill -f vllm"
                ],
                dir='./',
                tag="killer",
                use_pty=True
            )
            killer.launch(
                launch_wait_time=10,
                success_std_string="what a good hunt!",
                force_restart=True,
            )
        except Exception as e:
            ...


# [do not edit this line]
# killer ray && killer python && python launcher.py --with-appworld && python eval_checkpoints.py
