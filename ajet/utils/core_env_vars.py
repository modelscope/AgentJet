import os
import copy
from pathlib import Path

from beast_logger import print_dict
from dotenv import load_dotenv
from ajet.utils.networking import find_free_port, get_host_ip


def _discover_nvidia_lib_dirs() -> list:
    """Return lib dirs of installed nvidia-* pip wheels (e.g. .../nvidia/cu13/lib,
    .../nvidia/cuda_nvrtc/lib). These ship versioned CUDA .so files that vLLM's
    cumem_allocator needs but that carry no RUNPATH, so they must be on
    LD_LIBRARY_PATH. Discovered from the running interpreter's site-packages so
    the result is correct regardless of how the launcher set up its environment."""
    import site
    import sysconfig

    roots = set()
    for getter in (lambda: [sysconfig.get_paths().get("purelib")],
                   lambda: site.getsitepackages() if hasattr(site, "getsitepackages") else []):
        try:
            for p in getter():
                if p:
                    roots.add(p)
        except Exception:
            pass

    lib_dirs = []
    for root in roots:
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir() and str(lib_dir) not in lib_dirs:
                lib_dirs.append(str(lib_dir))
    return lib_dirs


def get_runtime_env(config, is_trinity: bool = False) -> dict:
    if os.path.exists(".env"):
        load_dotenv(".env")

    master_node_ip = get_host_ip(os.environ.get("NETWORK_INTERFACE", None))
    if config.ajet.trainer_common.nnodes == 1:
        master_node_ip = "localhost"
    else:
        if config.ajet.enable_interchange_server:
            if config.ajet.interchange_server.interchange_method == "ipc":
                raise ValueError("IPC interchange method is not supported for multi-node setup. Please set `ajet.interchange_server.interchange_method: tcp` ")

    if config.ajet.interchange_server.interchange_server_port != 'auto':
        data_interchange_port = str(int(config.ajet.interchange_server.interchange_server_port))
    else:
        data_interchange_port = str(find_free_port())

    runtime_env = {
        "env_vars": {
            "NCCL_DEBUG": "WARN",

            "VLLM_USE_V1": "1",
            "VLLM_LOGGING_LEVEL": "WARN",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            "VLLM_DISABLE_COMPILE_CACHE": "1",

            "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
            "HCCL_NPU_SOCKET_PORT_RANGE": "auto",

            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "TOKENIZERS_PARALLELISM": "true",
            # use ajet.backbone as plugin directory
            "TRINITY_PLUGIN_DIRS": str((Path(__file__).parent.parent / "backbone").resolve()),
            # "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            "SWANLAB_API_KEY": os.getenv("SWANLAB_API_KEY", ""),
            "SWANLAB_LOG_DIR": os.getenv("SWANLAB_LOG_DIR", "saved_experiments/swanlog"),
            "AJET_CONFIG_REDIRECT": os.getenv("AJET_CONFIG_REDIRECT", ""),
            "AJET_DAT_INTERCHANGE_PORT": os.getenv("AJET_DAT_INTERCHANGE_PORT", data_interchange_port),
            "MASTER_NODE_IP": os.getenv("MASTER_NODE_IP", master_node_ip),
        }
    }

    optional_env_vars = [
        "RAY_record_task_actor_creation_sites",
        "BEST_LOGGER_WEB_SERVICE_URL",
        "AJET_GIT_HASH",
        "AJET_REQ_TXT",
        "SWANLAB_WEB_HOST",
        "SWANLAB_API_HOST",
        "AJET_BENCHMARK_NAME",
        "FINANCE_MCP_URL",
        # one-shot training-side token-id capture (tutorial/claudecode_geo3k_swarm)
        "AJET_CAPTURE_TOKEN_IDS",
        # API Keys for RM Gallery and other services
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "API_KEY",
        "BASE_URL",
        "AGENTJET_FIND_MAX_PPO_TOKEN_LEN",
        "AGENTJET_FIND_MAX_START",
        "AGENTJET_FIND_MAX_CAP",
        "AGENTJET_FIND_MAX_TOL",
        "AGENTJET_FIND_MAX_BUDGET_S",
        "AGENTJET_FIND_MAX_SEQ",
    ]

    for var in optional_env_vars:
        if os.getenv(var):
            runtime_env["env_vars"].update({var: os.getenv(var, "")})

    # Propagate the CUDA toolkit / loader paths from the driver process to all
    # Ray actors on every node. Without this, actors on remote nodes inherit
    # only the raylet's env and may fail to load versioned CUDA .so files
    # (e.g. vLLM's cumem_allocator needs libnvrtc.so.13 for sleep-mode weight
    # sync). These are only forwarded when set in the driver environment.
    for var in ("LD_LIBRARY_PATH", "CUDA_HOME", "PATH"):
        if os.getenv(var):
            runtime_env["env_vars"].update({var: os.getenv(var, "")})

    # Ensure the CUDA runtime libs shipped as nvidia-* pip wheels are on the
    # loader path. vLLM's cumem_allocator.abi3.so links libnvrtc.so.13 with no
    # RUNPATH, so it can only be found via LD_LIBRARY_PATH. The wheels install
    # under site-packages/nvidia/<component>/lib; discover and prepend them so
    # sleep-mode weight sync works without relying on the operator's env.
    nvidia_lib_dirs = _discover_nvidia_lib_dirs()
    if nvidia_lib_dirs:
        existing = runtime_env["env_vars"].get("LD_LIBRARY_PATH", os.getenv("LD_LIBRARY_PATH", ""))
        parts = nvidia_lib_dirs + ([existing] if existing else [])
        runtime_env["env_vars"]["LD_LIBRARY_PATH"] = os.pathsep.join(parts)

    if is_trinity:
        assert "AJET_CONFIG_REDIRECT" in runtime_env["env_vars"]

    print_env_dict = copy.deepcopy(runtime_env["env_vars"])
    # limit value length for printing
    for k, v in print_env_dict.items():
        _len_limit = 500
        _len_limit_half = _len_limit // 2
        if len(v) > _len_limit:
            print_env_dict[k] = v[:_len_limit_half] + "..." + v[-_len_limit_half:]
    print_dict(print_env_dict, "runtime_env")
    return runtime_env
