import os
import copy
from pathlib import Path

from beast_logger import print_dict
from dotenv import load_dotenv
from ajet.utils.networking import find_free_port, get_host_ip


def get_runtime_env(config, is_trinity: bool = False) -> dict:
    if os.path.exists(".env"):
        load_dotenv(".env")

    master_node_ip = get_host_ip(os.environ.get("NETWORK_INTERFACE", None))
    if config.ajet.trainer_common.nnodes == 1:
        master_node_ip = "localhost"
    else:
        if config.ajet.enable_experimental_interchange_server:
            if config.ajet.interchange_server.interchange_method == "ipc":
                raise ValueError("IPC interchange method is not supported for multi-node setup. Please set `ajet.interchange_server.interchange_method: tcp` ")

    if config.ajet.interchange_server.interchange_server_port != 'auto':
        data_interchange_port = str(int(config.ajet.interchange_server.interchange_server_port))
    else:
        data_interchange_port = str(find_free_port())

    runtime_env = {
        "env_vars": {
            "VLLM_USE_V1": "1",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "TOKENIZERS_PARALLELISM": "true",
            # use ajet.backbone as plugin directory
            "TRINITY_PLUGIN_DIRS": str((Path(__file__).parent.parent / "backbone").resolve()),
            # "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            "SWANLAB_API_KEY": os.getenv("SWANLAB_API_KEY", ""),
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
        "AJET_BENCHMARK_NAME",
        "FINANCE_MCP_URL",
        # API Keys for RM Gallery and other services
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "API_KEY",
        "BASE_URL",
    ]

    for var in optional_env_vars:
        if os.getenv(var):
            runtime_env["env_vars"].update({var: os.getenv(var, "")})

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
