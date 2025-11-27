import os
from dotenv import load_dotenv
from beast_logger import print_dict

def get_runtime_env(is_trinity: bool = False) -> dict:

    if os.path.exists(".env"):
        load_dotenv(".env")

    runtime_env = {
        "env_vars": {
            "VLLM_USE_V1": "1",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            "SWANLAB_API_KEY": os.getenv("SWANLAB_API_KEY", ""),
            "ASTUNE_CONFIG_REDIRECT": os.getenv("ASTUNE_CONFIG_REDIRECT", ""),
        }
    }
    if os.getenv("RAY_record_task_actor_creation_sites"):
        runtime_env["env_vars"].update(
            {
                "RAY_record_task_actor_creation_sites": os.getenv(
                    "RAY_record_task_actor_creation_sites", ""
                ),
            }
        )
    if os.getenv("BEST_LOGGER_WEB_SERVICE_URL"):
        runtime_env["env_vars"].update(
            {
                "BEST_LOGGER_WEB_SERVICE_URL": os.getenv(
                    "BEST_LOGGER_WEB_SERVICE_URL", ""
                ),
            }
        )

    if is_trinity:
        assert "ASTUNE_CONFIG_REDIRECT" in runtime_env["env_vars"]

    print_dict(runtime_env["env_vars"], "runtime_env")
    return runtime_env