import ray
import os
from trinity.cli.launcher import main
from trinity.common.config import Config
from trinity.explorer.explorer import Explorer
from trinity.trainer.trainer import Trainer

from ajet.utils.config_utils import read_ajet_config_with_cache
from ajet.utils.core_env_vars import get_runtime_env
from ajet.utils.launch_utils import set_loguru_default_color


set_loguru_default_color()


def get_ajet_config_from_trinity_side():
    yaml_path = os.environ.get("AJET_CONFIG_REDIRECT", None)
    if yaml_path is None:
        raise ValueError("AJET_CONFIG_REDIRECT is not set in environment variables")
    ajet_config = read_ajet_config_with_cache(yaml_path)
    return ajet_config


def patch_runtime_env_to_get_actor():
    """Patch the classmethod of Explorer and Trainer to pass in the runtime env."""
    ajet_config = get_ajet_config_from_trinity_side()
    runtime_env = get_runtime_env(ajet_config, is_trinity=True)
    os.environ.update(runtime_env["env_vars"])

    def patched_explorer_get_actor(cls, config: Config):
        return (
            ray.remote(cls)
            .options(
                name=config.explorer.name,
                namespace=ray.get_runtime_context().namespace,
                runtime_env=runtime_env,
            )
            .remote(config)
        )

    def patched_trainer_get_actor(cls, config: Config):
        return (
            ray.remote(cls)
            .options(
                name=config.trainer.name,
                namespace=ray.get_runtime_context().namespace,
                runtime_env=runtime_env,
            )
            .remote(config)
        )

    Explorer.get_actor = classmethod(patched_explorer_get_actor)
    Trainer.get_actor = classmethod(patched_trainer_get_actor)

    if ajet_config.ajet.enable_experimental_interchange_server:
        from ajet.tuner_lib.weight_tuner.experimental.as_oai_model_server import start_interchange_server
        start_interchange_server(ajet_config)


if __name__ == "__main__":
    patch_runtime_env_to_get_actor()
    main()
