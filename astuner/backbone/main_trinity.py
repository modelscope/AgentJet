import ray
from trinity.cli.launcher import main
from trinity.common.config import Config
from trinity.explorer.explorer import Explorer
from trinity.trainer.trainer import Trainer

import astuner.backbone.trinity_trainer  # noqa: F401
from astuner.utils.core_env_vars import get_runtime_env


def patch_runtime_env_to_get_actor():
    """Patch the classmethod of Explorer and Trainer to pass in the runtime env."""
    runtime_env = get_runtime_env(is_trinity=True)

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


if __name__ == "__main__":
    patch_runtime_env_to_get_actor()
    main()
