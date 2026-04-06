# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import atexit
import os
import socket

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from verl.utils.device import is_cuda_available
from verl.utils.dataset.rl_dataset import collate_fn
from torch.utils.data import Dataset as TorchDataset

# Create training and validation datasets.
from ajet.backbone.warm_up import warm_up_process
from ajet.task_reader import RouterTaskReader, task_to_standard_dataset
from ajet.utils.process_dataset import create_rl_sampler
from ajet.utils.core_env_vars import get_runtime_env
from ajet.utils.launch_utils import set_loguru_default_color

set_loguru_default_color()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config: DictConfig) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # this is for local ray cluster
        runtime_env = get_runtime_env(config)
        ray.init(
            runtime_env=runtime_env,
        )

    def on_shutdown():
        if ray.is_initialized():
            ray.shutdown()
        if config.ajet.enable_interchange_server:
            if config.ajet.enable_swarm_mode:
                from ajet.tuner_lib.experimental.interchange_utils import http_change_engine_status
                print("Changing engine status to OFFLINE before shutdown...")
                http_change_engine_status(config, "ENGINE.OFFLINE", global_step=0)

    atexit.register(on_shutdown)  # ray shutdown on exit

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.trainer.get("profile_steps") is not None
        and len(config.trainer.get("profile_steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert (
            is_nvtx_available()
        ), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from loguru import logger
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        warm_up_process(config)

        logger.info(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.ajet.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from ajet.backbone.verl import AjetActorRolloutRefWorker
            from ajet.backbone.verl import AjetAsyncActorRolloutRefWorker



            ActorRolloutRefWorker = AjetActorRolloutRefWorker
            actor_rollout_cls = AjetAsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import (
                ActorRolloutRefWorker,
                AjetAsyncActorRolloutRefWorker,
            )

            actor_rollout_cls = AjetAsyncActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }


        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id


        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        task_reader = RouterTaskReader(
            config.ajet.task_reader.type,
            config.ajet.task_reader,
        )

        train_dataset: TorchDataset = task_to_standard_dataset(task_reader.generate_training_tasks)  # type: ignore
        val_dataset: TorchDataset = task_to_standard_dataset(task_reader.generate_validation_tasks)  # type: ignore
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from ajet.backbone.trainer_verl import AjetRayPPOTrainer

        if config.ajet.enable_interchange_server:
            from ajet.tuner_lib.experimental.oai_model_server import start_interchange_server
            start_interchange_server(config)

        # Initialize the PPO trainer.
        trainer = AjetRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


if __name__ == "__main__":
    main()
