"""Programmatic training entry point for AgentJet.

This class mirrors the CLI launcher by materializing a YAML config and
spawning a subprocess to run the existing training pipeline. The goal is to
keep the public surface minimal while reusing the mature CLI code paths.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Union

import ray
import yaml
from loguru import logger

from ajet.launcher import (
    check_avail_gpu,
    get_backbone_target,
    setup_environment_vars,
)
from ajet.default_config.ajet_default import Config
from ajet.utils.config_utils import (
    expand_ajet_hierarchical_config,
    prepare_experiment_config,
    read_ajet_hierarchical_config,
)
from ajet.utils.dynamic_import import cls_to_path
from ajet.utils.launch_utils import execute_training_process


class AgentJetJob:
    """Lightweight builder that launches AgentJet training as a subprocess."""

    def __init__(
        self,
        backbone: str = "verl",
        model: str = "Qwen/Qwen2___5-7B-Instruct",
        n_gpu: int = 8,
        algorithm: str = "grpo",
        n_gpu_for_infer: int | None = None, # only for trinity backbone
        grpo_n: int = 8,
        tinkerscript_mode: bool = True,
        *kwargs,
    ) -> None:
        self.backbone = backbone
        if tinkerscript_mode:
            default_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "default_config/ajet_ts_default.yaml"))
        else:
            default_yaml = None
        self.config_as_dict: dict = self.build_job_from_yaml(default_yaml)
        self.config = Config.update_from_dict_recursive(Config(), self.config_as_dict)

        self.config.ajet.backbone = backbone
        self.config.ajet.model.path = model
        self.config.ajet.trainer_common.n_gpus_per_node = n_gpu
        self.config.ajet.trainer_common.algorithm.adv_estimator = algorithm
        self.config.ajet.rollout.num_repeat = grpo_n
        if n_gpu_for_infer is None and backbone == "trinity":
            raise ValueError("Please specify `n_gpu_for_infer` (n_gpu_for_infer < n_gpu) for trinity backbone.")
        if (n_gpu_for_infer is not None) and backbone == "verl":
            raise ValueError("n_gpu_for_infer is only for trinity backbone, please set it to `None`.")
        else:
            if backbone == "trinity":
                assert isinstance(n_gpu_for_infer, int), f"`n_gpu_for_infer` should be int, got {type(n_gpu_for_infer)}."
                assert n_gpu_for_infer < n_gpu, "`n_gpu_for_infer` should be less than `n_gpu`."
                self.config.ajet.rollout.n_vllm_engine = n_gpu_for_infer
                self.config.ajet.rollout.tensor_model_parallel_size = 1

    def build_job_from_yaml(self, yaml_path: str | None) -> dict:
        self.exp_name = datetime.now().strftime("ajet_job_%Y%m%d_%H%M%S")
        self.exp_dir_final = "saved_experiments"
        self.config_as_dict = read_ajet_hierarchical_config(
            yaml_path,
            exp_name=self.exp_name,
            backbone=self.backbone,
            write_to=None,
            exp_dir=self.exp_dir_final,
        )
        self.config_as_dict = expand_ajet_hierarchical_config(self.config_as_dict, write_to=None)
        logger.info(f"Built AgentJet job config: {yaml_path}")
        return self.config_as_dict

    def dump_job_as_yaml(self, yaml_path: str) -> str:
        if os.path.dirname(yaml_path):
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config.to_dict(), f, sort_keys=False)
        logger.info(f"Saved training config to {yaml_path}")
        return yaml_path

    def set_workflow(
        self, workflow: Union[str, Callable[..., Any]], ensure_reward_in_workflow: bool = False
    ) -> "AgentJetJob":
        self.config.ajet.rollout.user_workflow = cls_to_path(workflow)
        # TODO: validate workflow outputs contain reward
        # ensure_reward_in_workflow
        return self

    def set_data(
        self,
        type: str,
        dataset_path: str,
        training_split: str = "train",
        validation_split: str = "test",
    ) -> "AgentJetJob":
        """Configure the task reader. Defaults to HuggingFace datasets."""

        # available types:
        # `env_service` or `jsonl_dataset_file` or `huggingface_dat_repo` or `data_generation` or `random_dummy`

        if type in {"hf", "huggingface", "huggingface_dat_repo"}:
            self.config.ajet.task_reader.type = "huggingface_dat_repo"
            self.config.ajet.task_reader.huggingface_dat_repo.dataset_path = dataset_path
            self.config.ajet.task_reader.huggingface_dat_repo.training_split = training_split
            self.config.ajet.task_reader.huggingface_dat_repo.validation_split = validation_split
        elif type in {"random_dummy", "dummy"}:
            self.config.ajet.task_reader.type = "random_dummy"
        else:
            raise NotImplementedError(
                f"Please edit yaml to directly set up task reader of type {type}."
            )

        return self

    def tune(self, *args, **kwargs) -> "AgentJetJob":
        ast_cfg = self.config.ajet
        if not ast_cfg.rollout or not ast_cfg.rollout.user_workflow:
            raise ValueError("Workflow must be set via set_workflow before tuning.")
        if not ast_cfg.task_reader:
            raise ValueError("Data source must be set via set_data before tuning.")

        backbone = self.config.ajet.backbone
        exp_dir = self.config.ajet.experiment_dir

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml") as temp_yaml:
            yaml_path = temp_yaml.name
            self.dump_job_as_yaml(yaml_path)
            args = SimpleNamespace(
                conf=yaml_path,
                backbone=backbone,
                exp_dir=exp_dir,
                with_logview=False,
                debug=False,
            )

            if args.backbone != "debug":
                # Enforce GPU availability and free memory threshold before proceeding
                check_avail_gpu(min_free_ratio=0.95)

            # finalize experiment config
            main_yaml_fp, exe_exp_base, exp_name, exp_config = prepare_experiment_config(
                yaml_path, exp_dir, backbone
            )

        # setup environment variables for ray
        env = setup_environment_vars(args, exp_config, main_yaml_fp)

        # start ray if not already started
        if not ray.is_initialized():
            from ajet.utils.launch_utils import start_ray_service

            start_ray_service(args, env)
        else:
            raise RuntimeError(
                "Ray is already initialized. Please shutdown existing Ray instance before starting a new tuning job."
            )

        # start training process
        if args.conf and main_yaml_fp and exe_exp_base and exp_config:
            execute_training_process(
                args,
                get_backbone_target(args.backbone),
                main_yaml_fp,
                exe_exp_base,
                main_yaml_fp,
                env,
                exp_config,
            )

        return self
